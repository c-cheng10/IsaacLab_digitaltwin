import argparse
import time

import cv2
import numpy as np
import depthai as dai


TAG_SIZE = 0.024
FAMILIES = "tag36h11"


def _rgb_socket():
    for name in ("CAM_A", "RGB"):
        if hasattr(dai.CameraBoardSocket, name):
            return getattr(dai.CameraBoardSocket, name)
    raise RuntimeError("no usable colour camera socket in this depthai build")


def get_detector(nthreads=4):
    from pupil_apriltags import Detector
    return Detector(families=FAMILIES, nthreads=nthreads)


def to_gray(bgr):
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return np.ascontiguousarray(gray)


def detect_tag(detector, bgr, params, tag_size):
    dets = detector.detect(to_gray(bgr), estimate_tag_pose=True,
                           camera_params=params, tag_size=tag_size)
    out = []
    for d in dets:
        pose_R = np.asarray(d.pose_R, dtype=float)
        pose_t = np.asarray(d.pose_t, dtype=float).flatten()
        out.append({
            "tag_id": int(d.tag_id),
            "pose_R": pose_R,
            "pose_t": pose_t,
            "pose_err": float(d.pose_err),
            "distance": float(np.linalg.norm(pose_t)),
            "center_px": np.asarray(d.center, dtype=float),
            "corners_px": np.asarray(d.corners, dtype=float),
            "decision_margin": float(d.decision_margin),
            "hamming": int(d.hamming),
        })
    return out


def print_raw(d, i):
    print(f"  --- raw detection[{i}] ---")
    for attr in ("tag_family", "tag_id", "hamming", "decision_margin", "homography",
                 "center", "corners", "pose_R", "pose_t", "pose_err"):
        if hasattr(d, attr):
            print(f"    {attr} =")
            print("      " + repr(getattr(d, attr)).replace("\n", "\n      "))


def draw_overlay(bgr, dets, params, tag_size, fps):
    fx, fy, cx, cy = params
    h, w = bgr.shape[:2]
    cv2.drawMarker(bgr, (int(cx), int(cy)), (120, 120, 120), cv2.MARKER_CROSS, 14, 1)

    for det in dets:
        pts = det["corners_px"].astype(int)
        cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for n, (px, py) in enumerate(det["corners_px"]):
            cv2.putText(bgr, str(n), (int(px) + 4, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        u, v = det["center_px"]
        cv2.circle(bgr, (int(u), int(v)), 4, (0, 0, 255), -1)

        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (255, 0, 255), 1)

        t = det["pose_t"]
        span = float(np.linalg.norm(det["corners_px"][0] - det["corners_px"][2]))
        lines = [
            f"ID {det['tag_id']}  {det['distance']*1000:.0f} mm",
            f"xyz {t[0]*1000:+.0f} {t[1]*1000:+.0f} {t[2]*1000:+.0f} mm",
            f"err {det['pose_err']:.1e}  margin {det['decision_margin']:.0f}",
            f"span {span:.0f} px",
        ]
        for k, line in enumerate(lines):
            cv2.putText(bgr, line, (x0, max(14, y0 - 8 - 14 * (len(lines) - 1 - k))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    hud = (f"{w}x{h}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  "
           f"tag={tag_size*1000:.1f}mm  {len(dets)} det  {fps:.1f} fps")
    cv2.putText(bgr, hud, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def _is_v3():
    return not hasattr(dai.node, "XLinkOut")


class OakCamera:

    def __init__(self, width, height, fps, focus):
        self.width, self.height, self.fps, self.focus = width, height, fps, focus
        self.device = None
        self.pipeline = None
        self.q = None
        self.api = "v3" if _is_v3() else "v2"

    def __enter__(self):
        sock = _rgb_socket()
        if self.api == "v3":
            self.device = dai.Device()
            self.pipeline = dai.Pipeline(self.device)
            cam = self.pipeline.create(dai.node.Camera).build(sock)
            self._set_focus(cam)
            try:
                out = cam.requestOutput((self.width, self.height),
                                        dai.ImgFrame.Type.BGR888i, fps=float(self.fps))
            except TypeError:
                out = cam.requestOutput((self.width, self.height),
                                        dai.ImgFrame.Type.BGR888i)
            try:
                self.q = out.createOutputQueue(maxSize=4, blocking=False)
            except TypeError:
                self.q = out.createOutputQueue()
            self.pipeline.start()
        else:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(sock)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setPreviewSize(self.width, self.height)
            cam.setPreviewKeepAspectRatio(False)
            cam.setFps(self.fps)
            self._set_focus(cam)
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.preview.link(xout.input)
            self.device = dai.Device(pipeline)
            self.q = self.device.getOutputQueue("rgb", 4, False)
        return self

    def _set_focus(self, cam):
        if self.focus is None:
            return
        try:
            cam.initialControl.setManualFocus(int(self.focus))
        except Exception as e:
            print(f"  [oakd] WARNING: could not lock focus ({e}); running autofocus")

    def calibration(self, sock):
        calib = self.device.readCalibration()
        K = np.array(calib.getCameraIntrinsics(sock, self.width, self.height),
                     dtype=np.float64)
        try:
            dist = np.array(calib.getDistortionCoefficients(sock), dtype=np.float64)
        except Exception:
            dist = np.zeros(8)
        return K, dist

    def get_frame(self):
        pkt = self.q.tryGet()
        return None if pkt is None else pkt.getCvFrame()

    def __exit__(self, *exc):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.device is not None:
                self.device.close()
        except Exception:
            pass
        return False


def main():
    ap = argparse.ArgumentParser(description="OAK-D Lite AF AprilTag live test")
    ap.add_argument("--tag-size", type=float, default=TAG_SIZE,
                    help="black-border width in metres (default 0.024)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--focus", type=int, default=None,
                    help="lock autofocus to 0-255 (recommended: intrinsics assume a fixed focus)")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--raw", action="store_true", help="dump raw detections to stdout")
    ap.add_argument("--save", type=str, default=None, help="write annotated frames to this .avi")
    args = ap.parse_args()

    detector = get_detector()

    with OakCamera(args.width, args.height, args.fps, args.focus) as cam:
        sock = _rgb_socket()
        K, dist = cam.calibration(sock)

        print(f"  [oakd] depthai {dai.__version__} -- using {cam.api} API")
        print(f"  [oakd] factory K @ {args.width}x{args.height}:\n{np.round(K, 3)}")
        print(f"  [oakd] distortion: {np.round(dist[:8], 5).tolist()}")
        print(f"  [oakd] focus: {'AUTO (intrinsics drift as it hunts)' if args.focus is None else args.focus}")

        undistort = not args.no_undistort and np.any(np.abs(dist) > 1e-9)
        if undistort:
            newK, _roi = cv2.getOptimalNewCameraMatrix(
                K, dist, (args.width, args.height), 0, (args.width, args.height))
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, newK, (args.width, args.height), cv2.CV_16SC2)
            Kd = newK
            print(f"  [oakd] undistorting; effective K:\n{np.round(Kd, 3)}")
        else:
            Kd = K
            print("  [oakd] NOT undistorting -- pose accuracy will suffer at frame edges")

        params = (float(Kd[0, 0]), float(Kd[1, 1]), float(Kd[0, 2]), float(Kd[1, 2]))

        writer = None
        if args.save:
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"MJPG"),
                                     args.fps, (args.width, args.height))

        t_prev, fps = time.time(), 0.0
        frame_i = 0
        print("  [oakd] streaming. keys: q/ESC quit, r print raw once, s save png")

        while True:
            bgr = cam.get_frame()
            if bgr is None:
                if cv2.waitKey(1) in (27, ord("q")):
                    break
                continue

            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            if bgr.shape[1] != args.width or bgr.shape[0] != args.height:
                bgr = cv2.resize(bgr, (args.width, args.height))
            if undistort:
                bgr = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR)

            dets = detect_tag(detector, bgr, params, args.tag_size)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            if args.raw and dets:
                raw = detector.detect(to_gray(bgr), estimate_tag_pose=True,
                                      camera_params=params, tag_size=args.tag_size)
                for i, d in enumerate(raw):
                    print_raw(d, i)

            vis = draw_overlay(bgr.copy(), dets, params, args.tag_size, fps)
            cv2.imshow("OAK-D Lite AF - AprilTag", vis)
            if writer is not None:
                writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r") and dets:
                raw = detector.detect(to_gray(bgr), estimate_tag_pose=True,
                                      camera_params=params, tag_size=args.tag_size)
                for i, d in enumerate(raw):
                    print_raw(d, i)
            if key == ord("s"):
                path = f"oakd_frame_{frame_i:04d}.png"
                cv2.imwrite(path, vis)
                print(f"  [oakd] saved {path}")
                frame_i += 1

        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
