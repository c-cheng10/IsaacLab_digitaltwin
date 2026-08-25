import argparse
import os
import threading
import queue


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Demonstrates the PF400 articulation."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="s1160",
    choices=["s1160", "xs750"],
    help="Which PF400 model config to load.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg


import pf400_model
import pf400_s_1160_cfg as _cfg_s1160
import pf400_xs_750_cfg as _cfg_xs750

S_BASE = (0.0, 0.0, 0.0)
S_YAW_DEG = 0.0

# The XS sits on the bench, not on the floor: 0.9 m up clears the table it was
# spawning inside. The 90 deg yaw turns it to face across the cell.
# base_yaw is fed to PF400Model as well as the USD, so FK/IK stay world-frame --
# rotating only the prim would leave the kinematics solving in an unrotated
# frame and kincheck would go wrong immediately.
XS_BASE = (1.2, 0.0, 0.9)
XS_YAW_DEG = 90.0


def _yaw_quat(deg):
    """Isaac Lab wants (w, x, y, z). Rotation about world Z only."""
    h = math.radians(deg) * 0.5
    return (math.cos(h), 0.0, 0.0, math.sin(h))

ARM_SPEC = {
    "s1160": dict(key="pf400",    prim="Robot",
                  cfg=_cfg_s1160.PF400S_1160_CFG, base=S_BASE, yaw_deg=S_YAW_DEG),
    "xs750": dict(key="pf400_xs", prim="RobotXS",
                  cfg=_cfg_xs750.PF400XS_750_CFG, base=XS_BASE, yaw_deg=XS_YAW_DEG),
}

_MODELS = {
    "s1160": _cfg_s1160.build_model(name="s1160", base_offset=S_BASE,
                                    base_yaw=math.radians(S_YAW_DEG)),
    "xs750": _cfg_xs750.build_model(name="xs750", base_offset=XS_BASE,
                                    base_yaw=math.radians(XS_YAW_DEG)),
}

ACTIVE = args_cli.robot
ACTIVE_ARM = ARM_SPEC[ACTIVE]["key"]
_model = _MODELS[ACTIVE]
PF400_CFG = ARM_SPEC[ACTIVE]["cfg"]

ARM_JOINTS = _model.arm_joints
GRIPPER_JOINTS = _model.gripper_joints
HOME_ARM = _model.home_arm

for _k, _s in ARM_SPEC.items():
    print(f"[cfg] {_k:6s} -> scene['{_s['key']}']  base={_s['base']}  "
          f"yaw={_s['yaw_deg']:.0f}deg  "
          f"{_s['cfg'].spawn.usd_path.rsplit('/', 1)[-1]}")
print(f"[cfg] active arm = {ACTIVE}")


USE_LAB_SCENE = True

LAB_USD_PATH = ("/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/"
                "models/pf400/lab_with_tags.usd")


@configclass
class NewRobotsSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    pf400: ArticulationCfg = _cfg_s1160.PF400S_1160_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_cfg_s1160.PF400S_1160_CFG.init_state.replace(pos=S_BASE),
    )

    pf400_xs: ArticulationCfg = _cfg_xs750.PF400XS_750_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RobotXS",
        init_state=_cfg_xs750.PF400XS_750_CFG.init_state.replace(
            pos=XS_BASE, rot=_yaw_quat(XS_YAW_DEG)),
    )

    dummy_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/dummy_cam",
        spawn=sim_utils.PinholeCameraCfg(),
        update_period=0.1,
        height=64,
        width=64,
        data_types=["rgb"],
    )

    ee_cam_left = CameraCfg(
        prim_path="/World/ee_cam_left",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.37,
            horizontal_aperture=4.71,
            clipping_range=(0.01, 1.0e6),
        ),
        update_period=0.1,
        height=960,
        width=1280,
        data_types=["rgb"],
    )

    ee_cam_middle = CameraCfg(
        prim_path="/World/ee_cam_middle",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.37,
            horizontal_aperture=4.71,
            clipping_range=(0.01, 1.0e6),
        ),
        update_period=0.1,
        height=960,
        width=1280,
        data_types=["rgb"],
    )

    ee_cam_right = CameraCfg(
        prim_path="/World/ee_cam_right",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.37,
            horizontal_aperture=4.71,
            clipping_range=(0.01, 1.0e6),
        ),
        update_period=0.1,
        height=960,
        width=1280,
        data_types=["rgb"],
    )

    if USE_LAB_SCENE:
        lab = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Lab",
            spawn=sim_utils.UsdFileCfg(
                usd_path=LAB_USD_PATH,
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),
                                                   rot=(1.0, 0.0, 0.0, 0.0)),
        )
    else:
        box = RigidObjectCfg(
            prim_path = "{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 1),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(),
                mass_props = sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props = sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
            init_state = RigidObjectCfg.InitialStateCfg(pos=(0.3, -1.3, 0.5)),
        )

        apriltag = AssetBaseCfg(
            prim_path = "{ENV_REGEX_NS}/Box/AprilTag",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/models/pf400/AprilTag_ID2.usd",
                scale=(0.001,0.001,0.001),
            ),
            init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.27, 0.3), rot=(0.7071,-0.7071,0,0)),
        )


def input_reader(q: queue.Queue):
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        q.put(line)


CAMERA_NAMES = ["ee_cam_left", "ee_cam_middle", "ee_cam_right"]

PRIMARY_CAMERA = "ee_cam_middle"

_WORLD_CAM_POSE = {}

_JOINT_IDX = {}
_ARM_IDX = []
_GRIP_IDX = []
HOME_POSE = None


def active_robot(scene):
    return scene[ACTIVE_ARM]


def _mirror_active_globals():
    global _JOINT_IDX, _ARM_IDX, _GRIP_IDX, HOME_POSE, JOINT_LIMITS, ARM_LIMITS
    global ARM_JOINTS, GRIPPER_JOINTS, HOME_ARM
    _JOINT_IDX = _model.joint_idx
    _ARM_IDX = _model.arm_idx
    _GRIP_IDX = _model.grip_idx
    HOME_POSE = _model.home_pose
    JOINT_LIMITS = _model.joint_limits
    ARM_LIMITS = _model.arm_limits
    ARM_JOINTS = _model.arm_joints
    GRIPPER_JOINTS = _model.gripper_joints
    HOME_ARM = _model.home_arm


def setup_joint_map(robot):
    _model.bind(robot)
    _mirror_active_globals()


def set_active_arm(scene, which):
    global ACTIVE, ACTIVE_ARM, _model, PF400_CFG
    if which not in ARM_SPEC:
        raise ValueError(f"unknown arm '{which}' (have {list(ARM_SPEC)})")
    ACTIVE = which
    ACTIVE_ARM = ARM_SPEC[which]["key"]
    PF400_CFG = ARM_SPEC[which]["cfg"]
    _model = _MODELS[which]
    if _model.home_pose is None:
        _model.bind(scene[ACTIVE_ARM])
    _mirror_active_globals()
    setup_camera_mounts(scene)
    print(f"  [arm] active = {which}  (scene['{ACTIVE_ARM}'], "
          f"base={ARM_SPEC[which]['base']}, yaw={ARM_SPEC[which]['yaw_deg']:.0f}deg, "
          f"reach={_model.reach*1000:.1f}mm, "
          f"rail={'yes' if _model.has_rail else 'no'})")
    return scene[ACTIVE_ARM]


def arm_from_full(q_full):
    return _model.arm_from_full(q_full)


def full_from_arm(q_arm, q_full_current):
    return _model.full_from_arm(q_arm, base=q_full_current)


def _as_arm(q):
    return _model.as_arm(q)


def cmd_arm(robot, sim, q_arm):
    full = full_from_arm(_as_arm(q_arm), robot.data.joint_pos[0].cpu().numpy())
    robot.set_joint_position_target(
        torch.as_tensor(full, device=sim.device).unsqueeze(0))
    return full


def cmd_full(robot, sim, q_full):
    q_full = np.asarray(q_full, dtype=np.float32)
    robot.set_joint_position_target(
        torch.as_tensor(q_full, device=sim.device).unsqueeze(0))
    return q_full

_CAM_OFFSET_EE = {
    "ee_cam_middle": np.array([0.0,     -0.085, -0.024]),
    "ee_cam_right":  np.array([0.0375,  -0.085, -0.024]),
    "ee_cam_left":   np.array([-0.0375, -0.085, -0.024]),
}
_CAM_ROT_EE = np.array([[-1.0,  0.0,  0.0],
                        [ 0.0,  0.0, -1.0],
                        [ 0.0, -1.0,  0.0]])
_CAM_ALPHA = -np.pi / 2


def grab_camera(scene, cam_name):
    cam = scene[cam_name]
    out = cam.data.output

    if out is None or "rgb" not in out:
        return None
    rgb_t = out["rgb"]
    if rgb_t.numel() == 0 or rgb_t.ndim < 3:
        return None

    rgb = (rgb_t[0] if rgb_t.ndim == 4 else rgb_t)[..., :3].cpu().numpy()
    K_t = cam.data.intrinsic_matrices
    K = (K_t[0] if K_t.ndim == 3 else K_t).cpu().numpy()
    return rgb, K


TAG_SIZE = 0.024
CAM_OUT_DIR = "/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/cam"


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from pupil_apriltags import Detector
        _detector = Detector(families="tag36h11", nthreads=4)
    return _detector


def _to_gray(rgb):
    g = np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.ascontiguousarray(g)


def _tag_pose_world(T_world_cam, pose_R, pose_t):
    T_cam_tag = np.eye(4)
    T_cam_tag[:3, :3] = pose_R
    T_cam_tag[:3, 3] = np.asarray(pose_t).flatten()
    return T_world_cam @ T_cam_tag


def _print_raw(d, i):
    print(f"  --- raw detection[{i}] ---")
    for attr in ("tag_family", "tag_id", "hamming", "decision_margin", "homography",
                 "center", "corners", "pose_R", "pose_t", "pose_err"):
        if hasattr(d, attr):
            print(f"    {attr} =")
            print("      " + repr(getattr(d, attr)).replace("\n", "\n      "))


_BOX_SIZE = np.array([0.5, 0.5, 1.0])

_BOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _project_world(T_world_cam, params, pts_world):
    fx, fy, cx, cy = params
    T_cam_world = np.linalg.inv(T_world_cam)
    out = []
    for p in pts_world:
        pc = T_cam_world @ np.append(np.asarray(p, dtype=float), 1.0)
        out.append(None if pc[2] <= 1e-6
                   else (fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy))
    return out


def _has_box(scene):
    try:
        scene["box"]
        return True
    except (KeyError, AttributeError):
        return False


def _box_corners_world(scene):
    if not _has_box(scene):
        return []
    root = scene["box"].data.root_state_w[0].cpu().numpy()
    T = _pose_to_matrix(root[:3], root[3:7])
    h = _BOX_SIZE / 2.0
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * h[0], sy * h[1], sz * h[2]])
                pts.append((T @ np.append(local, 1.0))[:3])
    return pts


def save_overlay(rgb, detections, cam_name, box_px=None, tag_px=None):
    from PIL import Image, ImageDraw
    vis = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(vis)

    if box_px is not None:
        for a, b in _BOX_EDGES:
            if box_px[a] is not None and box_px[b] is not None:
                draw.line([box_px[a], box_px[b]], fill=(255, 0, 255), width=2)

    if tag_px is not None:
        tx, ty = tag_px
        draw.line([(tx - 18, ty), (tx + 18, ty)], fill=(0, 128, 255), width=3)
        draw.line([(tx, ty - 18), (tx, ty + 18)], fill=(0, 128, 255), width=3)
        draw.text((tx + 20, ty + 4), "PREDICTED", fill=(0, 128, 255))

    for det in detections:
        pts = [tuple(map(float, c)) for c in det["corners_px"]]
        draw.line(pts + [pts[0]], fill=(0, 255, 0), width=3)
        cx_, cy_ = float(det["center_px"][0]), float(det["center_px"][1])
        draw.ellipse([cx_ - 5, cy_ - 5, cx_ + 5, cy_ + 5], fill=(255, 0, 0))
        for n, (px, py) in enumerate(pts):
            draw.text((px + 4, py + 4), str(n), fill=(0, 255, 255))
        draw.text((cx_ + 8, cy_ - 14), f"DETECTED id{det['tag_id']}  {det['distance']:.3f}m",
                  fill=(255, 255, 0))
    vis.save(f"{CAM_OUT_DIR}/detect_{cam_name}.png")


def detect_tag(scene, cam_name=None, tag_size=None, verbose=True, overlay=True):
    cam_name = PRIMARY_CAMERA if cam_name is None else cam_name
    frame = grab_camera(scene, cam_name)
    if frame is None:
        if verbose:
            print(f"  [{cam_name}] no frame yet")
        return []
    rgb, K = frame

    tag_sz = TAG_SIZE if tag_size is None else tag_size
    params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    dets = get_detector().detect(_to_gray(rgb), estimate_tag_pose=True,
                                 camera_params=params, tag_size=tag_sz)

    if verbose:
        print(f"  [{cam_name}] fx,fy,cx,cy={tuple(round(p, 2) for p in params)}  "
              f"tag_size={tag_sz}  img={rgb.shape[1]}x{rgb.shape[0]}  {len(dets)} detection(s)")

    T_world_cam = live_camera_matrix(scene, cam_name)

    out = []
    for i, d in enumerate(dets):
        if verbose:
            _print_raw(d, i)
        pose_R = np.asarray(d.pose_R, dtype=float)
        pose_t = np.asarray(d.pose_t, dtype=float).flatten()
        out.append({
            "tag_id": int(d.tag_id),
            "camera": cam_name,
            "pose_R": pose_R,
            "pose_t": pose_t,
            "pose_err": float(d.pose_err),
            "distance": float(np.linalg.norm(pose_t)),
            "center_px": np.asarray(d.center, dtype=float),
            "corners_px": np.asarray(d.corners, dtype=float),
            "params": params,
            "T_world_cam": T_world_cam,
            "T_world_tag": _tag_pose_world(T_world_cam, pose_R, pose_t),
        })

    if overlay:
        corners = _box_corners_world(scene)
        box_px = _project_world(T_world_cam, params, corners) if corners else None
        T_exp = _expected_tag_world(scene)
        tag_px = (_project_world(T_world_cam, params, [T_exp[:3, 3]])[0]
                  if T_exp is not None else None)
        save_overlay(rgb, out, cam_name, box_px=box_px, tag_px=tag_px)
    return out


def detect_all_cameras(scene, tag_size=None, verbose=True, overlay=True):
    out = []
    for name in CAMERA_NAMES:
        out.extend(detect_tag(scene, cam_name=name, tag_size=tag_size,
                              verbose=verbose, overlay=overlay))
    return out


def fuse_poses(mats):
    if len(mats) == 1:
        return mats[0], 0.0
    positions = np.stack([T[:3, 3] for T in mats])
    pos = np.median(positions, axis=0)
    rot = np.median(np.stack([T[:3, :3] for T in mats]), axis=0)
    U, _, Vh = np.linalg.svd(rot)
    rot = U @ Vh
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vh
    spread = float(np.linalg.norm(positions - pos, axis=1).max())
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T, spread


def tag_pose_world(scene, tag_size, tag_id=2, verbose=False):
    mats = [d["T_world_tag"]
            for d in detect_all_cameras(scene, tag_size=tag_size, verbose=verbose)
            if d["tag_id"] == tag_id]
    if not mats:
        return None
    T, spread = fuse_poses(mats)
    if verbose or spread > 0.02:
        print(f"  [fuse] tag {tag_id}: {len(mats)} camera(s)  spread={spread:.4f} m"
              f"{'   <<< HIGH, cameras disagree' if spread > 0.02 else ''}")
    return T


_ee_idx = None
_T_ee_cam = {}

def _pose_to_matrix(pos, quat_wxyz):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    w, x, y, z = quat_wxyz
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(pos).flatten()
    return T


def _ee_matrix(scene):
    robot = active_robot(scene)
    return _pose_to_matrix(
        robot.data.body_pos_w[0, _ee_idx].cpu().numpy(),
        robot.data.body_quat_w[0, _ee_idx].cpu().numpy(),
    )


def _stage_world_transform(prim_name):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            raw = np.array([[m[j][i] for j in range(4)] for i in range(4)])
            U, _, Vh = np.linalg.svd(raw[:3, :3])
            R = U @ Vh
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vh
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = raw[:3, 3]
            return T
    return None


def setup_camera_mounts(scene):
    global _ee_idx, _T_ee_cam
    robot = active_robot(scene)

    if "endeffector" not in robot.body_names:
        print("  [mount] 'endeffector' not in body_names:", robot.body_names)
        return
    _ee_idx = robot.body_names.index("endeffector")

    for name in CAMERA_NAMES:
        if name in _WORLD_CAM_POSE:
            print(f"  [mount] {name}: world-fixed, no EE mount needed")
            continue
        T = np.eye(4)
        T[:3, :3] = _CAM_ROT_EE
        T[:3, 3] = _CAM_OFFSET_EE[name]
        _T_ee_cam[name] = T
    print(f"  [mount] fixed offsets set for: {list(_T_ee_cam.keys())}")



def live_camera_matrix(scene, cam_name):
    if cam_name in _WORLD_CAM_POSE:
        pos, quat = _WORLD_CAM_POSE[cam_name]
        return _pose_to_matrix(pos, quat)
    return _ee_matrix(scene) @ _T_ee_cam[cam_name]


def sync_ee_cameras(scene):
    if not _T_ee_cam:
        return
    from scipy.spatial.transform import Rotation as R
    T_we = _ee_matrix(scene)
    for name in CAMERA_NAMES:
        if name not in _T_ee_cam:
            continue
        T = T_we @ _T_ee_cam[name]
        q = R.from_matrix(T[:3, :3]).as_quat()
        dev = scene[name].data.pos_w.device
        pos = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev).unsqueeze(0)
        quat = torch.tensor([q[3], q[0], q[1], q[2]],
                            dtype=torch.float32, device=dev).unsqueeze(0)
        scene[name].set_world_poses(pos, quat, convention="ros")


_APRILTAG_LOCAL_POS = np.array([0.0, 0.27, 0.3])
_APRILTAG_LOCAL_QUAT = np.array([0.7071, -0.7071, 0.0, 0.0])


def _expected_tag_world(scene):
    if not _has_box(scene):
        return None
    box = scene["box"]
    root = box.data.root_state_w[0].cpu().numpy()
    T_world_box = _pose_to_matrix(root[:3], root[3:7])
    T_box_tag = _pose_to_matrix(_APRILTAG_LOCAL_POS, _APRILTAG_LOCAL_QUAT)
    return T_world_box @ T_box_tag


def detect_tags(scene, tag_size):
    from scipy.spatial.transform import Rotation as R
    T_expected = _expected_tag_world(scene)

    dets = detect_all_cameras(scene, tag_size=tag_size)
    if not dets:
        print("  [detect] no tags detected on any camera")
        return

    if T_expected is not None:
        exp_euler = R.from_matrix(T_expected[:3, :3]).as_euler("xyz", degrees=True)
        print(f"\n  [expected] tag world_pos={np.round(T_expected[:3, 3], 4).tolist()}  "
              f"world_rot(xyz deg)={np.round(exp_euler, 1).tolist()}  "
              f"(analytical from box root pose -- REPORTING ONLY, never used to pick a pose)")
    else:
        print("\n  [expected] no analytic ground truth (lab scene -- tags are baked into the USD)")

    for d in dets:
        T = d["T_world_tag"]
        Tc = d["T_world_cam"]
        print(f"  [{d['camera']}] cam world_pos={np.round(Tc[:3, 3], 4).tolist()}  "
              f"fwd={np.round(Tc[:3, 2], 3).tolist()}")
        euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        cam_euler = R.from_matrix(d["pose_R"]).as_euler("xyz", degrees=True)
        span = float(np.linalg.norm(d["corners_px"][0] - d["corners_px"][2]))
        print(f"  [detected] tag {d['tag_id']}: "
              f"world_pos={np.round(T[:3, 3], 4).tolist()}  "
              f"world_rot(xyz deg)={np.round(euler, 1).tolist()}")
        print(f"             cam_frame: t={np.round(d['pose_t'], 4).tolist()}  "
              f"rot(xyz deg)={np.round(cam_euler, 1).tolist()}  dist={d['distance']:.4f}m")

        if T_expected is None:
            print(f"             pose_err={d['pose_err']:.2e}  corner_span={span:.1f}px")
            continue

        pos_diff = float(np.linalg.norm(T[:3, 3] - T_expected[:3, 3]))
        R_rel = T[:3, :3] @ T_expected[:3, :3].T
        rot_diff = math.degrees(math.acos(
            max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))))
        print(f"             pos_diff={pos_diff:.4f}m  rot_diff={rot_diff:.2f}deg  "
              f"pose_err={d['pose_err']:.2e}  corner_span={span:.1f}px")

        fx, fy, cx, cy = d["params"]
        p_cam = np.linalg.inv(d["T_world_cam"]) @ np.append(T_expected[:3, 3], 1.0)
        if p_cam[2] > 1e-6:
            u_pred = fx * p_cam[0] / p_cam[2] + cx
            v_pred = fy * p_cam[1] / p_cam[2] + cy
            u_det, v_det = float(d["center_px"][0]), float(d["center_px"][1])
            px_err = math.hypot(u_pred - u_det, v_pred - v_det)
            print(f"  [reproj]   expected tag projects to ({u_pred:.1f}, {v_pred:.1f})px  "
                  f"detected center ({u_det:.1f}, {v_det:.1f})px  err={px_err:.1f}px")
            print(f"             signed du={u_pred - u_det:+.1f}px  dv={v_pred - v_det:+.1f}px")
            print(f"             expected in cam frame: "
                  f"{np.round(p_cam[:3], 4).tolist()}  vs solved {np.round(d['pose_t'], 4).tolist()}")
        else:
            print("  [reproj]   expected tag is behind the camera -- pose/mount suspect")

        R_wc = d["T_world_cam"][:3, :3]
        cam_world_solved = T_expected[:3, 3] - R_wc @ d["pose_t"]
        if d["camera"] in _WORLD_CAM_POSE:
            print(f"  [solve]    camera world pos implied by detection = "
                  f"{np.round(cam_world_solved, 4).tolist()}")
            print(f"             configured (fixed, exact)             = "
                  f"{np.round(_WORLD_CAM_POSE[d['camera']][0], 4).tolist()}")
        else:
            off_ee_solved = np.linalg.inv(_ee_matrix(scene)) @ np.append(cam_world_solved, 1.0)
            print(f"  [solve]    _CAM_OFFSET_EE implied by this detection = "
                  f"{np.round(off_ee_solved[:3], 4).tolist()}")
            print(f"             currently configured                    = "
                  f"{np.round(_CAM_OFFSET_EE[d['camera']], 4).tolist()}")

    by_tag = {}
    for d in dets:
        by_tag.setdefault(d["tag_id"], []).append(d["T_world_tag"])
    for tag_id, mats in by_tag.items():
        T, spread = fuse_poses(mats)
        euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        line = (f"  [FUSED]    tag {tag_id} from {len(mats)} camera(s): "
                f"world_pos={np.round(T[:3, 3], 4).tolist()}  "
                f"world_rot(xyz deg)={np.round(euler, 1).tolist()}")
        print(line)
        pd = ""
        if T_expected is not None:
            pd = f"pos_diff={np.linalg.norm(T[:3, 3] - T_expected[:3, 3]):.4f}m  "
        print(f"             {pd}cam_spread={spread:.4f}m"
              f"{'   <<< HIGH, cameras disagree' if spread > 0.02 else ''}")


_P_SHOULDER = _model.p_shoulder
_P_ELBOW    = _model.p_elbow
_P_WRIST    = _model.p_wrist
_P_EE       = _model.p_ee

_kin = _model

_EE_YAW_CAL_DEG = _model.a3_rest
_OUTER_HOME_DEG = _model.outer_home_deg
_A1_REST = _model.a1_rest
_A2_REST = _model.a2_rest
_A3_REST = _model.a3_rest
J2_OFFSET = _model.j2_offset
J3_OFFSET = _model.j3_offset
J4_OFFSET = _model.j4_offset
_BASE_XY = _model.base_xy
_Z_REST = _model.z_rest
RAIL_AXIS = _model.rail_axis
RAIL_SIGN = _model.rail_sign

JOINT_LIMITS = None
ARM_LIMITS = None


def _wrap_deg(deg):
    return pf400_model.wrap_deg(deg)


def pf400_fk(joint_pos):
    return _model.fk(joint_pos)


def pf400_ik(x, y, z, yaw_rad, rail=0.0, q_current=None, quiet=False):
    return _model.ik(x, y, z, yaw_rad, rail=rail, q_current=q_current, quiet=quiet)


def pf400_ik_auto(x, y, z, yaw_rad, q_current=None, samples=61):
    return _model.ik_auto(x, y, z, yaw_rad, q_current=q_current, samples=samples)


def goal_pose_from_tag(T_world_tag, standoff, off_x, off_y, cam_name=PRIMARY_CAMERA):
    R = T_world_tag[:3, :3]
    p = T_world_tag[:3, 3]
    n = R[:, 2]

    base = np.array([_BASE_XY[0], _BASE_XY[1], p[2]])
    s = standoff
    if np.linalg.norm(p + s * n - base) > np.linalg.norm(p - s * n - base):
        s = -s
    cam_target = p + s * n + off_x * R[:, 0] + off_y * R[:, 1]

    d = -s * n
    theta = math.atan2(d[1], d[0]) - _CAM_ALPHA

    off_ee = _CAM_OFFSET_EE[cam_name]
    c, sn = math.cos(theta), math.sin(theta)
    rotated = np.array([c * off_ee[0] - sn * off_ee[1],
                        sn * off_ee[0] + c * off_ee[1],
                        off_ee[2]])
    ee_pos = cam_target - rotated
    return float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]), float(theta)


GOAL_DEFAULT_TAG = 1


def build_goal_pose(scene, tag_size, standoff, off_x, off_y, tag_id=None,
                    default_tag=GOAL_DEFAULT_TAG):
    """Hover pose in front of a tag.

    `tag_id` given  -> use it, and say so if it is not actually visible.
    `tag_id` None   -> if exactly one tag is in frame there is nothing to be
                       ambiguous about, so use it whatever its id. With several
                       in frame, fall back to `default_tag` rather than guessing.

    Detection runs once here and the fuse is done inline (rather than calling
    tag_pose_world) so choosing the tag does not cost a second render pass.
    """
    dets = detect_all_cameras(scene, tag_size=tag_size, verbose=False)
    visible = sorted({int(d["tag_id"]) for d in dets})
    if not visible:
        print("  [goal] no tags visible")
        return None

    if tag_id is None:
        if len(visible) == 1:
            tag_id = visible[0]
            print(f"  [goal] one tag visible -> tag {tag_id}")
        else:
            tag_id = default_tag
            print(f"  [goal] tags visible {visible} -> defaulting to tag {tag_id} "
                  f"(pass 'tag=N' to pick another)")
    else:
        tag_id = int(tag_id)

    mats = [d["T_world_tag"] for d in dets if int(d["tag_id"]) == tag_id]
    if not mats:
        print(f"  [goal] tag {tag_id} not detected  (visible: {visible})")
        return None

    T_tag, spread = fuse_poses(mats)
    if spread > 0.02:
        print(f"  [fuse] tag {tag_id}: {len(mats)} camera(s)  spread={spread:.4f} m"
              f"   <<< HIGH, cameras disagree")
    return goal_pose_from_tag(T_tag, standoff, off_x, off_y)


def ee_in_tag_frame(scene, T_world_tag):
    return np.linalg.inv(T_world_tag) @ _ee_matrix(scene)


def shift_box(scene, dx=None, dy=None, spread=0.12):
    if not _has_box(scene):
        print("  [box] no movable box in this scene (USE_LAB_SCENE=True) -- nothing to shift")
        return None, None
    box = scene["box"]
    if dx is None:
        dx = float(np.random.uniform(-spread, spread))
    if dy is None:
        dy = float(np.random.uniform(-spread, spread))
    root = box.data.root_state_w.clone()
    root[0, 0] += dx
    root[0, 1] += dy
    box.write_root_pose_to_sim(root[:, :7])
    box.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    return dx, dy


_demo = None


def demo_start(standoff, off_x=0.0, off_y=0.0, spread=0.12):
    global _demo
    _demo = {
        "phase": "home1", "t": 0, "target": None,
        "standoff": standoff, "off_x": off_x, "off_y": off_y, "spread": spread,
        "tagA": None, "tagB": None, "refA": None,
    }
    print(f"\n  [demo] tag-relative re-targeting: standoff={standoff} m, "
          f"box shift up to +/-{spread} m")


POSITION_TREE = {
    42: ("home",                   "root_home", None, (800.0,   0.0, 180.0, 180.0, -190.0), None),
    43: ("home_vspin",             "waypoint",    42, (800.0,   0.0, 180.0, 180.0, -509.0), None),
    44: ("home_vspin_lower",       "waypoint",    43, (260.0,   0.0, 180.0, 180.0, -509.0), None),
    45: ("regrip narrow top",      "waypoint",    42, (780.0, -37.3, 254.8,  53.2, -190.0), None),
    46: ("regrip narrow approach", "approach",    45, (755.0, -37.3, 254.8,  53.2, -190.0), None),
    47: ("regrip narrow",          "place",       46, (746.5, -37.3, 254.8,  53.2, -190.0), None),
    48: ("vspin entrance",         "waypoint",    44, (240.0,  49.8, 228.3,  -9.0, -509.0), None),
    49: ("vspin approach",         "approach",    48, (240.0, -22.0, 282.7,   8.3, -509.0), "y"),
    50: ("vspin",                  "place",       49, (220.5, -22.0, 282.7,   8.3, -509.0), None),
    51: ("home biomek",            "waypoint",    42, (1070.0,  0.0, 180.0, 180.0,  439.8), "z"),
    52: ("home biomek upper",      "waypoint",    51, (1070.0,  0.0, 180.0, 180.0,  439.8), None),
    54: ("biomek entrance",        "waypoint",    52, (1070.0, 60.0, 230.4, -19.9,  439.8), None),
    55: ("biomek middle approach", "approach",    54, (1070.0,-41.5, 316.2,  -4.2,  439.8), "y"),
    56: ("biomek middle",          "place",       55, (978.4, -41.5, 316.2,  -4.2,  439.8), None),
    57: ("home incubator",         "waypoint",    52, (1120.0,  0.0, 180.0, 180.0,  448.0), None),
    58: ("incubator approach",     "approach",    57, (1120.0, 82.5, 259.4, 111.4,  448.0), None),
    59: ("incubator position 1",   "place",       58, (1046.3, 82.5, 259.4, 111.4,  448.0), None),
}

TREE_ROOT = 42
VIA_ROOT = False

_current_node = None
_seq = None


DB_EE_OFFSET_DEG = -180.0
DB_RAIL_SIGN = -1.0


def node_q(node_id):
    col, inner, outer, eff, rail = POSITION_TREE[node_id][3]
    return np.array([DB_RAIL_SIGN * rail / 1000.0,
                     col / 1000.0,
                     math.radians(inner),
                     math.radians(outer),
                     math.radians(eff + DB_EE_OFFSET_DEG)],
                    dtype=np.float32)


def node_by_name(token):
    t = token.strip().lower()
    if t.isdigit() and int(t) in POSITION_TREE:
        return int(t)
    for nid, (name, _k, _p, _q, _pr) in POSITION_TREE.items():
        if name.lower() == t:
            return nid
    hits = [nid for nid, (name, _k, _p, _q, _pr) in POSITION_TREE.items()
            if t in name.lower()]
    return hits[0] if len(hits) == 1 else None


def node_ancestry(node_id):
    chain = []
    while node_id is not None:
        chain.append(node_id)
        node_id = POSITION_TREE[node_id][2]
    return list(reversed(chain))


def tree_route(from_id, to_id, via_root=None):
    via_root = VIA_ROOT if via_root is None else via_root
    a, b = node_ancestry(from_id), node_ancestry(to_id)
    if via_root:
        common = 1
    else:
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
    up = list(reversed(a[common:]))
    down = b[common:]
    turn = a[common - 1]
    return up + [turn] + down


def print_tree():
    kids = {}
    for nid, (_n, _k, par, _q, _pr) in POSITION_TREE.items():
        kids.setdefault(par, []).append(nid)

    def walk(nid, depth):
        name, kind, _p, _q, pr = POSITION_TREE[nid]
        mark = "  <== HERE" if nid == _current_node else ""
        prio = f"  prio={pr}" if pr else ""
        print(f"    {'  ' * depth}{nid:>3}  {name:<24s} [{kind}]{prio}{mark}")
        for c in sorted(kids.get(nid, [])):
            walk(c, depth + 1)

    print(f"  [tree] routing={'via root' if VIA_ROOT else 'lowest common ancestor'}"
          f"   current={_current_node}")
    walk(TREE_ROOT, 0)


CALIB_DIR = "/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/calib"

CALIB_POSES = {
    "table_a": {
        "base_node": 51,
        "joints": {"columnjoint": 0.6, "endeffectorjoint": -0.8},
        "tag": 19,
        "object": "Table_A",
    },
    "table_b": {
        "base_node": 51,
        "joints": {"columnjoint": 0.6, "endeffectorjoint": 0.8},
        "tag": 40,
        "object": "Table_B",
    },
}

CALIB_VIEWS = ["table_a", "table_b"]

CALIB_MAX_TRANS = 0.050
CALIB_MAX_YAW_DEG = 10.0
CALIB_SETTLE_STEPS = 30

_calib = None


def calib_pose_q(key):
    p = CALIB_POSES[key]
    q = node_q(p["base_node"]).astype(np.float32).copy()
    for jname, val in p["joints"].items():
        if jname not in ARM_JOINTS:
            raise KeyError(f"{jname} is not an arm joint {ARM_JOINTS}")
        q[ARM_JOINTS.index(jname)] = float(val)
    return q


def calib_pose_report(key):
    p = CALIB_POSES[key]
    q = calib_pose_q(key)
    print(f"  [calib] pose '{key}'  base node {p['base_node']} "
          f"'{POSITION_TREE[p['base_node']][0]}'  tag {p['tag']} -> {p['object']}")
    for jname, v in zip(ARM_JOINTS, q):
        if jname in ("railjoint", "columnjoint"):
            print(f"      {jname:18s} {v:+9.4f} m      ({v*1000:+8.1f} mm)")
        else:
            print(f"      {jname:18s} {v:+9.4f} rad    ({math.degrees(v):+9.4f} deg)")
    rail, col, inner, outer, eff = q
    print(f"      exnode DB row: (col={col*1000:.1f}, inner={math.degrees(inner):.4f}, "
          f"outer={math.degrees(outer):.4f}, eff={math.degrees(eff)-DB_EE_OFFSET_DEG:.4f}, "
          f"rail={rail*1000*DB_RAIL_SIGN:.1f})")


def calib_image_name(key, cam_name):
    return f"{key}_{cam_name}.png"


def calib_image_path(key, cam_name):
    return os.path.join(CALIB_DIR, calib_image_name(key, cam_name))


def _load_image(path):
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path))[..., :3]
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(path)
        return None if img is None else img[:, :, ::-1]
    except Exception:
        return None


def detect_in_image(path, params, tag_size=None):
    rgb = _load_image(path)
    if rgb is None:
        return None, []
    tag_sz = TAG_SIZE if tag_size is None else tag_size
    dets = get_detector().detect(_to_gray(rgb), estimate_tag_pose=True,
                                 camera_params=params, tag_size=tag_sz)
    out = []
    for d in dets:
        R_ = np.asarray(d.pose_R, dtype=float)
        t_ = np.asarray(d.pose_t, dtype=float).flatten()
        T = np.eye(4)
        T[:3, :3] = R_
        T[:3, 3] = t_
        out.append({"tag_id": int(d.tag_id), "T_cam_tag": T,
                    "pose_err": float(d.pose_err),
                    "center_px": np.asarray(d.center, dtype=float)})
    return rgb, out


def _constrain_se2_z(D):
    R_ = D[:3, :3]
    yaw = math.atan2(R_[1, 0], R_[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out[:3, 3] = D[:3, 3]
    return out, math.degrees(yaw)


def set_prim_world_transform(prim_name, T):
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() != prim_name:
            continue
        xf = UsdGeom.Xformable(prim)
        Tp = np.eye(4)
        par = prim.GetParent()
        if par and par.GetPath() != stage.GetPseudoRoot().GetPath():
            m = UsdGeom.Xformable(par).ComputeLocalToWorldTransform(0.0)
            Tp = np.array([[m[j][i] for j in range(4)] for i in range(4)])
        Tl = np.linalg.inv(Tp) @ T
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        op.Set(Gf.Matrix4d(*[float(Tl[c][r]) for r in range(4) for c in range(4)]))
        return True
    return False


def calib_info():
    print(f"  [calib] image folder: {CALIB_DIR}")
    print(f"  [calib] {len(CALIB_VIEWS)} pose(s) configured")
    print()
    for key in CALIB_VIEWS:
        calib_pose_report(key)
        for cam in CAMERA_NAMES:
            p = calib_image_path(key, cam)
            mark = "found" if os.path.exists(p) else "MISSING"
            print(f"      image: {calib_image_name(key, cam):34s} [{mark}]")
        print()


def calib_start(robot, sim, dry=False, only=None):
    global _calib
    if _current_node is None:
        print("  [calib] current node unknown. Run 'treehome' or 'setnode <name>' first.")
        return
    views = [k for k in CALIB_VIEWS if only is None or k == only]
    if not views:
        print(f"  [calib] no pose named '{only}'. Known: {list(CALIB_POSES)}")
        return
    missing = [k for k in views
               if not any(os.path.exists(calib_image_path(k, c)) for c in CAMERA_NAMES)]
    if missing:
        print(f"  [calib] no image found for pose(s) {missing} in {CALIB_DIR}")
        print("  [calib] run 'calibinfo' for the exact filenames expected")
        return
    _calib = {"views": views, "i": 0, "phase": "route", "wait": 0,
              "dry": dry, "results": []}
    print(f"  [calib] {len(views)} pose(s), dry_run={dry}")


def _calib_measure(scene, key, tag_id):
    out = []
    for cam in CAMERA_NAMES:
        path = calib_image_path(key, cam)
        if not os.path.exists(path):
            continue
        frame = grab_camera(scene, cam)
        if frame is None:
            continue
        rgb, K = frame
        params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        sim_dets = detect_tag(scene, cam_name=cam, verbose=False, overlay=False)
        real_rgb, real_dets = detect_in_image(path, params)
        if real_rgb is None:
            print(f"    [calib] {cam}: could not read {path}")
            continue
        if real_rgb.shape[:2] != rgb.shape[:2]:
            print(f"    [calib] {cam}: image is {real_rgb.shape[1]}x{real_rgb.shape[0]} but "
                  f"sim camera is {rgb.shape[1]}x{rgb.shape[0]} -- intrinsics do not match, "
                  f"result is not trustworthy")
        s = next((d for d in sim_dets if d["tag_id"] == tag_id), None)
        r = next((d for d in real_dets if d["tag_id"] == tag_id), None)
        if s is None or r is None:
            where = []
            if s is None:
                where.append("sim")
            if r is None:
                where.append("image")
            print(f"    [calib] {cam}: tag {tag_id} not found in {' and '.join(where)}")
            continue
        T_sim = np.eye(4)
        T_sim[:3, :3] = s["pose_R"]
        T_sim[:3, 3] = s["pose_t"]
        D_cam = r["T_cam_tag"] @ np.linalg.inv(T_sim)
        T_wc = s["T_world_cam"]
        D_world = T_wc @ D_cam @ np.linalg.inv(T_wc)
        out.append((cam, D_world))
        print(f"    [calib] {cam:14s} shift={np.linalg.norm(D_cam[:3,3])*1000:7.2f} mm  "
              f"sim_err={s['pose_err']:.2e}  img_err={r['pose_err']:.2e}")
    return out


def calib_step(scene, sim):
    global _calib, _seq
    d = _calib
    if d is None:
        return
    if d["i"] >= len(d["views"]):
        _calib_finish(scene)
        return
    key = d["views"][d["i"]]
    p = CALIB_POSES[key]
    base = p["base_node"]
    robot = active_robot(scene)

    if d["phase"] == "route":
        print(f"  [calib] pose '{key}' -> base node {base} '{POSITION_TREE[base][0]}'")
        if _current_node == base:
            d["phase"] = "adjust"
        else:
            seq_start(tree_route(_current_node, base), robot, sim)
            d["phase"] = "moving"
        return

    if d["phase"] == "moving":
        if _seq is None:
            if _current_node != base:
                print(f"  [calib] failed to reach node {base}, skipping '{key}'")
                d["i"] += 1
                d["phase"] = "route"
            else:
                d["phase"] = "adjust"
        return

    if d["phase"] == "adjust":
        q = calib_pose_q(key)
        cmd_arm(robot, sim, q)
        d["target"] = q
        d["wait"] = 0
        d["phase"] = "settle"
        print(f"      applying overrides {p['joints']}")
        return

    if d["phase"] == "settle":
        d["wait"] += 1
        err, vel = _joint_err(robot, d["target"])
        settled = d["wait"] >= CALIB_SETTLE_STEPS and err < 2.5e-2 and vel < 5e-3
        if settled or d["wait"] > 3000:
            if not settled:
                print(f"      settle TIMEOUT err={err:.4f} vel={vel:.4f}")
            d["phase"] = "measure"
        return

    if d["phase"] == "measure":
        print(f"  [calib] measuring '{key}'  tag {p['tag']} -> {p['object']}")
        meas = _calib_measure(scene, key, p["tag"])
        if meas:
            mats = [m for _c, m in meas]
            D = mats[0] if len(mats) == 1 else fuse_poses(mats)[0]
            d["results"].append((key, p["tag"], p["object"], D, len(mats)))
        else:
            print(f"    [calib] no usable measurement for '{key}'")
        d["i"] += 1
        d["phase"] = "route"
        return


def _calib_finish(scene):
    global _calib
    d = _calib
    _calib = None
    print()
    print("  [calib] ===== summary =====")
    by_obj = {}
    for key, tag_id, obj, D, ncam in d["results"]:
        Dc, yaw = _constrain_se2_z(D)
        trans = float(np.linalg.norm(Dc[:3, 3]))
        ok = trans <= CALIB_MAX_TRANS and abs(yaw) <= CALIB_MAX_YAW_DEG
        flag = "" if ok else "   <<< REJECTED (over threshold)"
        print(f"    '{key}'  tag {tag_id:>3}  obj={str(obj):<12s} cams={ncam}  "
              f"dxyz={np.round(Dc[:3,3]*1000,2).tolist()} mm  dyaw={yaw:+.2f} deg{flag}")
        if ok and obj:
            by_obj.setdefault(obj, []).append(Dc)
    if d["dry"]:
        print("  [calib] dry run -- nothing moved")
        return
    for obj, mats in by_obj.items():
        D = mats[0] if len(mats) == 1 else fuse_poses(mats)[0]
        T_old = _stage_world_transform(obj)
        if T_old is None:
            print(f"    [calib] prim '{obj}' not found on stage -- skipped")
            continue
        if set_prim_world_transform(obj, D @ T_old):
            print(f"    [calib] moved '{obj}' by {np.round(D[:3,3]*1000,2).tolist()} mm "
                  f"from {len(mats)} view(s)")
        else:
            print(f"    [calib] could not write transform for '{obj}'")
    print("  [calib] done. Re-run 'calibrate dry' -- residual should now be near zero.")


def seq_start(route, robot, sim):
    global _seq
    if len(route) > 1:
        err, _v = _joint_err(robot, node_q(route[0]))
        if bool(np.all(np.abs(err) < _tols(robot))):
            print(f"  [goto] already within tolerance of node {route[0]} "
                  f"(max_err {np.abs(err).max():.4f}) -- skipping it")
            route = route[1:]
    _seq = {"route": route, "i": -1, "target": None, "t": 0}
    names = " -> ".join(POSITION_TREE[n][0] for n in route)
    print(f"  [goto] {len(route)} waypoint(s): {names}")


SEQ_POS_TOL = 2.5e-2
SEQ_VEL_TOL = 5e-3
SEQ_CHECK_VELOCITY = False
SEQ_TIMEOUT = 600

SEQ_JOINT_TOL = {
    "railjoint":   0.01,
    "columnjoint": 0.01,
}


def _tols(robot):
    return np.array([SEQ_JOINT_TOL.get(n, SEQ_POS_TOL) for n in robot.joint_names])


def _joint_err(robot, target):
    q = robot.data.joint_pos[0].cpu().numpy()
    v = robot.data.joint_vel[0].cpu().numpy()
    t = np.asarray(target, dtype=np.float32)
    if t.shape[0] != q.shape[0]:
        t = full_from_arm(t, q)
    return q - t, v


def _seq_settled(robot, target):
    err, vel = _joint_err(robot, target)
    if not bool(np.all(np.abs(err) < _tols(robot))):
        return False
    if SEQ_CHECK_VELOCITY and not bool(np.all(np.abs(vel) < SEQ_VEL_TOL)):
        return False
    return True


def _err_report(robot, target):
    err, vel = _joint_err(robot, target)
    tols = _tols(robot)
    parts = []
    for n, e, v, t in zip(robot.joint_names, err, vel, tols):
        flag = " POS" if abs(e) >= t else ""
        if SEQ_CHECK_VELOCITY and abs(v) >= SEQ_VEL_TOL:
            flag += " VEL"
        parts.append(f"{n}: err={e:+.4f} (tol {t:.3f}) vel={v:+.4f}{flag}")
    return "\n             ".join(parts)


def seq_step(scene, sim):
    global _seq, _current_node
    robot = active_robot(scene)
    d = _seq
    d["t"] += 1

    if d["target"] is not None:
        if not _seq_settled(robot, d["target"]):
            if d["t"] >= SEQ_TIMEOUT:
                print(f"  [goto] TIMEOUT at node {d['route'][d['i']]} after {d['t']} steps "
                      f"-- advancing anyway. Blocking joints:")
                print(f"             {_err_report(robot, d['target'])}")
            elif d["t"] % 120 == 0:
                err, _v = _joint_err(robot, d["target"])
                k = int(np.argmax(np.abs(err)))
                print(f"  [goto] waiting {d['t']}/{SEQ_TIMEOUT} at node "
                      f"{d['route'][d['i']]}  worst={robot.joint_names[k]} "
                      f"err={err[k]:+.4f}")
                return
            else:
                return
        _current_node = d["route"][d["i"]]

    d["i"] += 1
    if d["i"] >= len(d["route"]):
        print(f"  [goto] arrived at '{POSITION_TREE[_current_node][0]}' (node {_current_node})")
        _seq = None
        return

    nid = d["route"][d["i"]]
    name, kind, _p, _q, pr = POSITION_TREE[nid]
    d["target"] = cmd_arm(robot, sim, node_q(nid))
    d["t"] = 0
    err, _v = _joint_err(robot, d["target"])
    print(f"  [goto] {d['i']+1}/{len(d['route'])} -> {nid} '{name}' [{kind}]"
          f"{'  prio=' + pr if pr else ''}  max_err={np.abs(err).max():.4f}")


def _demo_settled(robot, target, tol=2.5e-2, vel_tol=5e-3):
    err, v = _joint_err(robot, target)
    return bool(np.all(np.abs(err) < tol) and np.all(np.abs(v) < vel_tol))


def _demo_set(d, robot, sim, q_target):
    q = np.asarray(q_target, dtype=np.float32)
    if q.shape[0] == len(ARM_JOINTS):
        d["target"] = cmd_arm(robot, sim, q)
    else:
        d["target"] = cmd_full(robot, sim, q)


def _demo_phase(d, phase):
    d["phase"] = phase
    d["t"] = 0


def _demo_solve(d, scene, robot, T_tag, label):
    pose = goal_pose_from_tag(T_tag, d["standoff"], d["off_x"], d["off_y"])
    try:
        q = pf400_ik_auto(*pose, q_current=robot.data.joint_pos[0].cpu().numpy())
    except ValueError as e:
        print(f"  [demo] ABORT: {e}")
        return None
    print(f"  [demo] {label} -> xyz={np.round(pose[:3], 4).tolist()} "
          f"yaw={math.degrees(pose[3]):.1f}deg")
    return q


def demo_step(scene, sim, tag_size, tag_id=2):
    global _demo
    d = _demo
    robot = active_robot(scene)
    d["t"] += 1
    p = d["phase"]

    if p == "home1":
        if d["t"] == 1:
            print("  [demo] 1/7 moving to the safe home pose...")
            _demo_set(d, robot, sim, HOME_POSE)
        if _demo_settled(robot, d["target"]) or d["t"] > 600:
            _demo_phase(d, "detect1")

    elif p == "detect1":
        T_tag = tag_pose_world(scene, tag_size, tag_id)
        if T_tag is None:
            print("  [demo] ABORT: tag not visible from home. Move the box into "
                  "view or choose a different safe pose.")
            _demo = None
            return
        d["tagA"] = T_tag
        print(f"  [demo] 2/7 tag at {np.round(d['tagA'][:3, 3], 4).tolist()}")
        q = _demo_solve(d, scene, robot, d["tagA"], "3/7 hovering")
        if q is None:
            _demo = None
            return
        _demo_set(d, robot, sim, q)
        _demo_phase(d, "hover1")

    elif p == "hover1":
        if _demo_settled(robot, d["target"]) or d["t"] > 900:
            d["refA"] = ee_in_tag_frame(scene, d["tagA"])
            print(f"  [demo] 4/7 in position. EE in TAG frame = "
                  f"{np.round(d['refA'][:3, 3], 4).tolist()}")
            _demo_phase(d, "home2")

    elif p == "home2":
        if d["t"] == 1:
            print("  [demo] 5/7 returning to the safe pose...")
            _demo_set(d, robot, sim, HOME_POSE)
        if _demo_settled(robot, d["target"]) or d["t"] > 600:
            _demo_phase(d, "shift")

    elif p == "shift":
        if d["t"] == 1:
            dx, dy = shift_box(scene, spread=d["spread"])
            if dx is None:
                print("  [demo] 6/7 no movable box -- move a tagged item by hand, "
                      "re-detect follows in ~1.5 s")
            else:
                print(f"  [demo] 6/7 box shifted by ({dx:+.3f}, {dy:+.3f}) m")
        if d["t"] > 90:
            _demo_phase(d, "detect2")

    elif p == "detect2":
        T_tag = tag_pose_world(scene, tag_size, tag_id)
        if T_tag is None:
            print("  [demo] ABORT: tag not visible after the shift.")
            _demo = None
            return
        d["tagB"] = T_tag
        moved = float(np.linalg.norm(d["tagB"][:3, 3] - d["tagA"][:3, 3]))
        print(f"  [demo] 7/7 tag now at {np.round(d['tagB'][:3, 3], 4).tolist()}  "
              f"(moved {moved:.4f} m)")
        q = _demo_solve(d, scene, robot, d["tagB"], "re-targeting")
        if q is None:
            _demo = None
            return
        _demo_set(d, robot, sim, q)
        _demo_phase(d, "hover2")

    elif p == "hover2":
        if _demo_settled(robot, d["target"]) or d["t"] > 900:
            refB = ee_in_tag_frame(scene, d["tagB"])
            a = d["refA"][:3, 3]
            b = refB[:3, 3]
            err = float(np.linalg.norm(b - a))
            tag_moved = float(np.linalg.norm(d["tagB"][:3, 3] - d["tagA"][:3, 3]))
            print("\n  " + "=" * 66)
            print("  [demo] RESULT")
            print(f"    box/tag moved by        : {tag_moved:.4f} m")
            print(f"    EE in tag frame  before : {np.round(a, 4).tolist()}")
            print(f"    EE in tag frame  after  : {np.round(b, 4).tolist()}")
            print(f"    difference              : {err:.4f} m")
            verdict = "PASS" if err < 0.05 else "FAIL"
            print(f"    {verdict}: arm re-targeted to the same spot relative to the tag")
            print("  " + "=" * 66 + "\n")
            _demo = None


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    global _current_node, _seq

    robot = active_robot(scene)
    sim_dt = sim.get_physics_dt()

    print(f"  [diag] default_joint_pos = {robot.data.default_joint_pos[0].cpu().numpy().tolist()}")

    print("\n" + "=" * 90)
    print("MODEL CHECK  (scale / up-axis / chain)")
    print("=" * 90)
    try:
        import omni.usd
        from pxr import UsdGeom
        _stage = omni.usd.get_context().get_stage()
        print(f"  upAxis         = {UsdGeom.GetStageUpAxis(_stage)}   (Isaac expects Z)")
        print(f"  metersPerUnit  = {UsdGeom.GetStageMetersPerUnit(_stage)}   (expect 1.0)")
    except Exception as e:
        print(f"  stage metadata unavailable: {e}")

    def _chain_report(rb, label):
        print(f"\n  ----- {label} -----")
        _m = rb.data.default_mass[0].cpu().numpy()
        if float(_m.max()) > 1e6:
            print("  !! mass >1e6 kg -> mm authored, read as m (Notes 2.1)")
        print(f"  joints: {rb.joint_names}")
        lo = rb.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
        hi = rb.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
        for _j, (_l, _h) in zip(rb.joint_names, zip(lo, hi)):
            _flag = "  <-- NO LIMITS" if not np.isfinite([_l, _h]).all() else ""
            print(f"    {_j:18s} [{_l:9.4f}, {_h:9.4f}]{_flag}")
        base = scene.env_origins[0].cpu().numpy()
        bp = rb.data.body_pos_w[0].cpu().numpy() - base
        for _n, _p in zip(rb.body_names, bp):
            print(f"    {_n:16s} {np.round(_p, 5).tolist()}")
        print("    link lengths (consecutive body origins):")
        for i in range(len(rb.body_names) - 1):
            d = float(np.linalg.norm(bp[i + 1] - bp[i]))
            print(f"      {rb.body_names[i]:12s} -> {rb.body_names[i+1]:14s} "
                  f"{d*1000:8.1f} mm")
        return bp

    for _k, _s in ARM_SPEC.items():
        _chain_report(scene[_s["key"]], f"{_k}  (scene['{_s['key']}'])")

    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]

    print("\n" + "=" * 90)
    print("ACTUATOR MAPPING")
    print("=" * 90)

    for actuator_name, actuator in robot.actuators.items():
        print(f"  {actuator_name:16s} {actuator.joint_names}  "
              f"indices={actuator.joint_indices.tolist()}")

    print("\n" + "=" * 90)
    print("JOINT PARAMETERS")
    print("=" * 90)

    for joint_id, joint_name in enumerate(robot.joint_names):
        print(
            f"{joint_id}: {joint_name:20s} "
            f"limits=["
            f"{lower[0, joint_id].item():.6f}, "
            f"{upper[0, joint_id].item():.6f}] "
            f"stiffness={robot.data.joint_stiffness[0, joint_id].item():.3f} "
            f"damping={robot.data.joint_damping[0, joint_id].item():.3f} "
            f"effort={robot.data.joint_effort_limits[0, joint_id].item():.3f}"
        )

    global _demo

    for _k, _s in ARM_SPEC.items():
        _rb = scene[_s["key"]]
        _m = _MODELS[_k]
        _m.bind(_rb)

        _rs = _rb.data.default_root_state.clone()
        _rs[:, :3] += scene.env_origins
        _rb.write_root_pose_to_sim(_rs[:, :7])
        _rb.write_root_velocity_to_sim(_rs[:, 7:])

        _hp = torch.as_tensor(
            _m.home_pose, device=_rb.data.default_joint_pos.device
        ).unsqueeze(0).repeat(_rb.data.default_joint_pos.shape[0], 1)
        _rb.write_joint_state_to_sim(_hp, torch.zeros_like(_rb.data.default_joint_vel))
        _rb.set_joint_position_target(_hp)

    _mirror_active_globals()

    reset_position = torch.as_tensor(HOME_POSE, device=robot.data.default_joint_pos.device
                                      ).unsqueeze(0).repeat(robot.data.default_joint_pos.shape[0], 1)

    scene.reset()
    sim.step()
    scene.update(sim_dt)

    target_position = reset_position.clone()
    robot.set_joint_position_target(target_position)

    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    setup_camera_mounts(scene)

    _KNOWN_UNBOUNDED = {"innerlinkjoint", "endeffectorjoint"}
    for _jname, (_lo, _hi) in zip(robot.joint_names, JOINT_LIMITS):
        if not np.isfinite([_lo, _hi]).all() and _jname not in _KNOWN_UNBOUNDED:
            print(f"  [ik] WARNING: {_jname} still has no real limits "
                  f"([-inf,inf]) -- elbow-branch scoring can't validate it")
    print(f"  [ik] analytical PF400 kinematics ready. joint order: {robot.joint_names}")

    joint_index_by_name = {name: i for i, name in enumerate(robot.joint_names)}

    print("\n" + "=" * 90)
    print("INTERACTIVE JOINT CONTROL")
    print("=" * 90)
    print("Joints:", ", ".join(f"{i}:{n}" for i, n in enumerate(robot.joint_names)))
    print("Enter: <joint name or index> <target value>   e.g.  innerlinkjoint 0.5")
    print("       snap [cam]                 -> save a camera frame")
    print("       detect [tag_size_m]        -> run AprilTag detection")
    print("       goal <standoff> [ox] [oy] [tag=N]")
    print("                                  -> hover in front of a tag (standoff >= 0.5).")
    print("                                     No tag=N: uses the only visible tag, or")
    print(f"                                     falls back to tag {GOAL_DEFAULT_TAG} if several are.")
    print("       move <x> <y> <z> [yaw_deg] -> move EE to a base-frame position")
    print("       kincheck                   -> compare FK prediction to actual EE pose")
    print("       jointcheck                 -> print authored joint geometry from USD")
    print("       camcheck                   -> compare how each camera is mounted")
    print("       stop                       -> cancel an active move")
    print("       home / reset               -> return arm to start pose")
    print("       demo [standoff] [shift]    -> full tag-relative re-targeting demo")
    print("       shiftbox [dx dy]           -> nudge the box (random if no args)")
    print("       arms                       -> list both arms, mark the active one")
    print("       use <s1160|xs750>          -> switch which arm commands drive")
    print("Type 'q' to quit.\n")

    cmd_queue: queue.Queue = queue.Queue()
    reader_thread = threading.Thread(target=input_reader, args=(cmd_queue,), daemon=True)
    reader_thread.start()

    tolerance = 1e-3
    active_move = None
    active_ik_move = None
    ik_move_steps = 0

    while simulation_app.is_running():
        sync_ee_cameras(scene)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        try:
            raw = cmd_queue.get_nowait()
        except queue.Empty:
            raw = None

        if raw is not None:
            raw = raw.strip()
            if raw.lower() in ("q", "quit", "exit"):
                break
            elif raw.lower().startswith("use") or raw.lower() == "arms":
                toks = raw.split()
                if raw.lower() == "arms" or len(toks) < 2:
                    for _k, _s in ARM_SPEC.items():
                        _m = _MODELS[_k]
                        print(f"  {'<==' if _k == ACTIVE else '   '} {_k:6s} "
                              f"scene['{_s['key']}']  base={_s['base']}  "
                              f"reach={_m.reach*1000:.1f}mm  "
                              f"rail={'yes' if _m.has_rail else 'no'}  "
                              f"joints={len(_m.arm_joints)}+{len(_m.grip_idx)}")
                elif toks[1].lower() not in ARM_SPEC:
                    print(f"  usage: use <{'|'.join(ARM_SPEC)}>")
                else:
                    active_move = None
                    active_ik_move = None
                    _demo = None
                    robot = set_active_arm(scene, toks[1].lower())
                    target_position = robot.data.joint_pos.clone()
                    robot.set_joint_position_target(target_position)
                    joint_index_by_name = {n: i for i, n
                                           in enumerate(robot.joint_names)}
            elif raw.lower().startswith("snap"):
                toks = raw.split()
                cam_name = toks[1] if len(toks) == 2 else PRIMARY_CAMERA
                result = grab_camera(scene, cam_name)
                if result is not None:
                    rgb, K = result
                    pos = live_camera_matrix(scene, cam_name)[:3, 3] if cam_name in _T_ee_cam else None
                    print(f"[{cam_name}] rgb shape: {rgb.shape}  "
                          f"min/max: {int(rgb.min())}/{int(rgb.max())}")
                    print(f"  fx,fy,cx,cy: {K[0,0]:.2f}, {K[1,1]:.2f}, {K[0,2]:.2f}, {K[1,2]:.2f}")
                    print(f"  cam world pos: {None if pos is None else np.round(pos, 4).tolist()}")
                    from PIL import Image
                    out_path = f"/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/cam/snap_{cam_name}.png"
                    Image.fromarray(rgb).save(out_path)
                    print(f"  saved {out_path}")
                else:
                    print(f"  [{cam_name}] no frame (check the camera name / --enable_cameras)")
            elif raw.lower().startswith("detect"):
                toks = raw.split()
                ts = TAG_SIZE
                if len(toks) == 2:
                    try:
                        ts = float(toks[1])
                    except ValueError:
                        print("  usage: detect [tag_size_meters]")
                detect_tags(scene, ts)
            elif raw.lower() == "stop":
                active_ik_move = None
                _demo = None
                if _seq is not None:
                    print("  [goto] sequence cancelled mid-route -- current node is now "
                          "UNKNOWN, run 'setnode' or 'treehome' before the next goto")
                    _seq = None
                    _current_node = None
                target_position = robot.data.joint_pos.clone()
                robot.set_joint_position_target(target_position)
                print("  [move] cancelled, holding current pose.")
            elif raw.lower().startswith("demo"):
                toks = raw.split()
                try:
                    standoff = float(toks[1]) if len(toks) > 1 else 0.60
                    spread = float(toks[2]) if len(toks) > 2 else 0.12
                except ValueError:
                    print("  usage: demo [standoff_m] [box_shift_m]")
                else:
                    active_ik_move = None
                    active_move = None
                    demo_start(standoff, spread=spread)
            elif raw.lower().startswith("shiftbox"):
                toks = raw.split()
                try:
                    dx = float(toks[1]) if len(toks) > 2 else None
                    dy = float(toks[2]) if len(toks) > 2 else None
                except ValueError:
                    print("  usage: shiftbox [dx dy]   (omit for a random nudge)")
                else:
                    dx, dy = shift_box(scene, dx, dy)
                    if dx is not None:
                        print(f"  [box] shifted by ({dx:+.3f}, {dy:+.3f}) m")
            elif raw.lower().startswith("grip"):
                toks = raw.split()
                if not _GRIP_IDX:
                    print("  [grip] no gripper joints in this model")
                else:
                    lo, hi = JOINT_LIMITS[_GRIP_IDX[0]]
                    arg = toks[1].lower() if len(toks) > 1 else ""
                    if arg in ("open", "o"):
                        val = hi if np.isfinite(hi) else 0.0
                    elif arg in ("close", "c", "shut"):
                        val = lo if np.isfinite(lo) else 0.0
                    else:
                        try:
                            val = float(arg)
                        except ValueError:
                            print(f"  usage: grip <open|close|value>   "
                                  f"(each finger limits {lo:.4f}..{hi:.4f} m)")
                            val = None
                    if val is not None:
                        tp = robot.data.joint_pos.clone()
                        for i in _GRIP_IDX:
                            tp[:, i] = val
                        robot.set_joint_position_target(tp)
                        target_position = tp
                        print(f"  [grip] {', '.join(GRIPPER_JOINTS)} -> {val:.4f} m each")
            elif (raw.lower().split() or [""])[0] in (
                    "tree", "goto", "treehome", "setnode") and ACTIVE != "s1160":
                print(f"  [tree] the taught position tree is s1160 production data "
                      f"(5-axis, rail-relative joint values). Active arm is "
                      f"'{ACTIVE}' with {len(_model.arm_joints)} arm joints and "
                      f"a {_model.reach*1000:.0f}mm reach -- these nodes do not "
                      f"transfer. Run 'use s1160' first.")
            elif raw.lower() == "tree":
                print_tree()
            elif raw.lower().startswith("setnode"):
                toks = raw.split(maxsplit=1)
                nid = node_by_name(toks[1]) if len(toks) > 1 else None
                if nid is None:
                    print("  usage: setnode <name|id>   (declare where the arm already is)")
                else:
                    _current_node = nid
                    print(f"  [node] current node declared as {nid} "
                          f"'{POSITION_TREE[nid][0]}' (no motion)")
            elif raw.lower().startswith("treehome"):
                active_ik_move = None
                active_move = None
                _seq = None
                _current_node = TREE_ROOT
                cmd_arm(robot, sim, node_q(TREE_ROOT))
                print(f"  [goto] DIRECT move to tree root {TREE_ROOT} "
                      f"'{POSITION_TREE[TREE_ROOT][0]}' -- this is the one unguarded move, "
                      f"make sure the arm is clear first")
            elif raw.lower().startswith("goto"):
                toks = raw.split(maxsplit=1)
                nid = node_by_name(toks[1]) if len(toks) > 1 else None
                if nid is None:
                    print("  usage: goto <name|id>   (try 'tree' to list nodes)")
                elif _current_node is None:
                    print("  [goto] current node unknown. Run 'treehome' to drive to the root, "
                          "or 'setnode <name>' if you know where the arm already is.")
                elif nid == _current_node:
                    print(f"  [goto] already at '{POSITION_TREE[nid][0]}'")
                else:
                    active_ik_move = None
                    active_move = None
                    seq_start(tree_route(_current_node, nid), robot, sim)
            elif raw.lower().startswith("calibinfo"):
                calib_info()
            elif raw.lower().startswith("calibrate"):
                toks = raw.lower().split()
                dry = "dry" in toks
                only = next((t for t in toks[1:] if t in CALIB_POSES), None)
                active_ik_move = None
                active_move = None
                calib_start(robot, sim, dry=dry, only=only)
            elif raw.lower() in ("home", "reset"):
                active_ik_move = None
                active_move = None
                _seq = None
                target_position = torch.as_tensor(
                    HOME_POSE, device=robot.data.joint_pos.device
                ).unsqueeze(0).repeat(robot.data.joint_pos.shape[0], 1)
                robot.set_joint_position_target(target_position)
                _current_node = None
                print("  [home] returning to sim HOME_POSE (outerlinkjoint folded to 180deg). "
                      "NOTE: this is NOT tree node 42 -- current node cleared. Use 'treehome'.")
            elif raw.lower() == "camcheck":
                from scipy.spatial.transform import Rotation as R
                for name in CAMERA_NAMES:
                    T = _T_ee_cam.get(name)
                    if T is None:
                        print(f"  [camcheck] {name}: no mount captured")
                        continue
                    eul = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
                    print(f"  [camcheck] {name:14s} pos={np.round(T[:3,3],4).tolist()}")
                    print(f"               rot(xyz deg)={np.round(eul,1).tolist()}  "
                          f"fwd={np.round(T[:3,2],3).tolist()}")
                    Tw = live_camera_matrix(scene, name)
                    fwd_w, up_cam = Tw[:3, 2], -Tw[:3, 1]
                    wu = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], fwd_w) * fwd_w
                    n_wu = np.linalg.norm(wu)
                    if n_wu > 1e-6:
                        wu = wu / n_wu
                        roll = math.degrees(math.atan2(
                            float(np.dot(np.cross(wu, up_cam), fwd_w)),
                            float(np.dot(wu, up_cam))))
                        print(f"               roll about optical axis vs world up: "
                              f"{roll:+.2f} deg")
                fwds = [_T_ee_cam[n][:3, 2] for n in CAMERA_NAMES if n in _T_ee_cam]
                if len(fwds) == 3:
                    for a, b in ((0, 1), (0, 2), (1, 2)):
                        ang = math.degrees(math.acos(
                            max(-1.0, min(1.0, float(np.dot(fwds[a], fwds[b]))))))
                        print(f"  [camcheck] angle {CAMERA_NAMES[a]} vs "
                              f"{CAMERA_NAMES[b]}: {ang:.1f} deg")

                print("  --- TRUE mount check (camera pose from the USD stage; EE pose from "
                      "pf400_fk at the CURRENT joint values, not nominal HOME) ---")
                # Must be FK at the live joint values, not at HOME_POSE. The column
                # droops a few mm under gravity and never sits exactly at its
                # commanded 0.05 m, so comparing against nominal HOME charged that
                # sag to the camera mount -- it showed up as a pure-Z offset error
                # of 8-10 mm that moved when the arm did.
                _q_now = robot.data.joint_pos[0].cpu().numpy()
                hx, hy, hz, hyaw = pf400_fk(_q_now)
                c_, s_ = math.cos(hyaw), math.sin(hyaw)
                T_ee_home = np.array([[c_, -s_, 0, hx],
                                      [s_,  c_, 0, hy],
                                      [0,    0, 1, hz],
                                      [0,    0, 0,  1]])
                T_ee_world_home = np.linalg.inv(T_ee_home)
                for name in CAMERA_NAMES:
                    if name in _WORLD_CAM_POSE:
                        print(f"  [true-mount] {name}: world-fixed, EE mount check N/A")
                        continue
                    cam = scene[name]
                    cp = cam.data.pos_w
                    cp = (cp[0] if cp.ndim == 2 else cp).cpu().numpy()
                    cq = cam.data.quat_w_ros
                    cq = (cq[0] if cq.ndim == 2 else cq).cpu().numpy()
                    T_stage = _stage_world_transform(name)
                    if T_stage is not None:
                        src = "stage"
                        # USD -> ROS camera convention. USD cameras look down -Z
                        # with +Y up; ROS look down +Z with +Y down, which is a
                        # 180 deg rotation about the optical right axis. We SET the
                        # pose with convention="ros" but read the raw USD transform
                        # back, so without this the check reports a spurious 180 deg
                        # error on fwd and down while right matches perfectly --
                        # the signature of a convention mismatch, not a bad mount.
                        T_stage = T_stage.copy()
                        T_stage[:3, :3] = T_stage[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                        T_world_cam_frozen = T_stage
                        if float(np.linalg.norm(cp)) < 1e-6:
                            print(f"  [true-mount] {name:14s} NOTE: cam.data.pos_w is "
                                  f"zeros -- world-spawned cameras driven by "
                                  f"set_world_poses do not populate it. Using the "
                                  f"stage transform instead.")
                    else:
                        src = "cam.data"
                        T_world_cam_frozen = _pose_to_matrix(cp, cq)
                    print(f"  [true-mount] {name:14s} source={src}  "
                          f"world_pos={np.round(T_world_cam_frozen[:3,3],4).tolist()}")
                    T_ee_cam_true = T_ee_world_home @ T_world_cam_frozen
                    fwd_true = T_ee_cam_true[:3, 2]
                    alpha_true = math.degrees(math.atan2(fwd_true[1], fwd_true[0]))
                    assumed_fwd = _CAM_ROT_EE[:, 2]
                    alpha_assumed = math.degrees(math.atan2(assumed_fwd[1], assumed_fwd[0]))
                    diff = alpha_true - alpha_assumed
                    print(f"  [true-mount] {name:14s} true_pos={np.round(T_ee_cam_true[:3,3],4).tolist()}  "
                          f"true_fwd={np.round(fwd_true,4).tolist()}  "
                          f"true_alpha={alpha_true:.3f}deg  assumed_alpha={alpha_assumed:.3f}deg  "
                          f"diff={diff:+.3f}deg")
                    right_true, down_true = T_ee_cam_true[:3, 0], T_ee_cam_true[:3, 1]
                    right_assumed, down_assumed = _CAM_ROT_EE[:, 0], _CAM_ROT_EE[:, 1]
                    right_diff = float(np.linalg.norm(right_true - right_assumed))
                    down_diff = float(np.linalg.norm(down_true - down_assumed))
                    print(f"  [true-mount] {name:14s} true_right={np.round(right_true,4).tolist()}  "
                          f"assumed_right={np.round(right_assumed,4).tolist()}  "
                          f"right_diff(norm)={right_diff:.4f}")
                    print(f"  [true-mount] {name:14s} true_down={np.round(down_true,4).tolist()}   "
                          f"assumed_down={np.round(down_assumed,4).tolist()}   "
                          f"down_diff(norm)={down_diff:.4f}")
                    pos_true = T_ee_cam_true[:3, 3]
                    pos_assumed = _CAM_OFFSET_EE[name]
                    pos_delta = pos_true - pos_assumed
                    print(f"  [true-mount] {name:14s} true_offset={np.round(pos_true,4).tolist()}  "
                          f"assumed_offset={np.round(pos_assumed,4).tolist()}")
                    print(f"  [true-mount] {name:14s} OFFSET DELTA={np.round(pos_delta,4).tolist()}  "
                          f"norm={np.linalg.norm(pos_delta):.4f}m")

                # CONVENTION CHECK -- REMOVED 2026-08-19, it measured nothing.
                # It compared cam.data.quat_w_{ros,world,opengl} against
                # _CAM_ROT_EE. For world-spawned cameras driven by
                # set_world_poses those buffers are never populated and read as
                # identity, so the block was really reporting the ARM's own
                # orientation: 180 deg on the s1160 (yaw 0) and 120 deg on the
                # xs750 (yaw 90), moving with the base rather than the camera.
                # The question it was built to answer is settled anyway: we
                # author in ROS convention and USD stores the 180-deg-about-X
                # rotation of that, which the true-mount check above now undoes
                # explicitly. A diagnostic that always prints a non-zero number
                # is worse than no diagnostic -- it teaches you to skip camcheck.
                print("  --- convention check removed: cam.data quats are not "
                      "populated for world-spawned cameras (see true-mount above) ---")

                print("  --- CAMERA PRIM ATTRIBUTES (authored values, ground truth) ---")
                try:
                    import omni.usd as _ou
                    _st = _ou.get_context().get_stage()
                    for _p in _st.Traverse():
                        if _p.GetName() not in CAMERA_NAMES:
                            continue
                        print(f"  [attr] {_p.GetName()}  path={_p.GetPath()}")
                        for _a in ("focalLength", "horizontalAperture", "verticalAperture",
                                   "horizontalApertureOffset", "verticalApertureOffset",
                                   "projection", "clippingRange", "fStop", "focusDistance"):
                            _at = _p.GetAttribute(_a)
                            if not _at or not _at.IsValid():
                                continue
                            _v = _at.Get()
                            _warn = ""
                            if _a.endswith("ApertureOffset") and _v not in (None, 0.0):
                                _warn = "   <<< NON-ZERO: Isaac Lab reports cx/cy as centred anyway"
                            print(f"  [attr]     {_a:26s} = {_v}{_warn}")
                        _w, _h = 1024.0, 1024.0
                        _f = _p.GetAttribute("focalLength").Get()
                        _ha = _p.GetAttribute("horizontalAperture").Get()
                        _hao = _p.GetAttribute("horizontalApertureOffset")
                        _hao = _hao.Get() if (_hao and _hao.IsValid()) else 0.0
                        if _f and _ha:
                            print(f"  [attr]     implied fx @1024 = {_w * _f / _ha:.2f}")
                            if _hao:
                                print(f"  [attr]     implied cx @1024 = "
                                      f"{_w * 0.5 + _w * _hao / _ha:.2f}   (Isaac reports 512.0)")
                except Exception as _e:
                    print(f"  [attr] read failed: {_e}")

                print("  --- PRIM SCALE CHECK (raw world xform, NOT orthonormalised) ---")
                try:
                    import omni.usd
                    from pxr import UsdGeom
                    _stage = omni.usd.get_context().get_stage()
                    for _prim in _stage.Traverse():
                        if _prim.GetName() in CAMERA_NAMES:
                            _m = UsdGeom.Xformable(_prim).ComputeLocalToWorldTransform(0.0)
                            _raw = np.array([[_m[j][i] for j in range(4)] for i in range(4)])
                            _sc = [float(np.linalg.norm(_raw[:3, k])) for k in range(3)]
                            _flag = "" if max(abs(s - 1.0) for s in _sc) < 1e-4 else "   <<< NOT UNIT SCALE"
                            print(f"  [scale] {_prim.GetName():14s} col norms="
                                  f"{[round(s, 6) for s in _sc]}{_flag}")
                            print(f"  [scale] {_prim.GetName():14s} path={_prim.GetPath()}")
                            _par = _prim.GetParent()
                            while _par and _par.GetPath() != _stage.GetPseudoRoot().GetPath():
                                _pm = UsdGeom.Xformable(_par).ComputeLocalToWorldTransform(0.0)
                                _praw = np.array([[_pm[j][i] for j in range(4)] for i in range(4)])
                                _psc = [float(np.linalg.norm(_praw[:3, k])) for k in range(3)]
                                if max(abs(s - 1.0) for s in _psc) > 1e-4:
                                    print(f"  [scale]   ancestor {str(_par.GetPath()):55s} "
                                          f"scale={[round(s, 6) for s in _psc]}   <<<")
                                _par = _par.GetParent()
                except Exception as _e:
                    print(f"  [scale] check failed: {_e}")

                print("  --- USD stage (REST POSE ONLY: this read never tracks the "
                      "articulation, so it is only comparable right after startup) ---")
                for name in CAMERA_NAMES:
                    T = _stage_world_transform(name)
                    if T is None:
                        print(f"  [stage] {name}: prim not found")
                        continue
                    print(f"  [stage] {name:14s} rest pos={np.round(T[:3,3],4).tolist()}")
            elif raw.lower() == "kincheck":
                q = robot.data.joint_pos[0].cpu().numpy()
                fk_x, fk_y, fk_z, fk_yaw = pf400_fk(q)
                actual = robot.data.body_pos_w[0, _ee_idx].cpu().numpy()
                print(f"  [kincheck] joints(rail,col,inner,outer,eff)={np.round(q, 4).tolist()}")
                print(f"  [kincheck] FK predicts:  x={fk_x:.4f} y={fk_y:.4f} z={fk_z:.4f} "
                      f"yaw={math.degrees(fk_yaw):.2f}deg")
                from scipy.spatial.transform import Rotation as _R
                q_ee = robot.data.body_quat_w[0, _ee_idx].cpu().numpy()
                act_yaw = _R.from_quat([q_ee[1], q_ee[2], q_ee[3], q_ee[0]]).as_euler("xyz")[2]
                yaw_err = _wrap_deg(math.degrees(fk_yaw) - math.degrees(act_yaw))
                print(f"  [kincheck] actual EE:    x={actual[0]:.4f} y={actual[1]:.4f} "
                      f"z={actual[2]:.4f} yaw={math.degrees(act_yaw):.2f}deg")
                print(f"  [kincheck] pos error: {np.linalg.norm([fk_x, fk_y, fk_z] - actual):.4f} m"
                      f"   yaw error: {yaw_err:+.2f} deg")
                ee_eul = _R.from_quat([q_ee[1], q_ee[2], q_ee[3], q_ee[0]]).as_euler(
                    "xyz", degrees=True)
                print(f"  [kincheck] EE body FULL euler(xyz deg)={np.round(ee_eul, 3).tolist()}")
                print(f"  [kincheck] FK assumes pure yaw, i.e. roll=0 pitch=0.  "
                      f"ACTUAL roll={ee_eul[0]:+.3f}  pitch={ee_eul[1]:+.3f}"
                      f"{'   <<< NON-ZERO: _CAM_ROT_EE was calibrated against a frame that does not exist at runtime' if max(abs(ee_eul[0]), abs(ee_eul[1])) > 0.05 else ''}")
                R_body = _R.from_quat([q_ee[1], q_ee[2], q_ee[3], q_ee[0]]).as_matrix()
                c_fk, s_fk = math.cos(act_yaw), math.sin(act_yaw)
                R_yawonly = np.array([[c_fk, -s_fk, 0.0], [s_fk, c_fk, 0.0], [0.0, 0.0, 1.0]])
                frame_gap = math.degrees(math.acos(max(-1.0, min(1.0,
                    (np.trace(R_body @ R_yawonly.T) - 1.0) / 2.0))))
                print(f"  [kincheck] angle between body_quat_w frame and FK pure-yaw frame = "
                      f"{frame_gap:.3f} deg")
                all_pos = robot.data.body_pos_w[0].cpu().numpy()
                for name, pos in zip(robot.body_names, all_pos):
                    print(f"    body {name:12s} pos={np.round(pos, 4).tolist()}")
            elif raw.lower() == "jointcheck":
                import omni.usd
                from pxr import UsdPhysics, UsdGeom, Gf
                stage = omni.usd.get_context().get_stage()
                target_names = {"railjoint", "columnjoint", "innerlinkjoint",
                                 "outerlinkjoint", "endeffectorjoint"}
                found = 0
                for prim in stage.Traverse():
                    if prim.GetName() not in target_names:
                        continue
                    if not prim.HasAPI(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.Joint):
                        continue
                    found += 1
                    joint = UsdPhysics.Joint(prim)
                    b0 = joint.GetBody0Rel().GetTargets()
                    b1 = joint.GetBody1Rel().GetTargets()
                    p0 = joint.GetLocalPos0Attr().Get()
                    p1 = joint.GetLocalPos1Attr().Get()
                    print(f"  [jointcheck] {prim.GetName()}  path={prim.GetPath()}")
                    print(f"    body0={b0}  localPos0={p0}")
                    print(f"    body1={b1}  localPos1={p1}")
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        rj = UsdPhysics.RevoluteJoint(prim)
                        axis_tok = rj.GetAxisAttr().Get()
                        local_axis = {"X": np.array([1.0, 0, 0]),
                                      "Y": np.array([0, 1.0, 0]),
                                      "Z": np.array([0, 0, 1.0])}.get(str(axis_tok), None)
                        if local_axis is not None and b0:
                            body0_name = b0[0].name
                            T = _stage_world_transform(body0_name)
                            if T is not None:
                                world_axis = T[:3, :3] @ local_axis
                                world_axis = world_axis / np.linalg.norm(world_axis)
                                tilt = math.degrees(math.acos(
                                    max(-1.0, min(1.0, abs(float(world_axis[2]))))))
                                print(f"    axis={axis_tok}  world_axis={np.round(world_axis,4).tolist()}  "
                                      f"tilt_from_vertical={tilt:.3f}deg"
                                      f"{'  <-- NOT VERTICAL' if tilt > 0.5 else ''}")
                        from pxr import UsdPhysics as _UP
                        drive = _UP.DriveAPI.Get(prim, "angular")
                        if drive:
                            stiff = drive.GetStiffnessAttr().Get()
                            damp = drive.GetDampingAttr().Get()
                            maxf = drive.GetMaxForceAttr().Get()
                            print(f"    drive:angular  stiffness={stiff}  damping={damp}  "
                                  f"maxForce={maxf}")
                        else:
                            print(f"    drive:angular  <-- MISSING (no DriveAPI authored on this "
                                  f"joint -- PhysX has nothing to actuate it with)")
                if found == 0:
                    print("  [jointcheck] no matching joint prims found under stage root.")
            elif raw.lower().startswith("goal"):
                toks = raw.split()
                # Pull the tag selector out first so it can sit anywhere in the
                # line -- 'goal 0.2 tag=3' and 'goal tag=3 0.2' both work.
                want_tag = None
                rest = []
                for t in toks[1:]:
                    tl = t.lower()
                    if tl.startswith(("tag=", "id=")) or tl.startswith("#"):
                        want_tag = tl.split("=", 1)[-1].lstrip("#")
                    else:
                        rest.append(t)
                try:
                    standoff = float(rest[0]) if len(rest) > 0 else 0.2
                    off_x = float(rest[1]) if len(rest) > 1 else 0.0
                    off_y = float(rest[2]) if len(rest) > 2 else 0.0
                    if len(rest) > 3 and want_tag is None:
                        want_tag = rest[3]
                    want_tag = int(want_tag) if want_tag is not None else None
                except ValueError:
                    print("  usage: goal <standoff_m> [off_x] [off_y] [tag=N]")
                else:
                    pose = build_goal_pose(scene, TAG_SIZE, standoff, off_x, off_y,
                                           tag_id=want_tag)
                    if pose is not None:
                        x, y, z, yaw = pose
                        try:
                            q_target = pf400_ik_auto(x, y, z, yaw,
                                                q_current=robot.data.joint_pos[0].cpu().numpy())
                        except ValueError as e:
                            print(f"  [goal] IK failed: {e}")
                        else:
                            active_ik_move = cmd_arm(robot, sim, q_target)
                            active_move = None
                            ik_move_steps = 0
                            print(f"  [goal] target xyz={np.round([x,y,z],3).tolist()} "
                                  f"yaw={math.degrees(yaw):.1f}deg -> "
                                  f"joints={np.round(q_target, 3).tolist()}")
            elif raw.lower().startswith("move"):
                toks = raw.split()
                if len(toks) not in (4, 5):
                    print("  usage: move <x> <y> <z> [yaw_deg]   (base-frame meters)")
                else:
                    try:
                        x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
                        yaw = math.radians(float(toks[4])) if len(toks) == 5 else 0.0
                    except ValueError:
                        print("  usage: move <x> <y> <z> [yaw_deg]   (base-frame meters)")
                    else:
                        try:
                            q_target = pf400_ik_auto(x, y, z, yaw,
                                                q_current=robot.data.joint_pos[0].cpu().numpy())
                        except ValueError as e:
                            print(f"  [move] IK failed: {e}")
                        else:
                            active_ik_move = cmd_arm(robot, sim, q_target)
                            active_move = None
                            ik_move_steps = 0
                            print(f"  [move] target xyz={[x,y,z]} yaw={math.degrees(yaw):.1f}deg "
                                  f"-> joints={np.round(q_target, 3).tolist()}")
            else:   
                parts = raw.split()
                if len(parts) != 2:
                    print("  Format: <joint name or index> <value>")
                else:
                    joint_key, value_str = parts

                    if joint_key in joint_index_by_name:
                        joint_id = joint_index_by_name[joint_key]
                    else:
                        joint_id = None
                        if joint_key.lstrip("-").isdigit():
                            idx = int(joint_key)
                            if 0 <= idx < len(robot.joint_names):
                                joint_id = idx
                        if joint_id is None:
                            print(f"  Unknown joint '{joint_key}'.")

                    if joint_id is not None:
                        try:
                            value = float(value_str)
                        except ValueError:
                            print(f"  '{value_str}' is not a number.")
                            value = None

                        if value is not None:
                            joint_lower = lower[0, joint_id].item()
                            joint_upper = upper[0, joint_id].item()
                            if not (joint_lower <= value <= joint_upper):
                                print(f"  {robot.joint_names[joint_id]} range is "
                                    f"[{joint_lower:.4f}, {joint_upper:.4f}] — clamping.")
                                value = max(joint_lower, min(joint_upper, value))

                            active_ik_move = None
                            if _seq is not None:
                                _seq = None
                                print("  [goto] route cancelled by manual joint move")
                            target_position = robot.data.joint_pos.clone()
                            target_position[:, joint_id] = value
                            robot.set_joint_position_target(target_position)
                            active_move = (joint_id, value)
                            here = (f" from node {_current_node} "
                                    f"'{POSITION_TREE[_current_node][0]}'"
                                    if _current_node in POSITION_TREE else "")
                            print(f"  Moving {robot.joint_names[joint_id]} -> {value:.4f}"
                                  f"{here} (other joints held) ...")

        if active_move is not None:
            joint_id, value = active_move
            current = robot.data.joint_pos[0, joint_id].item()
            velocity = robot.data.joint_vel[0, joint_id].item()
            if abs(current - value) < tolerance and abs(velocity) < tolerance:
                print(f"  Settled: {robot.joint_names[joint_id]} = {current:.4f}\n")
                active_move = None

        if _seq is not None:
            seq_step(scene, sim)

        if _calib is not None:
            calib_step(scene, sim)

        if _demo is not None:
            demo_step(scene, sim, TAG_SIZE)
            if _demo is None:
                target_position = robot.data.joint_pos.clone()

        if active_ik_move is not None:
            ik_move_steps += 1
            current = robot.data.joint_pos[0].cpu().numpy()
            velocity = robot.data.joint_vel[0].cpu().numpy()
            pos_ok = np.all(np.abs(current - active_ik_move) < tolerance)
            vel_ok = np.all(np.abs(velocity) < tolerance)
            if pos_ok and vel_ok:
                print(f"  [move] settled ({ik_move_steps} steps): "
                      f"joints={np.round(current, 4).tolist()}\n")
                active_ik_move = None
            elif ik_move_steps > 3000:
                print(f"  [move] timed out after {ik_move_steps} steps — "
                      f"a joint may be at its limit. current={np.round(current, 4).tolist()}\n")
                active_ik_move = None

    print("[INFO] Interactive session ended.")

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[3.5, 0.0, 3.2],
        target=[0.0, 0.0, 0.5],
    )

    scene_cfg = NewRobotsSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO]: Setup complete.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()