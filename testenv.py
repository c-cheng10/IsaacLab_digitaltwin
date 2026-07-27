import argparse
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg


PF400_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            "/home/sdl6/simulation/IsaacLab/source/"
            "isaaclab/isaaclab/pf400/PF400revisedv2.usd"
        ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),

    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=None,
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
    ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.0),
        joint_pos={
            "railjoint": 0.0,
            "columnjoint": 0.0,
            "shoulderjoint": 0.0,
            "innerlinkjoint": 0.0,
            "outerlinkjoint": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "linear_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "railjoint",
                "columnjoint",
            ],
            effort_limit_sim=10000.0,
            velocity_limit_sim=1.0,
            stiffness=10000.0,
            damping=1000.0,
        ),

        "rotary_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulderjoint",
                "innerlinkjoint",
                "outerlinkjoint",
            ],
            effort_limit_sim=5000.0,
            velocity_limit_sim=2.0,
            stiffness=5000.0,
            damping=500.0,
        ),
    },
)


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

    pf400: ArticulationCfg = PF400_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dummy_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/dummy_cam",
        spawn=sim_utils.PinholeCameraCfg(),
        update_period=0.1,
        height=64,
        width=64,
        data_types=["rgb"],
    )

    left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/endeffector/left_camera",
        spawn=None,
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
    )

    right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/endeffector/right_camera",
        spawn=None,
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
    )

    middle_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot/endeffector/middle_camera",
        spawn=None,
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
    )

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
            usd_path="/home/sdl6/simulation/IsaacLab/source/isaaclab/isaaclab/pf400/AprilTag_ID2.usd",
            scale=(0.005,0.005,0.005),
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.282, 0.3), rot=(0.7071,-0.7071,0,0)),
    )


def input_reader(q: queue.Queue):
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        q.put(line)


CAMERA_NAMES = ["left_camera", "right_camera", "middle_camera"]


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


TAG_SIZE = 0.12012

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from pyapriltags import Detector
        _detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            refine_edges=True,
        )
    return _detector


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
    robot = scene["pf400"]
    return _pose_to_matrix(
        robot.data.body_pos_w[0, _ee_idx].cpu().numpy(),
        robot.data.body_quat_w[0, _ee_idx].cpu().numpy(),
    )


def calibrate_camera_mount(scene):
    global _ee_idx, _T_ee_cam
    robot = scene["pf400"]

    if "endeffector" not in robot.body_names:
        print("  [mount] 'endeffector' not in body_names:", robot.body_names)
        return
    _ee_idx = robot.body_names.index("endeffector")

    T_ee_world = np.linalg.inv(_ee_matrix(scene))
    for name in CAMERA_NAMES:
        cam = scene[name]
        cam_pos = cam.data.pos_w
        cam_pos = (cam_pos[0] if cam_pos.ndim == 2 else cam_pos).cpu().numpy()
        cam_quat = cam.data.quat_w_ros
        cam_quat = (cam_quat[0] if cam_quat.ndim == 2 else cam_quat).cpu().numpy()
        _T_ee_cam[name] = T_ee_world @ _pose_to_matrix(cam_pos, cam_quat)
    print(f"  [mount] captured offsets for: {list(_T_ee_cam.keys())}")


def live_camera_matrix(scene, cam_name):
    return _ee_matrix(scene) @ _T_ee_cam[cam_name]


def detect_one_camera(scene, cam_name, tag_size, detector):
    result = grab_camera(scene, cam_name)
    if result is None:
        print(f"  [{cam_name}] no frame yet")
        return []
    rgb, K = result

    T_world_cam = live_camera_matrix(scene, cam_name)

    gray = np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dets = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(float(fx), float(fy), float(cx), float(cy)),
        tag_size=tag_size,
    )

    from PIL import Image, ImageDraw
    vis = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(vis)

    out = []
    if not dets:
        print(f"  [{cam_name}] no tags detected")
    for d in dets:
        t_cam = np.asarray(d.pose_t).flatten()
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = d.pose_R
        T_cam_tag[:3, 3] = t_cam
        T = T_world_cam @ T_cam_tag
        out.append((int(d.tag_id), T, float(d.pose_err)))
        print(f"  [{cam_name}] tag {d.tag_id}: dist={np.linalg.norm(t_cam):.4f}m  "
              f"world={np.round(T[:3, 3], 4).tolist()}")

        corners = [tuple(map(float, c)) for c in np.asarray(d.corners)]
        draw.line(corners + [corners[0]], fill=(0, 255, 0), width=3)
        cx_, cy_ = float(d.center[0]), float(d.center[1])
        draw.ellipse([cx_ - 4, cy_ - 4, cx_ + 4, cy_ + 4], fill=(255, 0, 0))
        draw.text((cx_ + 6, cy_ - 10), f"ID {d.tag_id}", fill=(255, 255, 0))

    vis.save(f"/home/sdl6/simulation/IsaacLab/scripts/digitaltwin/cam/detect_{cam_name}.png")
    return out


def fused_tag_poses(scene, tag_size):
    from scipy.spatial.transform import Rotation as R
    detector = get_detector()

    by_tag = {}
    for name in CAMERA_NAMES:
        for tag_id, T, _err in detect_one_camera(scene, name, tag_size, detector):
            by_tag.setdefault(tag_id, []).append(T)

    fused = {}
    for tag_id, mats in by_tag.items():
        positions = np.stack([T[:3, 3] for T in mats])
        pos_avg = positions.mean(axis=0)
        rot_avg = R.from_matrix(np.stack([T[:3, :3] for T in mats])).mean().as_matrix()
        spread = float(np.linalg.norm(positions - pos_avg, axis=1).max()) if len(mats) > 1 else 0.0
        F = np.eye(4)
        F[:3, :3] = rot_avg
        F[:3, 3] = pos_avg
        fused[tag_id] = (F, len(mats), spread)
    return fused


def detect_tags(scene, tag_size):
    from scipy.spatial.transform import Rotation as R
    for tag_id, (F, n, spread) in fused_tag_poses(scene, tag_size).items():
        euler = R.from_matrix(F[:3, :3]).as_euler("xyz", degrees=True)
        print(f"  FUSED tag {tag_id} ({n} cam): pos={np.round(F[:3, 3], 4).tolist()}  "
              f"rot(xyz deg)={np.round(euler, 1).tolist()}  cam_spread={spread:.4f}m")


import math


_P_SHOULDER = np.array([0.0, -0.20300, 0.882])
_P_ELBOW    = np.array([0.0, -0.50550, 0.802])
_P_WRIST    = np.array([0.0, -0.22208, 0.817])
_P_EE       = np.array([0.0,  0.00000, 0.000])


def _link_len_xy(a, b):
    return float(np.linalg.norm((b - a)[:2]))


def _link_ang_deg(a, b):
    d = b - a
    return math.degrees(math.atan2(d[1], d[0]))


class PF400Kinematics:
    def __init__(self):
        self.shoulder_length = _link_len_xy(_P_SHOULDER, _P_ELBOW)
        self.elbow_length = _link_len_xy(_P_ELBOW, _P_WRIST)
        self.end_effector_length = _link_len_xy(_P_WRIST, _P_EE)

    @staticmethod
    def normalize_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def normalize_phi(self, angle):
        while angle < -360:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle

    def forward_kinematics(self, joint_states: list):
        adjusted_angle_j3 = joint_states[2] - 360 if joint_states[2] > 180 else joint_states[2]

        shoulder_angle = math.radians(joint_states[1])
        elbow_angle = math.radians(joint_states[2])
        gripper_angle = math.radians(joint_states[3])

        x = (
            self.shoulder_length * math.cos(shoulder_angle)
            + self.elbow_length * math.cos(shoulder_angle + elbow_angle)
            + self.end_effector_length
            * math.cos(shoulder_angle + elbow_angle + gripper_angle)
        )
        y = (
            self.shoulder_length * math.sin(shoulder_angle)
            + self.elbow_length * math.sin(shoulder_angle + elbow_angle)
            + self.end_effector_length
            * math.sin(shoulder_angle + elbow_angle + gripper_angle)
        )
        z = joint_states[0]
        phi = (
            math.degrees(shoulder_angle)
            + adjusted_angle_j3
            + math.degrees(gripper_angle)
        )

        if 0 < abs(phi) < 540:
            yaw = phi % 360
        elif 540 <= abs(phi) < 720:
            yaw = (phi % 360) - 360
        elif 720 <= abs(phi) < 900:
            yaw = phi % 720
        elif 900 <= abs(phi) < 1080:
            yaw = (phi % 720) - 720
        else:
            yaw = phi % 360
        yaw = self.normalize_angle(yaw)

        rail_length = joint_states[5] if len(joint_states) > 5 else 0
        x_adjusted = x - rail_length

        return [round(x_adjusted, 6), round(y, 6), round(z, 6), round(yaw, 6)], round(phi, 6), round(rail_length, 6)

    def inverse_kinematics(self, cartesian_coordinates: list, phi: float, rail: float = 0.0,
                           elbow: str = None):
        Joint_1 = cartesian_coordinates[2]
        xe = cartesian_coordinates[0] - rail
        ye = cartesian_coordinates[1]

        phi = self.normalize_phi(phi)
        phi_e = math.radians(phi)

        x_second_joint = xe - self.end_effector_length * math.cos(phi_e)
        y_second_joint = ye - self.end_effector_length * math.sin(phi_e)

        radius = math.sqrt(x_second_joint**2 + y_second_joint**2)
        if radius == 0:
            raise ValueError("Radius is zero. The Cartesian coordinates may be invalid.")

        acos_input = (
            radius**2 + self.shoulder_length**2 - self.elbow_length**2
        ) / (2 * radius * self.shoulder_length)
        acos_input = max(-1, min(1, acos_input))
        gamma = math.acos(acos_input)

        theta2 = math.pi - math.acos(
            max(-1, min(1, (
                self.shoulder_length**2 + self.elbow_length**2 - radius**2
            ) / (2 * self.shoulder_length * self.elbow_length)))
        )

        theta1 = math.atan2(y_second_joint, x_second_joint) - gamma
        theta3 = phi_e - theta1 - theta2

        if elbow is None:
            elbow = "up" if ye >= 0 else "down"
        if elbow == "up":
            Joint_2 = math.degrees(theta1)
            Joint_3 = math.degrees(theta2)
        else:
            Joint_2 = math.degrees(theta1 + 2 * gamma)
            Joint_3 = math.degrees(-theta2) + 360

        Joint_4 = phi - Joint_2 - Joint_3

        return [round(Joint_1, 6), round(Joint_2, 6), round(Joint_3, 6), round(Joint_4, 6), round(rail, 6)]


_kin = PF400Kinematics()

_A1_REST = _link_ang_deg(_P_SHOULDER, _P_ELBOW)
_A2_REST = _link_ang_deg(_P_ELBOW, _P_WRIST)
_A3_REST = _link_ang_deg(_P_WRIST, _P_EE)

J2_OFFSET = _A1_REST
J3_OFFSET = _A2_REST - _A1_REST
J4_OFFSET = _A3_REST - _A2_REST

_BASE_XY = _P_SHOULDER[:2]
_Z_REST = float(_P_EE[2])

RAIL_AXIS = 1
RAIL_SIGN = 1.0


def pf400_fk(joint_pos_5):
    rail, col, sh, inn, out = joint_pos_5
    joint_states = [
        col,
        math.degrees(sh) + J2_OFFSET,
        math.degrees(inn) + J3_OFFSET,
        math.degrees(out) + J4_OFFSET,
        0,
        0.0,
    ]
    (xa, ya, _z, _yaw_deg), phi, _r = _kin.forward_kinematics(joint_states)

    x = _BASE_XY[0] + xa
    y = _BASE_XY[1] + ya
    if RAIL_AXIS == 0:
        x += RAIL_SIGN * rail
    else:
        y += RAIL_SIGN * rail
    z = _Z_REST + col
    return x, y, z, math.radians(phi - _A3_REST)


JOINT_LIMITS = None


def _wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


def pf400_ik(x, y, z, yaw_rad, rail=0.0, q_current=None, quiet=False):
    xr = x - _BASE_XY[0]
    yr = y - _BASE_XY[1]
    if RAIL_AXIS == 0:
        xr -= RAIL_SIGN * rail
    else:
        yr -= RAIL_SIGN * rail
    phi = math.degrees(yaw_rad) + _A3_REST

    candidates = []
    for elbow in ("up", "down"):
        try:
            J1, J2, J3, J4, _ = _kin.inverse_kinematics(
                [xr, yr, z - _Z_REST], phi, rail=0.0, elbow=elbow
            )
        except ValueError:
            continue
        q = np.array([
            rail, J1,
            math.radians(_wrap_deg(J2 - J2_OFFSET)),
            math.radians(_wrap_deg(J3 - J3_OFFSET)),
            math.radians(_wrap_deg(J4 - J4_OFFSET)),
        ], dtype=np.float32)

        if JOINT_LIMITS is None:
            violation = 0.0
        else:
            violation = sum(max(0.0, lo - v) + max(0.0, v - hi)
                            for v, (lo, hi) in zip(q, JOINT_LIMITS))
        travel = 0.0 if q_current is None else float(np.abs(q - np.asarray(q_current)).sum())
        candidates.append((violation, travel, elbow, q))

    if not candidates:
        raise ValueError("target unreachable (no IK solution)")

    violation, _travel, elbow, q = min(candidates, key=lambda c: (c[0] > 1e-9, c[1]))
    if violation > 1e-9 and not quiet:
        print(f"  [ik] warning: best solution ({elbow}) is outside joint limits by "
              f"{violation:.3f} rad total -- the arm will be clamped and miss.")
    return q, violation


def pf400_ik_auto(x, y, z, yaw_rad, q_current=None, samples=61):
    lo, hi = JOINT_LIMITS[0] if JOINT_LIMITS else (-0.483389, 0.683126)
    cur_rail = float(q_current[0]) if q_current is not None else 0.0

    best = None
    for rail in np.linspace(lo, hi, samples):
        try:
            q, violation = pf400_ik(x, y, z, yaw_rad, rail=float(rail),
                                    q_current=q_current, quiet=True)
        except ValueError:
            continue
        if violation > 1e-9:
            continue
        cost = abs(float(rail) - cur_rail)
        if best is None or cost < best[0]:
            best = (cost, q)

    if best is None:
        raise ValueError(
            f"unreachable: no rail position in [{lo:.3f}, {hi:.3f}] puts "
            f"({x:.3f}, {y:.3f}, {z:.3f}) inside the arm's workspace"
        )
    return best[1]


def goal_pose_from_tag(T_world_tag, standoff, off_x, off_y, cam_name="middle_camera"):
    R = T_world_tag[:3, :3]
    p = T_world_tag[:3, 3]
    n = R[:, 2]

    base = np.array([_BASE_XY[0], _BASE_XY[1], p[2]])
    s = standoff
    if np.linalg.norm(p + s * n - base) > np.linalg.norm(p - s * n - base):
        s = -s
    cam_target = p + s * n + off_x * R[:, 0] + off_y * R[:, 1]

    T_ec = _T_ee_cam[cam_name]
    fwd_ee = T_ec[:3, 2]
    alpha = math.atan2(fwd_ee[1], fwd_ee[0])
    d = p - cam_target
    theta = math.atan2(d[1], d[0]) - alpha

    off_ee = T_ec[:3, 3]
    c, sn = math.cos(theta), math.sin(theta)
    rotated = np.array([c * off_ee[0] - sn * off_ee[1],
                        sn * off_ee[0] + c * off_ee[1],
                        off_ee[2]])
    ee_pos = cam_target - rotated
    return float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]), float(theta)


def build_goal_pose(scene, tag_size, standoff, off_x, off_y, tag_id=2):
    fused = fused_tag_poses(scene, tag_size)
    if tag_id not in fused:
        print(f"  [goal] tag {tag_id} not detected")
        return None
    return goal_pose_from_tag(fused[tag_id][0], standoff, off_x, off_y)


def ee_in_tag_frame(scene, T_world_tag):
    return np.linalg.inv(T_world_tag) @ _ee_matrix(scene)


def shift_box(scene, dx=None, dy=None, spread=0.12):
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


def _demo_settled(robot, target, tol=2.5e-2, vel_tol=5e-3):
    q = robot.data.joint_pos[0].cpu().numpy()
    v = robot.data.joint_vel[0].cpu().numpy()
    return bool(np.all(np.abs(q - target) < tol) and np.all(np.abs(v) < vel_tol))


def _demo_set(d, robot, sim, q_target):
    d["target"] = np.asarray(q_target, dtype=np.float32)
    robot.set_joint_position_target(
        torch.as_tensor(d["target"], device=sim.device).unsqueeze(0))


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
    robot = scene["pf400"]
    d["t"] += 1
    p = d["phase"]

    if p == "home1":
        if d["t"] == 1:
            print("  [demo] 1/7 moving to the safe home pose...")
            _demo_set(d, robot, sim, np.zeros(5, dtype=np.float32))
        if _demo_settled(robot, d["target"]) or d["t"] > 600:
            _demo_phase(d, "detect1")

    elif p == "detect1":
        fused = fused_tag_poses(scene, tag_size)
        if tag_id not in fused:
            print("  [demo] ABORT: tag not visible from home. Move the box into "
                  "view or choose a different safe pose.")
            _demo = None
            return
        d["tagA"] = fused[tag_id][0]
        print(f"  [demo] 2/7 tag at {np.round(d['tagA'][:3, 3], 4).tolist()}  "
              f"(cam_spread {fused[tag_id][2]:.4f} m)")
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
            _demo_set(d, robot, sim, np.zeros(5, dtype=np.float32))
        if _demo_settled(robot, d["target"]) or d["t"] > 600:
            _demo_phase(d, "shift")

    elif p == "shift":
        if d["t"] == 1:
            dx, dy = shift_box(scene, spread=d["spread"])
            print(f"  [demo] 6/7 box shifted by ({dx:+.3f}, {dy:+.3f}) m")
        if d["t"] > 90:
            _demo_phase(d, "detect2")

    elif p == "detect2":
        fused = fused_tag_poses(scene, tag_size)
        if tag_id not in fused:
            print("  [demo] ABORT: tag not visible after the shift.")
            _demo = None
            return
        d["tagB"] = fused[tag_id][0]
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

    robot = scene["pf400"]
    sim_dt = sim.get_physics_dt()

    print(f"  [diag] default_joint_pos = {robot.data.default_joint_pos[0].cpu().numpy().tolist()}")

    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]

    print("\n" + "=" * 90)
    print("ACTUATOR MAPPING")
    print("=" * 90)

    for actuator_name, actuator in robot.actuators.items():
        print("linear_joints matched:", robot.actuators["linear_joints"].joint_names)
        print("rotary_joints matched:", robot.actuators["rotary_joints"].joint_names)
        print(f"\nActuator: {actuator_name}")
        print(f"  joint names:   {actuator.joint_names}")
        print(f"  joint indices: {actuator.joint_indices}")

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

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    reset_position = torch.zeros_like(robot.data.default_joint_pos)
    reset_velocity = torch.zeros_like(robot.data.default_joint_vel)

    robot.write_joint_state_to_sim(
        reset_position,
        reset_velocity,
    )

    scene.reset()
    sim.step()
    scene.update(sim_dt)

    target_position = reset_position.clone()
    robot.set_joint_position_target(target_position)

    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    calibrate_camera_mount(scene)

    global JOINT_LIMITS, _demo
    JOINT_LIMITS = [(lower[0, i].item(), upper[0, i].item())
                    for i in range(len(robot.joint_names))]
    print(f"  [ik] analytical PF400 kinematics ready. joint order: {robot.joint_names}")
    print(f"  [ik] L1={_kin.shoulder_length:.4f} L2={_kin.elbow_length:.4f} "
          f"L3={_kin.end_effector_length:.4f}  offsets(deg) J2={J2_OFFSET:.0f} "
          f"J3={J3_OFFSET:.0f} J4={J4_OFFSET:.0f}")

    joint_index_by_name = {name: i for i, name in enumerate(robot.joint_names)}

    print("\n" + "=" * 90)
    print("INTERACTIVE JOINT CONTROL")
    print("=" * 90)
    print("Joints:", ", ".join(f"{i}:{n}" for i, n in enumerate(robot.joint_names)))
    print("Enter: <joint name or index> <target value>   e.g.  shoulderjoint 0.5")
    print("       snap [cam]                 -> save a camera frame")
    print("       detect [tag_size_m]        -> run AprilTag detection")
    print("       goal <standoff> [ox] [oy]  -> hover in front of the tag (standoff >= 0.5)")
    print("       move <x> <y> <z> [yaw_deg] -> move EE to a base-frame position")
    print("       kincheck                   -> compare FK prediction to actual EE pose")
    print("       jointcheck                 -> print authored joint geometry from USD")
    print("       camcheck                   -> compare how each camera is mounted")
    print("       stop                       -> cancel an active move")
    print("       home / reset               -> return arm to start pose")
    print("       demo [standoff] [shift]    -> full tag-relative re-targeting demo")
    print("       shiftbox [dx dy]           -> nudge the box (random if no args)")
    print("Type 'q' to quit.\n")

    cmd_queue: queue.Queue = queue.Queue()
    reader_thread = threading.Thread(target=input_reader, args=(cmd_queue,), daemon=True)
    reader_thread.start()

    tolerance = 1e-3
    active_move = None
    active_ik_move = None
    ik_move_steps = 0

    while simulation_app.is_running():
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
            elif raw.lower().startswith("snap"):
                toks = raw.split()
                cam_name = toks[1] if len(toks) == 2 else "middle_camera"
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
                    print(f"  [box] shifted by ({dx:+.3f}, {dy:+.3f}) m")
            elif raw.lower() in ("home", "reset"):
                active_ik_move = None
                active_move = None
                target_position = torch.zeros_like(robot.data.joint_pos)
                robot.set_joint_position_target(target_position)
                print("  [home] returning to literal zero pose.")
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
                fwds = [_T_ee_cam[n][:3, 2] for n in CAMERA_NAMES if n in _T_ee_cam]
                if len(fwds) == 3:
                    for a, b in ((0, 1), (0, 2), (1, 2)):
                        ang = math.degrees(math.acos(
                            max(-1.0, min(1.0, float(np.dot(fwds[a], fwds[b]))))))
                        print(f"  [camcheck] angle {CAMERA_NAMES[a]} vs "
                              f"{CAMERA_NAMES[b]}: {ang:.1f} deg")

                print("  --- from the USD stage (authoritative) ---")
                import omni.usd
                from pxr import UsdGeom
                stage = omni.usd.get_context().get_stage()
                looks = {}
                for prim in stage.Traverse():
                    if prim.GetName() not in CAMERA_NAMES:
                        continue
                    xf = UsdGeom.Xformable(prim)
                    m = xf.ComputeLocalToWorldTransform(0.0)
                    t = m.ExtractTranslation()
                    rot = np.array([[m[i][j] for j in range(3)] for i in range(3)])
                    look = -rot[2] / (np.linalg.norm(rot[2]) or 1.0)
                    looks[prim.GetName()] = look
                    print(f"  [stage] {prim.GetName():14s} path={prim.GetPath()}")
                    print(f"           world pos={np.round([t[0], t[1], t[2]], 4).tolist()}")
                    print(f"           look dir ={np.round(look, 3).tolist()}")
                    print(f"           xform ops="
                          f"{[op.GetOpName() for op in xf.GetOrderedXformOps()]}")
                if len(looks) == 3:
                    print("  --- verdict ---")
                    for a, b in ((0, 1), (0, 2), (1, 2)):
                        na, nb = CAMERA_NAMES[a], CAMERA_NAMES[b]
                        ang = math.degrees(math.acos(
                            max(-1.0, min(1.0, float(np.dot(looks[na], looks[nb]))))))
                        print(f"  [stage] {na} vs {nb}: {ang:.1f} deg apart")
            elif raw.lower() == "kincheck":
                q = robot.data.joint_pos[0].cpu().numpy()
                fk_x, fk_y, fk_z, fk_yaw = pf400_fk(q)
                actual = robot.data.body_pos_w[0, _ee_idx].cpu().numpy()
                print(f"  [kincheck] joints(rail,col,sh,in,out)={np.round(q, 4).tolist()}")
                print(f"  [kincheck] FK predicts:  x={fk_x:.4f} y={fk_y:.4f} z={fk_z:.4f} "
                      f"yaw={math.degrees(fk_yaw):.2f}deg")
                print(f"  [kincheck] actual EE:    x={actual[0]:.4f} y={actual[1]:.4f} "
                      f"z={actual[2]:.4f}")
                print(f"  [kincheck] pos error: {np.linalg.norm([fk_x, fk_y, fk_z] - actual):.4f} m")
                all_pos = robot.data.body_pos_w[0].cpu().numpy()
                for name, pos in zip(robot.body_names, all_pos):
                    print(f"    body {name:12s} pos={np.round(pos, 4).tolist()}")
            elif raw.lower() == "jointcheck":
                import omni.usd
                from pxr import UsdPhysics
                stage = omni.usd.get_context().get_stage()
                target_names = {"railjoint", "columnjoint", "shoulderjoint",
                                 "innerlinkjoint", "outerlinkjoint"}
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
                if found == 0:
                    print("  [jointcheck] no matching joint prims found under stage root.")
            elif raw.lower().startswith("goal"):
                toks = raw.split()
                try:
                    standoff = float(toks[1]) if len(toks) > 1 else 0.2
                    off_x = float(toks[2]) if len(toks) > 2 else 0.0
                    off_y = float(toks[3]) if len(toks) > 3 else 0.0
                except ValueError:
                    print("  usage: goal <standoff_m> [off_x] [off_y]")
                else:
                    pose = build_goal_pose(scene, TAG_SIZE, standoff, off_x, off_y)
                    if pose is not None:
                        x, y, z, yaw = pose
                        try:
                            q_target = pf400_ik_auto(x, y, z, yaw,
                                                q_current=robot.data.joint_pos[0].cpu().numpy())
                        except ValueError as e:
                            print(f"  [goal] IK failed: {e}")
                        else:
                            active_ik_move = q_target
                            active_move = None
                            ik_move_steps = 0
                            robot.set_joint_position_target(
                                torch.as_tensor(q_target, device=sim.device).unsqueeze(0)
                            )
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
                            active_ik_move = q_target
                            active_move = None
                            ik_move_steps = 0
                            robot.set_joint_position_target(
                                torch.as_tensor(q_target, device=sim.device).unsqueeze(0)
                            )
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
                            target_position[:, joint_id] = value
                            robot.set_joint_position_target(target_position)
                            active_move = (joint_id, value)
                            print(f"  Moving {robot.joint_names[joint_id]} -> {value:.4f} ...")

        if active_move is not None:
            joint_id, value = active_move
            current = robot.data.joint_pos[0, joint_id].item()
            velocity = robot.data.joint_vel[0, joint_id].item()
            if abs(current - value) < tolerance and abs(velocity) < tolerance:
                print(f"  Settled: {robot.joint_names[joint_id]} = {current:.4f}\n")
                active_move = None

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