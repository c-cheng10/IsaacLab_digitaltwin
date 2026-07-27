# PF400 digital twin

A simulation of a PF400 SCARA robot arm in NVIDIA Isaac Sim / Isaac Lab. Three wrist-mounted
cameras detect an AprilTag on a box, and when the box moves, the arm re-detects the tag and
recomputes where to go instead of driving to a hardcoded coordinate.

![demo](docs/demo.gif)

## What it does

The arm starts at a safe pose, hovers in front of a tagged box, and returns home. If someone moves
the box, the arm sees the tag in its new position and puts itself back in the same place *relative
to the tag*. No world coordinates are stored anywhere in that loop.

```
> demo

  [demo] 1/7 moving to the safe home pose...
  [demo] 2/7 tag at [0.3009, -1.7183, 0.7998]  (cam_spread 0.0400 m)
  [demo] 3/7 hovering -> xyz=[0.2701, -0.8584, 0.0631] yaw=3.4deg
  [demo] 4/7 in position. EE in TAG frame = [...]
  [demo] 5/7 returning to the safe pose...
  [demo] 6/7 box shifted by (+0.087, -0.041) m
  [demo] 7/7 tag now at [0.3879, -1.7593, 0.7998]  (moved 0.0962 m)
  [demo] re-targeting -> xyz=[...]

  [demo] RESULT
    box/tag moved by        : 0.0962 m
    EE in tag frame  before : [...]
    EE in tag frame  after  : [...]
    difference              : 0.0xxx m
    PASS: arm re-targeted to the same spot relative to the tag
```

## How it works

**Perception.** Each camera renders a frame, `pyapriltags` solves the tag's pose from its corner
pixels using the camera's live intrinsics, and the result is transformed into world coordinates by
chaining through the robot's kinematics:

```
T_world_tag = T_world_camera · T_camera_tag
```

Nothing reads ground-truth object positions out of the simulator. `T_camera_tag` comes from vision
and `T_world_camera` comes from joint state, both of which a real robot also has, so the same code
path runs on hardware. The three cameras are fused with proper rotation averaging (SVD, not
element-wise, which would not produce a valid rotation matrix).

**Targeting.** The goal is defined in the tag's own frame, which is why moving the box works. The
function below also aims the *camera* rather than the end-effector origin, because the end-effector's
Z range is `[-0.589, 0.300]` while the tag sits at `z = 0.8`, so targeting the origin at tag height
is unreachable by construction. Since every joint rotates about Z, the end-effector pose is
`[Rz(θ) | p]` and the conversion back through the camera mount is exact:

```python
def goal_pose_from_tag(T_world_tag, standoff, off_x, off_y, cam_name="middle_camera"):
    R, p = T_world_tag[:3, :3], T_world_tag[:3, 3]
    n = R[:, 2]

    # stand off on whichever side of the tag faces the robot
    base = np.array([_BASE_XY[0], _BASE_XY[1], p[2]])
    s = standoff
    if np.linalg.norm(p + s * n - base) > np.linalg.norm(p - s * n - base):
        s = -s
    cam_target = p + s * n + off_x * R[:, 0] + off_y * R[:, 1]

    # yaw the wrist so the camera's optical axis points back at the tag
    T_ec = _T_ee_cam[cam_name]
    alpha = math.atan2(T_ec[1, 2], T_ec[0, 2])
    d = p - cam_target
    theta = math.atan2(d[1], d[0]) - alpha

    # camera_world = p + Rz(theta) @ offset  ->  solve for p
    off_ee = T_ec[:3, 3]
    c, sn = math.cos(theta), math.sin(theta)
    rotated = np.array([c * off_ee[0] - sn * off_ee[1],
                        sn * off_ee[0] + c * off_ee[1],
                        off_ee[2]])
    return (*(cam_target - rotated), theta)
```

**Kinematics.** Closed-form, ported from the formulas that run on the real PF400, so the sim and the
hardware share one model. A differential IK controller was tried first and abandoned after it kept
driving into joint limits and jittering at singularities.

Every target has two solutions (elbow up and elbow down) and frequently only one is inside the joint
limits. The original formula picked by a geometric rule that often chose the out-of-limits one, which
the actuator then clamped silently, so the arm reached a plausible-looking but wrong pose. Solving
both and scoring them fixed it:

```python
candidates = []
for elbow in ("up", "down"):
    q = solve(elbow)
    violation = sum(max(0.0, lo - v) + max(0.0, v - hi)
                    for v, (lo, hi) in zip(q, JOINT_LIMITS))
    travel = float(np.abs(q - q_current).sum())
    candidates.append((violation, travel, elbow, q))

# prefer any in-limits solution, then the smaller move
violation, _, elbow, q = min(candidates, key=lambda c: (c[0] > 1e-9, c[1]))
```

The arm alone spans 0.808 m from the shoulder pivot, so distant targets are only reachable by sliding
the base along its rail. `pf400_ik_auto` sweeps the rail's travel, keeps the fully in-limits
solutions, and takes the one nearest the rail's current position.

## Verified results

| | |
|---|---|
| Forward kinematics vs. simulator | **0.00005 m** |
| IK round-trip (FK → IK → FK) | zero position error, 7/7 configurations in-limits |
| Camera intrinsics vs. OAK-D-Lite datasheet | `fx` 457.92 (RGB) / 433.33 (mono), both exact |
| Stereo baseline, measured | 74.9 mm (spec: 75 mm) |
| Three-camera tag agreement | 0.04 m |

Link geometry was read straight out of the USD joint prims rather than inferred from physics
readings, and cross-checks against the CAD to within a millimetre: L1 302.5 mm (CAD 302), L2 283.4 mm
(CAD 282).

## Running it

Requires Isaac Sim / Isaac Lab, plus `pyapriltags`, `scipy` and `pillow` in Isaac's Python.

```bash
python testenv.py --enable_cameras
```

`--enable_cameras` is mandatory; without it the render annotators never attach and the cameras
produce empty buffers.

| Command | Does |
|---|---|
| `demo [standoff] [shift]` | the full re-targeting loop above |
| `goal <standoff> [ox] [oy]` | hover in front of the tag (standoff ≥ 0.5 m) |
| `move x y z [yaw]` | analytical IK to a base-frame position |
| `detect [tag_size]` | detect on all cameras, print fused pose, save overlays |
| `snap [cam]` | save a frame with intrinsics and world pose |
| `kincheck` / `jointcheck` / `camcheck` | FK vs. physics, USD joint geometry, camera mounts |
| `shiftbox [dx dy]` | nudge the box |
| `home`, `stop`, `q` | reset, cancel, quit |

Input runs on a background thread so the simulator keeps rendering while you type.

## Layout

```
testenv.py          scene, kinematics, AprilTag pipeline, demo state machine
models/pf400/       robot USD
cam/                camera output and detection overlays
```
