"""
Hackathon Calibration Script
============================
Collects reference points by physically moving the robot and computes the
camera→end-effector transform (T_cam_to_ee) needed by motor_grasp_pipeline.py.

Math
----
At each repetition i the robot is at the overhead position with known pose
T_ee_to_base_i (forward kinematics). The depth camera detects the motor at
point p_cam_i in the camera frame. The user then jogs the robot TCP to
exactly the motor centre and saves that pose p_grasp_i in the robot base frame.

We need R_ce, t_ce (camera→EE rotation and translation) such that:
    T_ee_to_base_i @ (R_ce @ p_cam_i + t_ce) ≈ p_grasp_i

Rearranging: R_ce @ p_cam_i + t_ce ≈ T_base_to_ee_i @ p_grasp_i = q_i

So we collect pairs (p_cam_i, q_i) and run Kabsch/SVD to get R_ce, t_ce.
Three non-colinear pairs fully determine the rigid transform.
More pairs reduce noise — aim for 4–6 if time allows.

Procedure per repetition
-------------------------
Step 1  Move robot to the fixed overhead position.
        Script captures depth, detects motor, records p_cam.
Step 2  Move robot directly above the motor at RGB-scan height.
        Script captures RGB, detects centroid (cross-check / used for
        mounting_offset_deg estimation).
Step 3  Jog the robot TCP to the exact motor centre at table height.
        Script reads and saves the TCP pose (= ground truth p_grasp).

After N >= 3 repetitions the script solves and saves T_cam_to_ee.npy.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from motor_grasp_pipeline import (
    CameraInterface,
    RobotInterface,
    detect_motor_orientation,
    ee_pose_to_matrix,
    find_motor_in_depth,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CALIBRATION_DIR = Path("calibration")
CALIBRATION_DIR.mkdir(exist_ok=True)

RAW_DATA_FILE = CALIBRATION_DIR / "calibration_points.json"
T_CAM_TO_EE_FILE = CALIBRATION_DIR / "T_cam_to_ee.npy"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rigid_transform_svd(
    src: np.ndarray,   # (N, 3)  points in source frame (camera)
    dst: np.ndarray,   # (N, 3)  points in target frame (EE)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm: find R (3×3), t (3,) such that R @ src[i] + t ≈ dst[i].
    Returns (R, t).
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    A = src - centroid_src
    B = dst - centroid_dst

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    t = centroid_dst - R @ centroid_src

    return R, t


def reprojection_errors(
    src: np.ndarray,
    dst: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Return per-point Euclidean errors in metres."""
    predicted = (R @ src.T).T + t
    return np.linalg.norm(predicted - dst, axis=1)


def build_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_raw_data() -> List[dict]:
    if RAW_DATA_FILE.exists():
        with open(RAW_DATA_FILE) as f:
            data = json.load(f)
        log.info("Loaded %d existing calibration point(s) from %s", len(data), RAW_DATA_FILE)
        return data
    return []


def save_raw_data(data: List[dict]):
    with open(RAW_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved %d calibration point(s) to %s", len(data), RAW_DATA_FILE)


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt(msg: str):
    """Print a clearly visible instruction and wait for Enter."""
    print("\n" + "=" * 60)
    print(f"  ACTION REQUIRED: {msg}")
    print("=" * 60)
    input("  Press ENTER when done…")


def robot_read_tcp(robot: RobotInterface) -> dict:
    """
    Read the current TCP pose from the robot.
    Returns {x, y, z, rx, ry, rz} in metres / degrees.
    STUB — replace with your robot SDK call.
    """
    pose = robot.get_current_pose()
    log.info("TCP pose: %s", pose)
    return pose


# ---------------------------------------------------------------------------
# One calibration repetition
# ---------------------------------------------------------------------------

def collect_one_repetition(
    camera: CameraInterface,
    robot: RobotInterface,
    rep_index: int,
    save_debug_images: bool = True,
) -> dict:
    """
    Guide the user through one repetition and return a data dict with:
      - overhead_ee_pose: robot TCP pose at the overhead position
      - p_cam_depth: [Xc, Yc, Zc] motor in depth-camera frame (metres)
      - rgb_ee_pose: robot TCP pose at the RGB-scan position
      - image_angle_deg: detected orientation in the RGB image
      - centroid_px: (u, v) centroid in RGB image
      - grasp_ee_pose: robot TCP pose when TCP was at the motor centre
    """
    print(f"\n{'#'*60}")
    print(f"  REPETITION {rep_index + 1}")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # STEP 1 — Overhead depth scan
    # ------------------------------------------------------------------
    prompt(
        f"[Rep {rep_index+1} / Step 1] Move the robot to the FIXED OVERHEAD position "
        "and hold it steady."
    )

    overhead_ee_pose = robot_read_tcp(robot)
    T_overhead_ee_to_base = ee_pose_to_matrix(overhead_ee_pose)
    T_overhead_base_to_ee = np.linalg.inv(T_overhead_ee_to_base)

    log.info("Capturing depth frame…")
    depth_mm, ir_left = camera.capture_depth_frame(n_average=5)

    detection = find_motor_in_depth(depth_mm)
    u, v = detection["center_uv"]
    depth_m = detection["depth_mm"] / 1000.0

    p_cam_depth = camera.pixel_to_camera_3d(u, v, depth_m, use_color_intrinsics=False)
    log.info("Motor in camera frame (depth): [%.4f, %.4f, %.4f] m", *p_cam_depth)

    if save_debug_images:
        _save_debug_depth(ir_left, depth_mm, detection, rep_index, u, v)

    # ------------------------------------------------------------------
    # STEP 2 — RGB scan (close-up, for orientation cross-check)
    # ------------------------------------------------------------------
    prompt(
        f"[Rep {rep_index+1} / Step 2] Move the robot to the CLOSE-UP RGB position "
        "(directly above the motor at your chosen RGB scan height)."
    )

    rgb_ee_pose = robot_read_tcp(robot)

    log.info("Capturing RGB frame…")
    color_rgb, depth_aligned = camera.capture_rgb_frame(n_average=3)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    image_angle_deg, (cx_px, cy_px) = detect_motor_orientation(color_bgr)
    log.info("Orientation in RGB image: %.1f°, centroid px=(%.1f, %.1f)",
             image_angle_deg, cx_px, cy_px)

    if save_debug_images:
        _save_debug_rgb(color_bgr, cx_px, cy_px, image_angle_deg, rep_index)

    # ------------------------------------------------------------------
    # STEP 3 — Manual grasp point (ground truth)
    # ------------------------------------------------------------------
    prompt(
        f"[Rep {rep_index+1} / Step 3] JOG the robot TCP to the EXACT CENTRE of the motor "
        "at table height (the point where the gripper would close on the motor). "
        "Make sure the TCP is at the motor centre, not the surface — "
        "typically the midpoint of the motor thickness."
    )

    grasp_ee_pose = robot_read_tcp(robot)
    p_grasp_robot = np.array([grasp_ee_pose["x"],
                               grasp_ee_pose["y"],
                               grasp_ee_pose["z"]])
    log.info("Grasp point in robot frame: [%.4f, %.4f, %.4f] m", *p_grasp_robot)

    # q_i = T_base_to_ee_overhead @ p_grasp_robot  (target in EE frame)
    p_grasp_h  = np.array([*p_grasp_robot, 1.0])
    q_in_ee    = (T_overhead_base_to_ee @ p_grasp_h)[:3]

    return {
        "rep": rep_index + 1,
        # raw data (serialisable)
        "overhead_ee_pose": overhead_ee_pose,
        "p_cam_depth": p_cam_depth.tolist(),
        "depth_mm": float(detection["depth_mm"]),
        "depth_center_uv": [float(u), float(v)],
        "rgb_ee_pose": rgb_ee_pose,
        "image_angle_deg": float(image_angle_deg),
        "centroid_px": [float(cx_px), float(cy_px)],
        "grasp_ee_pose": grasp_ee_pose,
        # derived (also serialised for transparency)
        "p_grasp_robot": p_grasp_robot.tolist(),
        "q_in_ee": q_in_ee.tolist(),
    }


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_hand_eye(data: List[dict]) -> np.ndarray:
    """
    Given N >= 3 calibration repetitions, solve for T_cam_to_ee (4×4).

    At each rep i:
      p_cam_i  = motor in depth-camera frame (from depth backprojection)
      q_i      = motor in EE frame at overhead pose = T_base_to_ee_i @ p_grasp_i

    Find R, t such that  R @ p_cam_i + t ≈ q_i  for all i.
    """
    if len(data) < 3:
        raise ValueError(f"Need at least 3 repetitions, got {len(data)}.")

    p_cam = np.array([d["p_cam_depth"] for d in data], dtype=np.float64)  # (N, 3)
    q_ee  = np.array([d["q_in_ee"]     for d in data], dtype=np.float64)  # (N, 3)

    R, t = rigid_transform_svd(p_cam, q_ee)

    errors = reprojection_errors(p_cam, q_ee, R, t)
    log.info("Hand-eye solve: %d points, RMS error = %.4f m, max = %.4f m",
             len(data), float(np.sqrt(np.mean(errors**2))), float(errors.max()))
    for i, (e, d) in enumerate(zip(errors, data)):
        log.info("  Rep %d: error = %.4f m", d["rep"], e)

    T = build_transform(R, t)
    return T


def estimate_mounting_offset(data: List[dict]) -> float:
    """
    Estimate mounting_offset_deg from the saved repetitions.
    At each rep we know:
      - image_angle_deg: motor orientation in the RGB image
      - The motor's true orientation in robot frame can be approximated from
        the grasp TCP's rz (the operator aligned the gripper to the motor).
    Returns the mean offset in degrees.
    If the operator did NOT align the gripper during the grasp step, skip this.
    """
    offsets = []
    for d in data:
        grasp_rz    = d["grasp_ee_pose"].get("rz", 0.0)  # robot tool yaw at grasp
        rgb_rz      = d["rgb_ee_pose"].get("rz", 0.0)    # robot tool yaw at RGB scan
        image_angle = d["image_angle_deg"]
        # Angle of motor in robot frame (approximate) = grasp tool yaw
        # Angle of motor in image = image_angle_deg
        # offset = robot_angle - image_angle - current_tool_yaw_at_RGB_scan
        offset = (grasp_rz - image_angle - rgb_rz) % 360
        if offset > 180:
            offset -= 360
        offsets.append(offset)
        log.info("  Rep %d: image_angle=%.1f° grasp_rz=%.1f° → offset=%.1f°",
                 d["rep"], image_angle, grasp_rz, offset)
    mean_offset = float(np.mean(offsets))
    log.info("Estimated mounting_offset_deg: %.1f°", mean_offset)
    return mean_offset


# ---------------------------------------------------------------------------
# Debug image helpers
# ---------------------------------------------------------------------------

def _save_debug_depth(ir_left, depth_mm, detection, rep_index, u, v):
    os.makedirs("calibration/debug", exist_ok=True)
    vis = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [detection["box"]], -1, (0, 255, 0), 3)
    x, y, w, h = detection["roi"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
    cv2.circle(vis, (int(u), int(v)), 10, (0, 0, 255), -1)
    cv2.putText(vis, f"Rep {rep_index+1} depth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    path = f"calibration/debug/rep{rep_index+1}_phase1_depth.png"
    cv2.imwrite(path, vis)
    log.info("Saved %s", path)


def _save_debug_rgb(color_bgr, cx_px, cy_px, angle_deg, rep_index):
    os.makedirs("calibration/debug", exist_ok=True)
    vis = color_bgr.copy()
    cx, cy = int(round(cx_px)), int(round(cy_px))
    length = 80
    dx = int(length * np.cos(np.radians(angle_deg)))
    dy = int(length * np.sin(np.radians(angle_deg)))
    cv2.arrowedLine(vis, (cx - dx, cy - dy), (cx + dx, cy + dy),
                    (0, 220, 255), 3, tipLength=0.15)
    cv2.circle(vis, (cx, cy), 8, (255, 0, 0), -1)
    cv2.putText(vis, f"Rep {rep_index+1}  {angle_deg:.1f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    path = f"calibration/debug/rep{rep_index+1}_phase2_rgb.png"
    cv2.imwrite(path, vis)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          Motor Grasp Pipeline — Calibration              ║
╠══════════════════════════════════════════════════════════╣
║  You will be guided through 3–6 repetitions.             ║
║  Each repetition:                                        ║
║    Step 1 — Robot to overhead position (depth scan)      ║
║    Step 2 — Robot to close-up position (RGB scan)        ║
║    Step 3 — Jog TCP to exact motor centre (ground truth) ║
╚══════════════════════════════════════════════════════════╝
""")

    n_reps = int(input("How many repetitions? (min 3, recommended 5): ").strip() or "3")
    if n_reps < 3:
        print("Need at least 3. Setting to 3.")
        n_reps = 3

    # Load any previously saved points (allows resuming after a crash)
    data = load_raw_data()
    already_done = len(data)
    if already_done:
        cont = input(f"Found {already_done} existing point(s). Continue from there? [Y/n]: ").strip().lower()
        if cont == "n":
            data = []
            already_done = 0

    camera = CameraInterface()
    robot  = RobotInterface()

    try:
        camera.start()

        for i in range(already_done, n_reps):
            rep_data = collect_one_repetition(camera, robot, rep_index=i)
            data.append(rep_data)
            save_raw_data(data)  # save after each rep so nothing is lost on crash

    finally:
        camera.stop()

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SOLVING for T_cam_to_ee…")
    print("=" * 60)

    T_cam_to_ee = solve_hand_eye(data)

    print("\nT_cam_to_ee (camera → end-effector):")
    print(np.array2string(T_cam_to_ee, precision=6, suppress_small=True))

    np.save(str(T_CAM_TO_EE_FILE), T_cam_to_ee)
    print(f"\nSaved to {T_CAM_TO_EE_FILE}")

    # ------------------------------------------------------------------
    # Mounting offset for angle mapping
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ESTIMATING mounting_offset_deg…")
    print("=" * 60)
    print("NOTE: This is only valid if during Step 3 (jog to grasp point)")
    print("you also ROTATED the gripper to align with the motor's long axis.")
    print()
    do_offset = input("Did you align the gripper rotation during Step 3? [y/N]: ").strip().lower()
    if do_offset == "y":
        offset = estimate_mounting_offset(data)
        print(f"\nmounting_offset_deg = {offset:.1f}°")
        print("Update this value in motor_grasp_pipeline.py → _image_angle_to_robot_angle()")
    else:
        print("Skipping mounting offset estimation. Use the manual method in CALIBRATION.md.")

    # ------------------------------------------------------------------
    # Quick validation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    p_cam = np.array([d["p_cam_depth"] for d in data])
    q_ee  = np.array([d["q_in_ee"]     for d in data])
    R = T_cam_to_ee[:3, :3]
    t = T_cam_to_ee[:3, 3]
    from motor_grasp_pipeline import camera_point_to_robot, ee_pose_to_matrix
    for d in data:
        T_ee_to_base = ee_pose_to_matrix(d["overhead_ee_pose"])
        p_predicted = camera_point_to_robot(
            np.array(d["p_cam_depth"]), T_cam_to_ee, T_ee_to_base
        )
        p_actual = np.array(d["p_grasp_robot"])
        err_m = np.linalg.norm(p_predicted - p_actual)
        print(f"  Rep {d['rep']}: predicted=[{p_predicted[0]:.4f}, {p_predicted[1]:.4f}, "
              f"{p_predicted[2]:.4f}] m  |  actual=[{p_actual[0]:.4f}, {p_actual[1]:.4f}, "
              f"{p_actual[2]:.4f}] m  |  error={err_m*1000:.1f} mm")

    print("\nCalibration complete!")
    print(f"  T_cam_to_ee saved → {T_CAM_TO_EE_FILE}")
    print(f"  Raw data saved    → {RAW_DATA_FILE}")
    print("\nNext: run motor_grasp_pipeline.py to test the full pipeline.")


if __name__ == "__main__":
    main()
