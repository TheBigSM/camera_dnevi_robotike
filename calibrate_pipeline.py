"""
Motor Grasp Pipeline — Manual Calibration
==========================================
Calibrates the two image-to-robot mappings needed by motor_grasp_pipeline.py.
No robot SDK required: all robot coordinates are typed in manually.

What gets calibrated
--------------------
Two 2-D affine transforms and one angle offset:

  T_depth  : depth image pixel (u, v)  →  robot XY for close-up positioning
  T_rgb    : RGB image pixel  (cx, cy) →  robot XY for grasp positioning
  angle_offset_deg : image_angle + offset ≈ robot last-joint angle at grasp

Z heights (close-up and grasp) are fixed and averaged across repetitions.

All results are saved to calibration/transforms.json and can be loaded by
motor_grasp_pipeline.py once robot communication is available.

Procedure
---------
Setup (once):
  Move robot to the fixed top position and type in its coordinates.

Per repetition (minimum 3, motor placed at different spots each time):

  Step 1 — Depth scan
    Robot stays at top position.
    Script captures depth + IR, detects motor, shows annotated IR image.

  Step 2 — Visual alignment
    A split window opens:
      LEFT  — static IR reference with the detected motor bounding box
      RIGHT — live RGB feed from the camera
    Move the robot down/around until the motor in the live view matches
    the reference box as closely as possible, then press SPACE to confirm.
    Type in the robot coordinates for that close-up position.

  Step 3 — RGB capture and grasp
    Script captures RGB, detects centroid and orientation, shows the result.
    Jog the robot to the exact pick position (aligned gripper to motor axis).
    Type in the robot coordinates for that pick position.

After >= 3 repetitions the script fits the transforms and saves them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from motor_grasp_pipeline import (
    CameraInterface,
    detect_motor_orientation,
    find_motor_in_depth,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CALIB_DIR      = Path("calibration")
RAW_DATA_FILE  = CALIB_DIR / "calibration_points.json"
TRANSFORMS_FILE = CALIB_DIR / "transforms.json"
DEBUG_DIR      = CALIB_DIR / "debug"


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def prompt_enter(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)
    input("  Press ENTER when ready…")


def input_pose(label: str) -> dict:
    """Prompt the user to type robot XYZ + last joint angle."""
    print(f"\n  Enter robot coordinates for: {label}")
    x  = float(input("    x   (m)   : ").strip())
    y  = float(input("    y   (m)   : ").strip())
    z  = float(input("    z   (m)   : ").strip())
    rz = float(input("    rz  (deg) : ").strip())
    pose = {"x": x, "y": y, "z": z, "rz": rz}
    print(f"  Saved: {pose}")
    return pose


# ---------------------------------------------------------------------------
# Live alignment window
# ---------------------------------------------------------------------------

def live_alignment_view(
    camera: CameraInterface,
    ir_reference: np.ndarray,
    box: np.ndarray,
) -> bool:
    """
    Show a split window:
      LEFT  — static IR image with the detected motor bounding box
      RIGHT — live RGB feed from the camera

    The user moves the robot until the live view matches the reference,
    then presses SPACE to confirm or Q / ESC to abort.

    Returns True on confirm, False on abort.
    """
    H, W = ir_reference.shape[:2]

    # Build static left panel
    left = cv2.cvtColor(ir_reference, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(left, [box], -1, (0, 255, 0), 2)
    cx_box = int(box[:, 0].mean())
    cy_box = int(box[:, 1].mean())
    cv2.circle(left, (cx_box, cy_box), 6, (0, 0, 255), -1)
    cv2.putText(left, "IR reference (target)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.namedWindow("Alignment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alignment", W * 2, H)

    print("\n  [ALIGNMENT WINDOW OPEN]")
    print("  Move the robot until the motor in the live feed matches the reference box.")
    print("  SPACE = confirm position    Q / ESC = abort\n")

    while True:
        frames = camera._pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        rgb = np.asanyarray(color_frame.get_data())
        right = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(right, "Live RGB  —  align motor to box", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(right, "SPACE = confirm   Q = abort", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Alignment", np.hstack([left, right]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.destroyWindow("Alignment")
            return True
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow("Alignment")
            return False


# ---------------------------------------------------------------------------
# Annotation display
# ---------------------------------------------------------------------------

def show_rgb_detection(
    color_bgr: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    rep: int,
):
    """Show the RGB image annotated with centroid and orientation arrow."""
    vis = color_bgr.copy()
    cxi, cyi = int(round(cx)), int(round(cy))
    length = 80
    dx = int(length * np.cos(np.radians(angle)))
    dy = int(length * np.sin(np.radians(angle)))
    cv2.arrowedLine(vis, (cxi - dx, cyi - dy), (cxi + dx, cyi + dy),
                    (0, 220, 255), 3, tipLength=0.15)
    cv2.circle(vis, (cxi, cyi), 8, (255, 0, 0), -1)
    cv2.putText(vis, f"Rep {rep}   angle={angle:.1f} deg   centroid=({cx:.0f}, {cy:.0f})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(vis, "Press any key to continue", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2)
    cv2.imshow("RGB Detection", vis)
    cv2.waitKey(0)
    cv2.destroyWindow("RGB Detection")


# ---------------------------------------------------------------------------
# Transform fitting
# ---------------------------------------------------------------------------

def fit_affine_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Fit a 2-D affine transform M (2×3) from src (N,2) to dst (N,2).

    dst ≈ M @ [src_x, src_y, 1].T

    For N=3: exact solution. For N>3: least squares.
    """
    N = len(src)
    A = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)
    for i, (s, d) in enumerate(zip(src, dst)):
        A[2 * i,     :3] = [s[0], s[1], 1.0]
        A[2 * i + 1, 3:] = [s[0], s[1], 1.0]
        b[2 * i]     = d[0]
        b[2 * i + 1] = d[1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]]])


def apply_affine_2d(M: np.ndarray, pt) -> np.ndarray:
    """Apply a 2×3 affine matrix to a 2-D point. Returns [x, y]."""
    return M @ np.array([pt[0], pt[1], 1.0])


def circular_mean_deg(angles: list[float]) -> float:
    """Mean of a list of angles (degrees), handling wrap-around."""
    sin_m = np.mean([np.sin(np.radians(a)) for a in angles])
    cos_m = np.mean([np.cos(np.radians(a)) for a in angles])
    return float(np.degrees(np.arctan2(sin_m, cos_m)) % 360)


def solve_transforms(reps: list[dict]) -> dict:
    """
    Fit all transforms from the collected repetition data.

    Returns a dict ready to be saved as transforms.json.
    """
    depth_src  = np.array([[r["depth_centroid_px"][0],
                             r["depth_centroid_px"][1]] for r in reps])
    closeup_dst = np.array([[r["closeup_pose"]["x"],
                              r["closeup_pose"]["y"]] for r in reps])

    rgb_src   = np.array([[r["rgb_centroid_px"][0],
                            r["rgb_centroid_px"][1]] for r in reps])
    pick_dst  = np.array([[r["pick_pose"]["x"],
                            r["pick_pose"]["y"]] for r in reps])

    T_depth = fit_affine_2d(depth_src, closeup_dst)
    T_rgb   = fit_affine_2d(rgb_src,   pick_dst)

    # Angle offset: pick_rz - image_angle (circular, averaged)
    angle_offsets = [(r["pick_pose"]["rz"] - r["image_angle_deg"]) % 360
                     for r in reps]
    angle_offset = circular_mean_deg(angle_offsets)

    closeup_z = float(np.mean([r["closeup_pose"]["z"] for r in reps]))
    grasp_z   = float(np.mean([r["pick_pose"]["z"]    for r in reps]))

    return {
        "T_depth_to_closeup_xy": T_depth.tolist(),   # 2×3 matrix
        "T_rgb_to_grasp_xy":     T_rgb.tolist(),      # 2×3 matrix
        "angle_offset_deg":      angle_offset,
        "closeup_z":             closeup_z,
        "grasp_z":               grasp_z,
    }


def print_validation(reps: list[dict], transforms: dict):
    """Print per-rep reprojection errors for a quick sanity check."""
    T_d = np.array(transforms["T_depth_to_closeup_xy"])
    T_r = np.array(transforms["T_rgb_to_grasp_xy"])
    ang_off = transforms["angle_offset_deg"]

    print()
    for r in reps:
        pred_cu  = apply_affine_2d(T_d, r["depth_centroid_px"])
        act_cu   = np.array([r["closeup_pose"]["x"], r["closeup_pose"]["y"]])
        err_cu   = np.linalg.norm(pred_cu - act_cu) * 1000

        pred_gr  = apply_affine_2d(T_r, r["rgb_centroid_px"])
        act_gr   = np.array([r["pick_pose"]["x"], r["pick_pose"]["y"]])
        err_gr   = np.linalg.norm(pred_gr - act_gr) * 1000

        pred_ang = (r["image_angle_deg"] + ang_off) % 360
        err_ang  = abs((pred_ang - r["pick_pose"]["rz"] + 180) % 360 - 180)

        print(f"  Rep {r['rep']}:  close-up err={err_cu:5.1f} mm  "
              f"grasp err={err_gr:5.1f} mm  angle err={err_ang:4.1f}°")


# ---------------------------------------------------------------------------
# One repetition
# ---------------------------------------------------------------------------

def collect_one_rep(camera: CameraInterface, rep_index: int) -> dict:
    print(f"\n{'#' * 60}\n  REPETITION {rep_index + 1}\n{'#' * 60}")

    # ------------------------------------------------------------------
    # Step 1: depth scan from top position
    # ------------------------------------------------------------------
    prompt_enter(
        f"[Rep {rep_index + 1} / Step 1]  Move the motor to a NEW position on the table.\n"
        "  Then move the robot to the FIXED TOP POSITION and hold steady."
    )

    log.info("Capturing depth + IR…")
    depth_mm, ir_left = camera.capture_depth_frame(n_average=5)
    detection = find_motor_in_depth(depth_mm)
    u, v = detection["center_uv"]
    log.info("Motor detected at pixel (%.1f, %.1f), depth=%.0f mm", u, v, detection["depth_mm"])

    # Save depth debug image
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    vis_depth = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_depth, [detection["box"]], -1, (0, 255, 0), 2)
    cv2.circle(vis_depth, (int(u), int(v)), 8, (0, 0, 255), -1)
    depth_debug_path = str(DEBUG_DIR / f"rep{rep_index + 1}_depth.png")
    cv2.imwrite(depth_debug_path, vis_depth)
    log.info("Depth debug image saved: %s", depth_debug_path)

    # ------------------------------------------------------------------
    # Step 2: visual alignment + type close-up coordinates
    # ------------------------------------------------------------------
    prompt_enter(
        f"[Rep {rep_index + 1} / Step 2]  The alignment window will open.\n"
        "  Move the robot down toward the motor.\n"
        "  Adjust position until the motor in the live RGB view matches\n"
        "  the bounding box shown in the IR reference panel.\n"
        "  Press SPACE in the window to confirm."
    )

    confirmed = live_alignment_view(camera, ir_left, detection["box"])
    if not confirmed:
        raise RuntimeError("Alignment aborted.")

    closeup_pose = input_pose(f"close-up position (rep {rep_index + 1})")

    # ------------------------------------------------------------------
    # Step 3: RGB capture, orientation, then jog to pick
    # ------------------------------------------------------------------
    log.info("Capturing RGB frame…")
    color_rgb, _ = camera.capture_rgb_frame(n_average=3)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    image_angle, (cx, cy) = detect_motor_orientation(color_bgr)
    log.info("Orientation: %.1f°   centroid: (%.1f, %.1f)", image_angle, cx, cy)

    show_rgb_detection(color_bgr, cx, cy, image_angle, rep_index + 1)

    # Save RGB debug image
    rgb_debug_path = str(DEBUG_DIR / f"rep{rep_index + 1}_rgb.png")
    annotated = color_bgr.copy()
    cxi, cyi = int(round(cx)), int(round(cy))
    dx = int(80 * np.cos(np.radians(image_angle)))
    dy = int(80 * np.sin(np.radians(image_angle)))
    cv2.arrowedLine(annotated, (cxi - dx, cyi - dy), (cxi + dx, cyi + dy),
                    (0, 220, 255), 3, tipLength=0.15)
    cv2.circle(annotated, (cxi, cyi), 8, (255, 0, 0), -1)
    cv2.imwrite(rgb_debug_path, annotated)
    log.info("RGB debug image saved: %s", rgb_debug_path)

    prompt_enter(
        f"[Rep {rep_index + 1} / Step 3]  Jog the robot to the PICK POSITION.\n"
        "  Position the TCP at the motor centre at grasp height.\n"
        "  Rotate the last joint so the gripper aligns with the motor's long axis.\n"
        f"  (Detected motor angle in image: {image_angle:.1f}°)"
    )

    pick_pose = input_pose(f"pick position (rep {rep_index + 1})")

    return {
        "rep":               rep_index + 1,
        "depth_centroid_px": [float(u), float(v)],
        "depth_mm":          float(detection["depth_mm"]),
        "closeup_pose":      closeup_pose,
        "rgb_centroid_px":   [float(cx), float(cy)],
        "image_angle_deg":   float(image_angle),
        "pick_pose":         pick_pose,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║       Motor Grasp Pipeline — Manual Calibration          ║
╠══════════════════════════════════════════════════════════╣
║  No robot SDK required.                                  ║
║  All robot coordinates are entered manually.             ║
║                                                          ║
║  Per repetition (min 3, motor in different positions):   ║
║    Step 1 — Top position → depth scan                    ║
║    Step 2 — Live alignment view → type close-up coords   ║
║    Step 3 — Capture RGB → jog to pick → type coords      ║
╚══════════════════════════════════════════════════════════╝
""")

    CALIB_DIR.mkdir(exist_ok=True)

    n_reps = int(input("How many repetitions? (min 3, recommended 5): ").strip() or "3")
    if n_reps < 3:
        print("Need at least 3. Setting to 3.")
        n_reps = 3

    # --- Save top position once ---
    prompt_enter(
        "Move the robot to the FIXED TOP POSITION used for all depth scans.\n"
        "  This is the overhead position — always the same for every run."
    )
    top_pose = input_pose("fixed top position")

    # --- Resume support ---
    reps: list[dict] = []
    if RAW_DATA_FILE.exists():
        with open(RAW_DATA_FILE) as f:
            saved = json.load(f)
        if saved.get("reps"):
            ans = input(f"\nFound {len(saved['reps'])} saved rep(s). Resume? [Y/n]: ").strip().lower()
            if ans != "n":
                reps     = saved["reps"]
                top_pose = saved.get("top_pose", top_pose)
                print(f"Resuming from rep {len(reps) + 1}.")

    # --- Collect repetitions ---
    camera = CameraInterface()
    try:
        camera.start()

        for i in range(len(reps), n_reps):
            rep_data = collect_one_rep(camera, i)
            reps.append(rep_data)
            with open(RAW_DATA_FILE, "w") as f:
                json.dump({"top_pose": top_pose, "reps": reps}, f, indent=2)
            log.info("Rep %d saved to %s", i + 1, RAW_DATA_FILE)

    finally:
        camera.stop()

    # --- Solve ---
    print("\n" + "=" * 60)
    print("  COMPUTING TRANSFORMS")
    print("=" * 60)

    transforms = solve_transforms(reps)
    transforms["top_pose"] = top_pose
    transforms["n_reps"]   = len(reps)

    with open(TRANSFORMS_FILE, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n  angle_offset_deg : {transforms['angle_offset_deg']:.1f}°")
    print(f"  closeup_z        : {transforms['closeup_z']:.4f} m")
    print(f"  grasp_z          : {transforms['grasp_z']:.4f} m")

    # --- Validation ---
    print("\n" + "=" * 60)
    print("  VALIDATION — reprojection errors (should be 0 with N=3 reps)")
    print("=" * 60)
    print_validation(reps, transforms)

    print(f"\n  Raw data   → {RAW_DATA_FILE}")
    print(f"  Transforms → {TRANSFORMS_FILE}")
    print("\nCalibration complete. Load transforms.json in motor_grasp_pipeline.py.")


if __name__ == "__main__":
    main()
