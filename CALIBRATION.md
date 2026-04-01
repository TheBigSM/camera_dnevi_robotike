# Calibration Guide — Motor Grasp Pipeline

## Overview

To go from pixels in a camera image to a grasp command in robot coordinates you need a
chain of calibrations. Miss one and everything downstream is wrong.

```
Motor pixel (u,v)
      │  camera intrinsics (K)           ← factory-done, verify once
      ▼
3-D point in camera frame  [Xc, Yc, Zc]
      │  hand-eye transform  T_cam→EE   ← computed by calibrate_pipeline.py
      ▼
3-D point in end-effector frame  [Xee, Yee, Zee]
      │  robot forward kinematics  T_EE→Base  (given by the robot controller)
      ▼
3-D point in robot base frame  [Xr, Yr, Zr]  ← grasp target
```

---

## 1. Hand-Eye Calibration via Reference Points  *(run this first)*

### Why this approach works for a hackathon
Full checkerboard hand-eye calibration requires a calibration target and 15+ poses.
Instead we collect **3–6 reference points** by physically moving the robot:

For each repetition i:
- Robot is at the overhead position with known TCP pose → depth camera detects
  the motor at `p_cam_i` (3D point in camera frame via depth backprojection).
- User jogs the TCP to the exact motor centre → saves `p_grasp_i` in robot base frame.

We then solve for **R, t** (rotation and translation of camera→EE) using SVD:

```
R @ p_cam_i  +  t  ≈  T_base_to_EE_i  @  p_grasp_i   for all i
```

Three non-colinear point pairs fully determine the 6-DOF rigid transform.
More pairs reduce noise — aim for 5 if time allows.

### Run the calibration script

```bash
python calibrate_pipeline.py
```

It will interactively guide you through each repetition:

```
REPETITION 1
────────────
  Step 1  →  Move robot to the FIXED OVERHEAD position, press Enter
             (script captures depth, detects motor automatically)
  Step 2  →  Move robot to the CLOSE-UP RGB position, press Enter
             (script captures RGB, detects orientation for cross-check)
  Step 3  →  JOG TCP to the exact motor centre at table height, press Enter
             (this is the ground truth used for the transform solve)

Repeat N times with the motor at DIFFERENT positions on the table.
```

After all repetitions the script:
1. Runs SVD to compute `T_cam_to_ee`.
2. Prints per-point reprojection errors (target: < 5 mm).
3. Saves `calibration/T_cam_to_ee.npy` — automatically loaded by the pipeline.
4. Optionally estimates `mounting_offset_deg` if you aligned the gripper during Step 3.

### Tips for good results
- **Spread the motor positions** across the workspace — don't just move it a few cm.
  The points must be non-colinear for the SVD to be well-conditioned.
- Use at least **4 positions**, more is better.
- During Step 3 be precise: the TCP should be at the motor's centre, not its edge.
- If one repetition has a large error (> 10 mm), discard it and redo it.
  The raw data is saved in `calibration/calibration_points.json` after every rep —
  the script can resume if it crashes.

### Inspecting / redoing a bad rep
Edit `calibration/calibration_points.json` and remove the bad entry, then re-run.
The script will ask if you want to continue from the existing data.

---

## 2. Teach the Area-Scan (Overhead) Position

The robot needs a **fixed home position** from which it takes every depth scan.
This must be the same position used during calibration (Step 1 above).

1. Jog the robot to the overhead position (camera pointing straight down,
   arm clear of the work area).
2. Read the TCP pose from the teach pendant.
3. Set `RobotInterface.AREA_SCAN_POSE` in `motor_grasp_pipeline.py`:

```python
AREA_SCAN_POSE = {
    "x":  0.0,   # metres — fill in
    "y":  0.0,   # metres — fill in
    "z":  0.8,   # metres — fill in
    "rx": 0.0,   # degrees
    "ry": 0.0,
    "rz": 0.0,
}
```

**Height guidance**: 0.6–0.9 m above the table is usually enough to see the
whole work area while still resolving the motor shape in the depth image.

---

## 3. Teach the RGB Scan Height

`RobotInterface.RGB_SCAN_HEIGHT` is the Z height (metres above table) at which
Phase 2 (close-up RGB) is captured.

1. Manually position the robot directly above a motor at candidate heights.
2. Capture RGB images and check that the motor fills **30–60% of the frame**.
3. Set `RGB_SCAN_HEIGHT` in `motor_grasp_pipeline.py`.

---

## 4. Image Angle → Robot Angle Offset

`mounting_offset_deg` in `motor_grasp_pipeline.py → _image_angle_to_robot_angle()`.

**Fastest method (from calibration data)**:
During Step 3 of calibration, if you also rotated the gripper to align with
the motor's long axis, the script automatically estimates this offset.
Accept the printed value and paste it into the pipeline.

**Manual method (if the above was not done)**:
1. Place the motor with its long axis pointing along the **robot X-axis**
   (verify with a ruler).
2. Run Phase 2 of the pipeline and read `angle_deg` from the log.
3. `mounting_offset_deg = 0 - detected_image_angle` (since robot angle = 0°).

---

## 5. Depth Scale Verification  *(quick sanity check, ~5 minutes)*

The RealSense reports depth as `uint16` units. The SDK depth scale converts
to metres. Verify it is accurate:

1. Place a flat object at a measured distance (tape measure, ± 2 mm).
2. Read the median depth from a captured frame at that region.
3. Compare: `depth_m = raw_value × camera.depth_scale`.
4. Acceptable error: < 5 mm at operating distance.

If it is off, apply a correction factor:
```python
correction = measured_distance_m / realsense_reading_m
# Multiply all depth readings by correction before use
```

---

## 6. RGB–Depth Alignment Verification  *(optional but recommended)*

Phase 2 reads the **depth value at the RGB centroid pixel**. If alignment is off,
the back-projected 3D point is wrong.

1. Hold a flat, textured object at ~0.3 m.
2. Capture an aligned frame (color + depth aligned to color).
3. Click a visible edge in color — check depth jumps at the same edge.
4. Acceptable: < 3 px misalignment at 0.3 m.

If alignment is off: update RealSense firmware or run
`Intel RealSense Viewer → Calibration`.

---

## Calibration File Summary

| File | Contents | How produced |
|---|---|---|
| `calibration/T_cam_to_ee.npy` | 4×4 camera→EE transform | `calibrate_pipeline.py` |
| `calibration/calibration_points.json` | Raw reference point data | `calibrate_pipeline.py` |
| `calibration/debug/rep*_phase1_depth.png` | Depth detection debug images | `calibrate_pipeline.py` |
| `calibration/debug/rep*_phase2_rgb.png` | RGB orientation debug images | `calibrate_pipeline.py` |

---

## Quick-Start Checklist

- [ ] Run `python calibrate_pipeline.py` with ≥ 3 motor positions (4–5 recommended)
- [ ] Check reprojection errors < 5 mm, redo any outlier reps
- [ ] Set `RobotInterface.AREA_SCAN_POSE` in `motor_grasp_pipeline.py`
- [ ] Set `RobotInterface.RGB_SCAN_HEIGHT` in `motor_grasp_pipeline.py`
- [ ] Set `mounting_offset_deg` in `motor_grasp_pipeline.py`
- [ ] Verify depth scale (section 5)
- [ ] Run `python motor_grasp_pipeline.py` end-to-end and verify grasp accuracy
