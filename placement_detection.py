"""
Placement Position Detection
============================
Detects the red 3D-printed bracket (placement target) in the workspace
and computes its centroid + orientation for the robot.
 
Approach: HSV color segmentation on RGB image.
The red bracket has very high contrast against the aluminium rail table,
making color-based detection much more robust than depth-based for this target.
 
Output: PlacementPose(x, y, z, angle_deg) in robot base frame.
"""
 
from __future__ import annotations
 
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
 
import cv2
import numpy as np
 
log = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
 
@dataclass
class PlacementPose:
    """Robot placement target expressed in the robot base frame."""
    x: float          # metres
    y: float          # metres
    z: float          # metres
    angle_deg: float  # rotation around Z-axis (tool yaw), [0, 360)
 
    def __str__(self) -> str:
        return (f"PlacementPose(x={self.x:.4f} m, y={self.y:.4f} m, "
                f"z={self.z:.4f} m, angle={self.angle_deg:.1f}°)")
 
 
# ---------------------------------------------------------------------------
# Red bracket detection
# ---------------------------------------------------------------------------
 
def detect_red_bracket(
    image_bgr: np.ndarray,
    # HSV thresholds for red — red wraps around 0/180 in HSV
    hue_low1:  int = 0,    hue_high1: int = 10,   # lower red range
    hue_low2:  int = 165,  hue_high2: int = 180,  # upper red range
    sat_low:   int = 120,  sat_high:  int = 255,
    val_low:   int = 80,   val_high:  int = 255,
    min_area:  int = 500,
    morph_kernel: int = 7,
    debug_dir: Optional[str] = None,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """
    Detect the red bracket in a BGR image.
 
    Returns:
        angle_deg : orientation of the bracket long axis in [0, 360)
        (cx, cy)  : centroid in image pixels
        mask      : binary mask of detected bracket (for debug)
 
    Raises ValueError if no red bracket found.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
 
    # Red wraps around 0 in HSV — need two ranges
    lower1 = np.array([hue_low1, sat_low, val_low])
    upper1 = np.array([hue_high1, sat_high, val_high])
    lower2 = np.array([hue_low2, sat_low, val_low])
    upper2 = np.array([hue_high2, sat_high, val_high])
 
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)
 
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
 
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No red bracket detected in image.")
 
    # Pick largest contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
 
    if area < min_area:
        raise ValueError(f"Largest red region too small ({area:.0f} px² < {min_area}). "
                         "Is the bracket in frame?")
 
    # Fit minimum area rectangle for orientation
    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect
 
    # Normalize angle to long axis
    if w < h:
        angle = angle + 90
    angle = angle % 180  # still 180° ambiguous
 
    # Centroid via moments (more accurate than bounding rect center)
    M  = cv2.moments(contour)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
 
    # Resolve 180° ambiguity using centroid offset from rect center
    dx = cx - center[0]
    dy = cy - center[1]
    ax = np.cos(np.radians(angle))
    ay = np.sin(np.radians(angle))
    cross = ax * dy - ay * dx
    if cross < 0:
        angle = (angle + 180) % 360
 
    log.info("Red bracket detected: centroid=(%.1f, %.1f) px, angle=%.1f°, area=%.0f px²",
             cx, cy, angle, area)
 
    # Save debug image if requested
    if debug_dir is not None:
        _save_debug(image_bgr, mask, contour, cx, cy, angle, center, debug_dir)
 
    return float(angle), (float(cx), float(cy)), mask
 
 
def _save_debug(image_bgr, mask, contour, cx, cy, angle, rect_center, debug_dir):
    """Save annotated debug image."""
    os.makedirs(debug_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    vis = image_bgr.copy()
 
    # Draw contour
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
 
    # Draw orientation arrow
    length = 80
    dx = int(length * np.cos(np.radians(angle)))
    dy = int(length * np.sin(np.radians(angle)))
    cxi, cyi = int(round(cx)), int(round(cy))
    cv2.arrowedLine(vis, (cxi - dx, cyi - dy), (cxi + dx, cyi + dy),
                    (0, 220, 255), 3, tipLength=0.2)
 
    # Draw centroid
    cv2.circle(vis, (cxi, cyi), 8, (255, 0, 0), -1)
    cv2.putText(vis, f"{angle:.1f} deg", (cxi + 12, cyi - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
 
    # Save
    vis_path  = os.path.join(debug_dir, f"{ts}_placement_detection.png")
    mask_path = os.path.join(debug_dir, f"{ts}_placement_mask.png")
    cv2.imwrite(vis_path,  vis)
    cv2.imwrite(mask_path, mask)
    log.info("Saved placement debug images: %s", vis_path)
 
 
# ---------------------------------------------------------------------------
# Full placement detection phase
# (mirrors Phase 2 of MotorGraspPipeline but for the red bracket)
# ---------------------------------------------------------------------------
 
def detect_placement_pose(
    camera,           # CameraInterface instance (already started)
    robot,            # RobotInterface instance
    T_cam_to_ee,      # 4×4 hand-eye transform (numpy)
    placement_scan_height: float = 0.35,   # metres above table — REPLACE
    debug_dir: Optional[str] = "data/pipeline",
) -> PlacementPose:
    """
    Move the robot to scan the placement zone, detect the red bracket,
    and return the placement pose in the robot base frame.
 
    Steps:
        1. Move robot to placement scan position (above the bracket zone).
        2. Capture RGB + aligned depth.
        3. Detect red bracket → centroid + orientation in image.
        4. Back-project centroid pixel to 3D using depth.
        5. Transform to robot frame using hand-eye calibration.
        6. Return PlacementPose.
    """
    # Import here to avoid circular dependency if used standalone
    from motor_grasp_pipeline import (
        ee_pose_to_matrix,
        camera_point_to_robot,
    )
 
    # 1. Move robot above placement zone
    # Define the scan position for the placement zone — teach this in once
    PLACEMENT_SCAN_POSE = {
        "x": 0.0,   # metres — REPLACE with actual position above bracket
        "y": 0.3,   # metres — REPLACE
        "z": placement_scan_height,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
    }
    log.info("Moving to placement scan pose…")
    robot.move_above(
        PLACEMENT_SCAN_POSE["x"],
        PLACEMENT_SCAN_POSE["y"],
        z_height=placement_scan_height
    )
    ee_pose      = robot.get_current_pose()
    T_ee_to_base = ee_pose_to_matrix(ee_pose)
 
    # 2. Capture RGB + aligned depth
    log.info("Capturing RGB frame for placement detection…")
    color_rgb, depth_mm_aligned = camera.capture_rgb_frame(n_average=3)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
 
    # 3. Detect red bracket
    angle_deg, (cx_px, cy_px), _ = detect_red_bracket(
        color_bgr,
        debug_dir=debug_dir,
    )
 
    # 4. Back-project centroid to 3D
    cx_i = int(np.clip(round(cx_px), 0, depth_mm_aligned.shape[1] - 1))
    cy_i = int(np.clip(round(cy_px), 0, depth_mm_aligned.shape[0] - 1))
    depth_m = float(depth_mm_aligned[cy_i, cx_i]) / 1000.0
 
    if depth_m < 0.05:
        log.warning("Depth at bracket centroid is near zero — using scan height as fallback.")
        depth_m = placement_scan_height
 
    p_camera = camera.pixel_to_camera_3d(cx_px, cy_px, depth_m, use_color_intrinsics=True)
 
    # 5. Transform to robot frame
    p_robot = camera_point_to_robot(p_camera, T_cam_to_ee, T_ee_to_base)
    log.info("Bracket in robot frame: x=%.4f y=%.4f z=%.4f m",
             p_robot[0], p_robot[1], p_robot[2])
 
    # 6. Convert image angle to robot tool angle
    robot_yaw      = ee_pose.get("rz", 0.0)
    tool_angle_deg = (-angle_deg + robot_yaw) % 360
 
    return PlacementPose(
        x=float(p_robot[0]),
        y=float(p_robot[1]),
        z=float(p_robot[2]),
        angle_deg=tool_angle_deg,
    )
 
 
# ---------------------------------------------------------------------------
# Standalone test — run directly to test detection on a saved image
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import sys
 
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
 
    if len(sys.argv) < 2:
        print("Usage: python placement_detection.py <image_path>")
        print("Example: python placement_detection.py data/bracket_test.jpg")
        sys.exit(1)
 
    img_path = sys.argv[1]
    image    = cv2.imread(img_path)
 
    if image is None:
        print(f"Could not load image: {img_path}")
        sys.exit(1)
 
    try:
        angle, (cx, cy), mask = detect_red_bracket(image, debug_dir="data/pipeline")
        print(f"\nBracket detected:")
        print(f"  Centroid : ({cx:.1f}, {cy:.1f}) px")
        print(f"  Angle    : {angle:.1f}°")
        print(f"\nDebug images saved to data/pipeline/")
 
        # Show result
        cv2.imshow("Detection", cv2.resize(image, (800, 600)))
        cv2.imshow("Mask",      cv2.resize(mask,  (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    except ValueError as e:
        print(f"Detection failed: {e}")
 
