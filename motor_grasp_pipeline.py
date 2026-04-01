"""
Motor Grasp Pipeline
====================
Real-time pipeline for detecting an e-motor on the workspace and computing
the grasp pose for the robot.

Flow
----
1. Robot moves to fixed overhead position.
2. RealSense depth frame → find motor ROI + approximate center in robot coords.
3. Robot moves directly above the detected motor.
4. RealSense RGB frame → refine centroid + orientation in robot coords.
5. Output grasp pose (x, y, z, angle) to the robot.

Coordinate conventions
----------------------
- Image:  (u, v) in pixels, origin top-left.
- Camera: (Xc, Yc, Zc) in metres, Z pointing forward (optical axis).
- Robot:  (Xr, Yr, Zr) in metres, defined by the robot's base frame.

All transforms are 4x4 homogeneous matrices (numpy float64).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GraspPose:
    """Robot grasp target expressed in the robot base frame."""
    x: float          # metres
    y: float          # metres
    z: float          # metres (approach height — to be lowered by robot)
    angle_deg: float  # rotation around Z-axis (tool yaw), [0, 360)

    def __str__(self) -> str:
        return (f"GraspPose(x={self.x:.4f} m, y={self.y:.4f} m, "
                f"z={self.z:.4f} m, angle={self.angle_deg:.1f}°)")


@dataclass
class DepthDetection:
    """Output of the area-scan (depth) phase."""
    center_image: Tuple[float, float]   # (u, v) in depth image pixels
    center_camera: np.ndarray           # [Xc, Yc, Zc] in metres (camera frame)
    center_robot: np.ndarray            # [Xr, Yr, Zr] in metres (robot frame)
    roi: Tuple[int, int, int, int]      # (x, y, w, h) in depth image pixels
    depth_mm: float                     # median depth of detected motor


# ---------------------------------------------------------------------------
# Camera interface — wraps pyrealsense2
# ---------------------------------------------------------------------------

class CameraInterface:
    """
    Thin wrapper around the RealSense D435i pipeline.

    Streams: depth (z16), IR-left (y8), color (rgb8) at 640×480.
    After start(), call capture_depth() or capture_rgb() to get frames.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15):
        self.width = width
        self.height = height
        self.fps = fps

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth,    width, height, rs.format.z16,  fps)
        self._config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        self._config.enable_stream(rs.stream.color,    width, height, rs.format.rgb8,  fps)

        self._align_to_depth = rs.align(rs.stream.depth)
        self._align_to_color = rs.align(rs.stream.color)
        self._profile: Optional[rs.pipeline_profile] = None

        # Intrinsics — populated after start()
        self.depth_intrinsics: Optional[rs.intrinsics] = None
        self.color_intrinsics: Optional[rs.intrinsics] = None
        # Depth-to-color extrinsics (from camera)
        self.depth_to_color_extrinsics: Optional[rs.extrinsics] = None
        # Depth scale (metres per unit)
        self.depth_scale: float = 0.001

    def start(self):
        self._profile = self._pipeline.start(self._config)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        log.info("Depth scale: %.6f m/unit", self.depth_scale)

        depth_stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()

        self.depth_intrinsics = depth_stream.get_intrinsics()
        self.color_intrinsics = color_stream.get_intrinsics()
        self.depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

        log.info("Camera started. Depth intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 self.depth_intrinsics.fx, self.depth_intrinsics.fy,
                 self.depth_intrinsics.ppx, self.depth_intrinsics.ppy)

        # Warm-up: discard a few frames so auto-exposure settles
        for _ in range(30):
            self._pipeline.wait_for_frames(timeout_ms=5000)
        log.info("Camera warmed up.")

    def stop(self):
        self._pipeline.stop()

    def capture_depth_frame(self, n_average: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture and return (depth_raw_mm, ir_left) as numpy arrays.
        Averages n_average frames to reduce noise.
        depth_raw_mm: uint16 array (H, W), values in mm (0 = invalid).
        ir_left:      uint8  array (H, W), grayscale.
        """
        depth_acc = np.zeros((self.height, self.width), dtype=np.float32)
        ir_last = None
        for _ in range(n_average):
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            frames = self._align_to_depth.process(frames)
            depth_frame = frames.get_depth_frame()
            ir_frame    = frames.get_infrared_frame(1)
            depth_acc += np.asanyarray(depth_frame.get_data()).astype(np.float32)
            ir_last = np.asanyarray(ir_frame.get_data())

        depth_avg = (depth_acc / n_average).astype(np.uint16)
        # Convert to mm using depth scale (RealSense units → mm)
        depth_mm = (depth_avg.astype(np.float32) * self.depth_scale * 1000).astype(np.uint16)
        return depth_mm, ir_last

    def capture_rgb_frame(self, n_average: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture and return (color_rgb, depth_raw_mm) aligned to the color camera.
        color_rgb:    uint8 array (H, W, 3), RGB.
        depth_raw_mm: uint16 array (H, W), depth in mm aligned to color frame.
        """
        color_last = None
        depth_acc = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(n_average):
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            frames = self._align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_last = np.asanyarray(color_frame.get_data())
            depth_acc += np.asanyarray(depth_frame.get_data()).astype(np.float32)

        depth_avg = (depth_acc / n_average).astype(np.uint16)
        depth_mm  = (depth_avg.astype(np.float32) * self.depth_scale * 1000).astype(np.uint16)
        return color_last, depth_mm

    def pixel_to_camera_3d(self, u: float, v: float, depth_m: float,
                            use_color_intrinsics: bool = False) -> np.ndarray:
        """
        Back-project a pixel (u, v) + depth into a 3-D point in the camera frame.
        Returns [Xc, Yc, Zc] in metres.
        Requires calibrated intrinsics (populated by start()).
        """
        intr = self.color_intrinsics if use_color_intrinsics else self.depth_intrinsics
        point = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
        return np.array(point, dtype=np.float64)  # [Xc, Yc, Zc] in metres


# ---------------------------------------------------------------------------
# Robot interface — STUB (replace with your robot's SDK)
# ---------------------------------------------------------------------------

class RobotInterface:
    """
    Stub robot interface. Replace with your actual robot SDK calls.

    Pose convention: (x, y, z, rx, ry, rz) in robot base frame.
    x, y, z in metres; rx, ry, rz are Euler angles in degrees (or use
    quaternion — adapt to your robot's API).
    """

    # Fixed overhead position where the robot takes the area depth scan.
    # Fill in the actual values from your robot teach-in / calibration.
    AREA_SCAN_POSE = {
        "x": 0.0,   # metres — REPLACE
        "y": 0.0,   # metres — REPLACE
        "z": 0.8,   # metres — REPLACE (height above table)
        "rx": 0.0,  # degrees — pointing straight down
        "ry": 0.0,
        "rz": 0.0,
    }

    # Height (z) for the close-up RGB scan, in metres above the table surface.
    RGB_SCAN_HEIGHT = 0.30  # REPLACE

    # Approach height before final grasp descent, metres above table.
    GRASP_APPROACH_HEIGHT = 0.15  # REPLACE

    def move_to_area_scan_pose(self):
        """Move to fixed overhead position for depth/IR area scan."""
        log.info("[ROBOT STUB] Moving to area scan pose: %s", self.AREA_SCAN_POSE)
        # TODO: call robot.move_joint(...) or robot.move_cartesian(...)
        time.sleep(0.5)  # simulate motion

    def move_above(self, x_robot: float, y_robot: float, z_height: float,
                   angle_deg: float = 0.0):
        """Move end-effector to (x, y) at z_height, tool rotated by angle_deg."""
        log.info("[ROBOT STUB] Moving above (%.4f, %.4f) at z=%.4f m, angle=%.1f°",
                 x_robot, y_robot, z_height, angle_deg)
        # TODO: call robot.move_cartesian(x, y, z, rx=0, ry=0, rz=angle_deg)
        time.sleep(0.5)

    def get_current_pose(self) -> dict:
        """Return current end-effector pose as a dict."""
        # TODO: return actual pose from robot
        return {**self.AREA_SCAN_POSE}

    def execute_grasp(self, grasp: GraspPose):
        """Lower to grasp height, activate gripper, lift."""
        log.info("[ROBOT STUB] Executing grasp: %s", grasp)
        # TODO: implement actual grasp sequence
        # 1. Move to approach height
        # 2. Open gripper
        # 3. Descend to grasp z
        # 4. Close gripper
        # 5. Lift


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def load_hand_eye_transform(path: str = "calibration/T_cam_to_ee.npy") -> np.ndarray:
    """
    Load the hand-eye calibration transform T_cam_to_ee (4×4).
    This is the pose of the camera frame expressed in the end-effector frame.
    Obtained via hand-eye calibration — see CALIBRATION.md.
    """
    try:
        T = np.load(path)
        assert T.shape == (4, 4), "Expected 4×4 matrix"
        log.info("Loaded hand-eye transform from %s", path)
        return T
    except FileNotFoundError:
        log.warning("Hand-eye calibration file not found at '%s'. "
                    "Using identity — results will be WRONG until calibrated.", path)
        return np.eye(4)


def ee_pose_to_matrix(pose: dict) -> np.ndarray:
    """
    Convert end-effector pose dict {x, y, z, rx, ry, rz} to a 4×4 matrix.
    rx, ry, rz are ZYX Euler angles in degrees (adapt to your robot's convention).
    """
    rx = np.radians(pose.get("rx", 0.0))
    ry = np.radians(pose.get("ry", 0.0))
    rz = np.radians(pose.get("rz", 0.0))

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz),  np.cos(rz), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx  # ZYX convention — change if your robot uses a different one
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [pose["x"], pose["y"], pose["z"]]
    return T


def camera_point_to_robot(p_camera: np.ndarray, T_cam_to_ee: np.ndarray,
                           T_ee_to_base: np.ndarray) -> np.ndarray:
    """
    Transform a 3-D point from the camera frame to the robot base frame.

    T_cam_to_ee  : camera→end-effector (hand-eye calibration result)
    T_ee_to_base : end-effector→robot-base (robot forward kinematics at capture time)
    Returns [Xr, Yr, Zr] in metres.
    """
    p_h = np.array([*p_camera, 1.0])          # homogeneous
    T_cam_to_base = T_ee_to_base @ T_cam_to_ee
    p_robot = (T_cam_to_base @ p_h)[:3]
    return p_robot


# ---------------------------------------------------------------------------
# Detection logic (from the notebooks)
# ---------------------------------------------------------------------------

def find_motor_in_depth(
    depth_mm: np.ndarray,
    near_percentile: float = 10,
    min_valid_mm: int = 150,
    max_valid_mm: int = 4000,
    min_area_px: int = 2500,
    max_area_px: int = 50000,
    morph_kernel: int = 5,
    pad_px: int = 20,
    depth_band_mm: float = 80,
) -> dict:
    """
    Locate the motor in a depth image. Returns detection dict with:
      - 'center_uv': (u, v) in depth image pixels (float)
      - 'depth_mm':  median depth of the detected region
      - 'roi':       (x, y, w, h) bounding box in pixels
      - 'box':       rotated rectangle corners (4×2 int32)
    Raises ValueError if no motor found.
    """
    depth = depth_mm.astype(np.float32)
    valid = (depth >= min_valid_mm) & (depth <= max_valid_mm)

    valid_values = depth[valid]
    if valid_values.size < 1000:
        raise ValueError("Not enough valid depth points.")

    near_threshold = np.percentile(valid_values, near_percentile)
    near_mask = ((depth <= near_threshold) & valid).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(near_mask, connectivity=8)

    best = None
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area_px or area > max_area_px:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        _, (rw, rh), _ = rect
        if rw < 1 or rh < 1:
            continue
        fill_ratio   = cv2.contourArea(contour) / (rw * rh + 1e-6)
        aspect_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        score = area * fill_ratio / (1.0 + abs(aspect_ratio - 1.0))
        candidate = dict(score=score, label=label, area=int(area),
                         contour=contour, rect=rect)
        if best is None or score > best["score"]:
            best = candidate

    if best is None:
        raise ValueError("No motor candidate found in depth image.")

    # Refine by depth band around the seed median
    seed_mask  = labels == best["label"]
    seed_depth = depth[seed_mask & valid]
    seed_median = float(np.median(seed_depth))

    depth_consistency = (np.abs(depth - seed_median) <= depth_band_mm) & valid
    refine_mask = (depth_consistency & seed_mask).astype(np.uint8) * 255
    refine_mask = cv2.morphologyEx(refine_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refine_mask = cv2.morphologyEx(refine_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours_r, _ = cv2.findContours(refine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours_r, key=cv2.contourArea) if contours_r else best["contour"]

    rect    = cv2.minAreaRect(contour)
    box     = cv2.boxPoints(rect).astype(np.int32)
    cx, cy  = rect[0]

    bx, by, bw, bh = cv2.boundingRect(box)
    bx = max(0, bx - pad_px)
    by = max(0, by - pad_px)
    bw = min(depth.shape[1] - bx, bw + 2 * pad_px)
    bh = min(depth.shape[0] - by, bh + 2 * pad_px)

    return {
        "center_uv": (cx, cy),
        "depth_mm":  seed_median,
        "roi":       (bx, by, bw, bh),
        "box":       box,
        "contour":   contour,
    }


def detect_motor_orientation(image_bgr: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    Detect motor orientation and centroid from an RGB (BGR) close-up image.
    Returns (angle_deg, (cx, cy)):
      - angle_deg: directed long-axis orientation in [0, 360)
        (0°=right, 90°=down, 180°=left, 270°=up in image coords)
      - (cx, cy): centroid in image pixels
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    center, (w, h), angle = rect
    if w < h:
        angle = angle + 90
    angle = angle % 180  # long-axis, still 180° ambiguous

    M  = cv2.moments(contour)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Resolve 180° ambiguity via cross-product with centroid vector
    dx = cx - center[0]
    dy = cy - center[1]
    ax = np.cos(np.radians(angle))
    ay = np.sin(np.radians(angle))
    cross = ax * dy - ay * dx
    if cross < 0:
        angle = (angle + 180) % 360

    return float(angle), (float(cx), float(cy))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class MotorGraspPipeline:
    """
    Orchestrates the two-phase motor detection and grasp pose computation.

    Phase 1 — Area scan  (depth + IR camera):
        Robot at fixed overhead position → detect motor location roughly.

    Phase 2 — Precision scan (RGB camera):
        Robot above motor → detect precise centroid + orientation.
    """

    def __init__(self,
                 hand_eye_path: str = "calibration/T_cam_to_ee.npy",
                 save_debug_images: bool = True,
                 debug_dir: str = "data/pipeline"):
        self.camera = CameraInterface()
        self.robot  = RobotInterface()
        self.T_cam_to_ee = load_hand_eye_transform(hand_eye_path)
        self.save_debug  = save_debug_images
        self.debug_dir   = debug_dir

        if save_debug_images:
            import os
            os.makedirs(debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> GraspPose:
        """
        Execute the full pipeline and return the grasp pose.
        The caller is responsible for calling camera.start() / stop() externally,
        or use the context manager (with MotorGraspPipeline(...) as p: p.run()).
        """
        log.info("=== Phase 1: Area scan (depth) ===")
        depth_detection = self._phase1_area_scan()

        log.info("=== Phase 2: Precision scan (RGB) ===")
        grasp = self._phase2_precision_scan(depth_detection)

        log.info("=== Grasp pose ready: %s ===", grasp)
        return grasp

    # ------------------------------------------------------------------
    # Context manager (starts / stops the camera automatically)
    # ------------------------------------------------------------------

    def __enter__(self):
        self.camera.start()
        return self

    def __exit__(self, *_):
        self.camera.stop()

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _phase1_area_scan(self) -> DepthDetection:
        # 1a. Move robot to fixed overhead position
        self.robot.move_to_area_scan_pose()
        ee_pose = self.robot.get_current_pose()
        T_ee_to_base = ee_pose_to_matrix(ee_pose)

        # 1b. Capture depth frame
        log.info("Capturing depth frame…")
        depth_mm, ir_left = self.camera.capture_depth_frame(n_average=5)

        # 1c. Detect motor in depth image
        detection = find_motor_in_depth(depth_mm)
        u, v = detection["center_uv"]
        depth_m = detection["depth_mm"] / 1000.0  # mm → metres
        log.info("Motor detected in depth image at (%.1f, %.1f) px, depth=%.3f m", u, v, depth_m)

        # 1d. Back-project to camera frame, then robot frame
        p_camera = self.camera.pixel_to_camera_3d(u, v, depth_m, use_color_intrinsics=False)
        p_robot  = camera_point_to_robot(p_camera, self.T_cam_to_ee, T_ee_to_base)
        log.info("Motor in robot frame (area scan): x=%.4f y=%.4f z=%.4f m",
                 p_robot[0], p_robot[1], p_robot[2])

        if self.save_debug:
            self._save_depth_debug(ir_left, depth_mm, detection)

        return DepthDetection(
            center_image=detection["center_uv"],
            center_camera=p_camera,
            center_robot=p_robot,
            roi=detection["roi"],
            depth_mm=detection["depth_mm"],
        )

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _phase2_precision_scan(self, area_result: DepthDetection) -> GraspPose:
        # 2a. Move robot directly above the approximate motor location
        target_x = area_result.center_robot[0]
        target_y = area_result.center_robot[1]
        self.robot.move_above(target_x, target_y, z_height=self.robot.RGB_SCAN_HEIGHT)
        ee_pose = self.robot.get_current_pose()
        T_ee_to_base = ee_pose_to_matrix(ee_pose)

        # 2b. Capture RGB frame (+ aligned depth for 3D back-projection)
        log.info("Capturing RGB frame…")
        color_rgb, depth_mm_aligned = self.camera.capture_rgb_frame(n_average=3)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        # 2c. Detect orientation + centroid in image
        angle_deg, (cx_px, cy_px) = detect_motor_orientation(color_bgr)
        log.info("Orientation: %.1f°, centroid px=(%.1f, %.1f)", angle_deg, cx_px, cy_px)

        # 2d. Back-project centroid pixel to 3D using aligned depth
        cx_i, cy_i = int(round(cx_px)), int(round(cy_px))
        cx_i = np.clip(cx_i, 0, depth_mm_aligned.shape[1] - 1)
        cy_i = np.clip(cy_i, 0, depth_mm_aligned.shape[0] - 1)
        centroid_depth_m = float(depth_mm_aligned[cy_i, cx_i]) / 1000.0

        if centroid_depth_m < 0.05:
            log.warning("Depth at centroid pixel is near zero (%.3f m). "
                        "Falling back to area-scan depth.", centroid_depth_m)
            centroid_depth_m = area_result.depth_mm / 1000.0

        p_camera = self.camera.pixel_to_camera_3d(cx_px, cy_px, centroid_depth_m,
                                                   use_color_intrinsics=True)
        p_robot  = camera_point_to_robot(p_camera, self.T_cam_to_ee, T_ee_to_base)
        log.info("Motor centroid in robot frame: x=%.4f y=%.4f z=%.4f m",
                 p_robot[0], p_robot[1], p_robot[2])

        if self.save_debug:
            self._save_rgb_debug(color_bgr, cx_px, cy_px, angle_deg)

        # 2e. The image orientation angle maps to robot tool rotation around Z.
        # The relationship depends on how the camera is mounted on the robot.
        # Adjust tool_angle_deg with your mounting offset if needed.
        tool_angle_deg = self._image_angle_to_robot_angle(angle_deg, ee_pose)

        return GraspPose(
            x=float(p_robot[0]),
            y=float(p_robot[1]),
            z=float(p_robot[2]),
            angle_deg=tool_angle_deg,
        )

    def _image_angle_to_robot_angle(self, image_angle_deg: float, ee_pose: dict) -> float:
        """
        Convert the motor's orientation angle in the image to a robot tool
        rotation angle (around Z-axis of the robot base frame).

        The camera's X-axis in image space corresponds to a direction in the
        robot frame that depends on how the camera is mounted. After hand-eye
        calibration you can compute this exactly.

        For now: assumes camera is mounted pointing straight down with the
        image X-axis aligned with the robot X-axis and no roll.
        Adjust `mounting_offset_deg` to match your camera mount.
        """
        mounting_offset_deg = 0.0  # REPLACE after measuring your camera mount
        robot_yaw = ee_pose.get("rz", 0.0)  # current tool rotation in robot frame

        # Angle in image (measured from image +X axis, CW positive in image coords)
        # → negate for CCW robot convention, add current robot yaw and mount offset
        tool_angle_deg = (-image_angle_deg + robot_yaw + mounting_offset_deg) % 360
        return tool_angle_deg

    # ------------------------------------------------------------------
    # Debug image saving
    # ------------------------------------------------------------------

    def _save_depth_debug(self, ir_left, depth_mm, detection):
        import os
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        vis = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [detection["box"]], -1, (0, 255, 0), 3)
        x, y, w, h = detection["roi"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cx, cy = [int(round(v)) for v in detection["center_uv"]]
        cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

        path = os.path.join(self.debug_dir, f"{ts}_phase1_depth.png")
        cv2.imwrite(path, vis)
        log.info("Saved phase-1 debug image: %s", path)

    def _save_rgb_debug(self, color_bgr, cx_px, cy_px, angle_deg):
        import os
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        vis = color_bgr.copy()
        cx, cy = int(round(cx_px)), int(round(cy_px))
        length = 80
        dx = int(length * np.cos(np.radians(angle_deg)))
        dy = int(length * np.sin(np.radians(angle_deg)))
        cv2.arrowedLine(vis, (cx - dx, cy - dy), (cx + dx, cy + dy),
                        (0, 220, 255), 3, tipLength=0.15)
        cv2.circle(vis, (cx, cy), 8, (255, 0, 0), -1)
        cv2.putText(vis, f"{angle_deg:.1f} deg", (cx + 12, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        path = os.path.join(self.debug_dir, f"{ts}_phase2_rgb.png")
        cv2.imwrite(path, vis)
        log.info("Saved phase-2 debug image: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with MotorGraspPipeline(save_debug_images=True) as pipeline:
        grasp = pipeline.run()

    print("\n" + "=" * 50)
    print("GRASP POSE:")
    print(f"  Position : x={grasp.x:.4f} m, y={grasp.y:.4f} m, z={grasp.z:.4f} m")
    print(f"  Tool yaw : {grasp.angle_deg:.1f}°")
    print("=" * 50)

    # TODO: send grasp pose to robot
    # robot = RobotInterface()
    # robot.move_above(grasp.x, grasp.y, RobotInterface.GRASP_APPROACH_HEIGHT, grasp.angle_deg)
    # robot.execute_grasp(grasp)
