# Motor Grasp Pipeline

Perception pipeline for detecting and grasping an electric motor using an Intel RealSense D435i depth camera. Estimates motor orientation from a top-down view using CAD template matching and rectangle-based centroid detection, then computes a grasp pose for the robot arm.

## Repository Structure

```
hakaton/
├── notebooks/
│   ├── realsense_viewer.ipynb       # Live RealSense stream & frame capture
│   ├── logi_capture.ipynb           # Logitech camera capture & annotation
│   ├── motor_detection.ipynb        # Motor detection exploration
│   ├── orientation_estimation.ipynb # Orientation via CAD template matching
│   ├── orientation_rectangle.ipynb  # Orientation via bounding rectangle + centroid (current)
│   └── generate_templates.ipynb     # Generate CAD silhouette templates from e-motor.stp
├── data/
│   ├── area/                        # Captures from the grasp area (RealSense GUI output)
│   ├── cad/                         # CAD model (e-motor.stp)
│   ├── templates_topdown/           # Top-down silhouette templates (5° step, 256px)
│   ├── templates_wide/              # Wide-field silhouette templates (10° step, 128px)
│   └── old/                         # Archived captures no longer in active use
├── realsense_gui.py                 # Live viewer GUI with capture button
├── calibrate_pipeline.py            # Camera-to-robot calibration
├── motor_grasp_pipeline.py          # Full grasp pipeline (detection → orientation → pose)
├── requirements.txt
└── CALIBRATION.md
```

## Setup

**1. System dependencies**

```bash
sudo apt install python3-tk
```

**2. Create and activate virtual environment**

```bash
python3 -m venv realsense_env
source realsense_env/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

## Running

### Full grasp pipeline

```bash
source realsense_env/bin/activate
python3 motor_grasp_pipeline.py
```

### Camera calibration

```bash
source realsense_env/bin/activate
python3 calibrate_pipeline.py
```

See [CALIBRATION.md](CALIBRATION.md) for the full calibration procedure.

### RealSense GUI (live feed + area capture)

```bash
source realsense_env/bin/activate
python3 realsense_gui.py
```

Captures are saved to `data/area/` as PNGs and raw `.npy` depth files.

### Notebooks

```bash
source realsense_env/bin/activate
jupyter lab
```

Open notebooks from the `notebooks/` directory. All notebooks reference `data/` via `../data/`.

## Pipeline Overview

1. **Template generation** (`generate_templates.ipynb`) — renders CAD silhouettes of the motor at every azimuth angle, saved to `data/templates_topdown/` and `data/templates_wide/`.
2. **Orientation estimation** (`orientation_rectangle.ipynb`) — fits a bounding rectangle to the motor mask from the depth image, uses the centroid offset to resolve the 180° ambiguity.
3. **Grasp pipeline** (`motor_grasp_pipeline.py`) — runs detection, orientation estimation, and outputs a 6-DOF grasp pose.

## Notes

- Camera must be connected via USB before starting the pipeline
- USB 2.x limits streams to depth + IR only (no color); max resolution 640×480 @ 15 fps
- If the pipeline fails to start, unplug and replug the camera, then try again
