# RealSense Viewer

Live viewer and frame capture for the Intel RealSense D435i over USB 2.x (depth + stereo IR).

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

### GUI viewer (live feed + capture)

```bash
source realsense_env/bin/activate
python3 realsense_gui.py
```

- **Start** — opens the RealSense pipeline and streams IR Left, IR Right, and Depth side by side
- **Stop** — cleanly stops the pipeline
- **Capture** — saves the next frame to `data/` as PNGs and a raw `.npy` depth file

### Jupyter notebook

```bash
source realsense_env/bin/activate
jupyter lab realsense_viewer.ipynb
```

## Output

Captured frames are saved to `data/` with filenames like:

```
data/capture_20260330_143201_ir_left.png
data/capture_20260330_143201_ir_right.png
data/capture_20260330_143201_depth.png
data/capture_20260330_143201_depth_raw.npy
```

## Notes

- Camera must be connected via USB before starting the pipeline
- USB 2.x limits streams to depth + IR only (no color); max resolution 640×480 @ 15 fps
- If the pipeline fails to start, unplug and replug the camera, then try again
