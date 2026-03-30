import threading
import time
from datetime import datetime

import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.cm as cm


# --- RealSense setup ---

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,    640, 480, rs.format.z16,  15)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)


def depth_to_rgb(depth_frame, vmin=0, vmax=3000):
    """Convert a raw depth array to an RGB image using the plasma colormap."""
    normed = np.clip(depth_frame.astype(float), vmin, vmax) / vmax
    rgba = (cm.plasma(normed) * 255).astype(np.uint8)
    return rgba[:, :, :3]


def gray_to_rgb(gray):
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


# --- GUI ---

class RealSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense Viewer")
        self.root.resizable(False, False)

        self.running = False
        self.capture_requested = False
        self.latest_frames = None  # (ir_left, ir_right, depth) numpy arrays
        self.lock = threading.Lock()

        # Layout: three image panels side by side
        frame_images = tk.Frame(root, bg="black")
        frame_images.pack(padx=8, pady=8)

        label_style = {"bg": "black", "fg": "white", "font": ("Helvetica", 11)}

        tk.Label(frame_images, text="IR Left",  **label_style).grid(row=0, column=0, padx=4)
        tk.Label(frame_images, text="IR Right", **label_style).grid(row=0, column=1, padx=4)
        tk.Label(frame_images, text="Depth",    **label_style).grid(row=0, column=2, padx=4)

        self.canvas_ir_left  = tk.Label(frame_images, bg="black")
        self.canvas_ir_right = tk.Label(frame_images, bg="black")
        self.canvas_depth    = tk.Label(frame_images, bg="black")

        self.canvas_ir_left .grid(row=1, column=0, padx=4, pady=4)
        self.canvas_ir_right.grid(row=1, column=1, padx=4, pady=4)
        self.canvas_depth   .grid(row=1, column=2, padx=4, pady=4)

        # Status bar
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(root, textvariable=self.status_var, anchor="w",
                 relief="sunken", bg="#222", fg="#aaa",
                 font=("Helvetica", 10)).pack(fill="x", padx=8, pady=(0, 4))

        # Buttons
        frame_btns = tk.Frame(root, bg=root.cget("bg"))
        frame_btns.pack(pady=(0, 10))

        btn_style = {"font": ("Helvetica", 12), "width": 12, "pady": 4}

        self.btn_start = tk.Button(frame_btns, text="Start",   bg="#4caf50", fg="white",
                                   command=self.start_stream, **btn_style)
        self.btn_stop  = tk.Button(frame_btns, text="Stop",    bg="#f44336", fg="white",
                                   command=self.stop_stream,  state="disabled", **btn_style)
        self.btn_cap   = tk.Button(frame_btns, text="Capture", bg="#2196f3", fg="white",
                                   command=self.request_capture, state="disabled", **btn_style)

        self.btn_start.grid(row=0, column=0, padx=6)
        self.btn_stop .grid(row=0, column=1, padx=6)
        self.btn_cap  .grid(row=0, column=2, padx=6)

        self._blank = self._make_blank()
        for c in (self.canvas_ir_left, self.canvas_ir_right, self.canvas_depth):
            c.configure(image=self._blank)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------

    def _make_blank(self, w=640, h=480):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img = img.resize((426, 320))
        return ImageTk.PhotoImage(img)

    def _array_to_tk(self, arr, resize=(426, 320)):
        img = Image.fromarray(arr.astype(np.uint8))
        img = img.resize(resize, Image.BILINEAR)
        return ImageTk.PhotoImage(img)

    # ------------------------------------------------------------------

    def start_stream(self):
        self.status_var.set("Starting pipeline…")
        self.btn_start.configure(state="disabled")
        threading.Thread(target=self._start_pipeline, daemon=True).start()

    def _start_pipeline(self):
        try:
            pipeline.start(config)
        except RuntimeError:
            # Already started — stop and restart
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline.start(config)

        self.running = True
        self.root.after(0, lambda: self.btn_stop.configure(state="normal"))
        self.root.after(0, lambda: self.btn_cap .configure(state="normal"))
        self.root.after(0, lambda: self.status_var.set("Streaming…"))
        self._stream_loop()

    def _stream_loop(self):
        while self.running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=30000)
            except RuntimeError:
                break

            ir_left  = np.asanyarray(frames.get_infrared_frame(1).get_data())
            ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())
            depth    = np.asanyarray(frames.get_depth_frame().get_data())

            with self.lock:
                self.latest_frames = (ir_left, ir_right, depth)

            if self.capture_requested:
                self.capture_requested = False
                self._save_capture(ir_left, ir_right, depth)

            # Push update to GUI thread
            tk_il = self._array_to_tk(gray_to_rgb(ir_left))
            tk_ir = self._array_to_tk(gray_to_rgb(ir_right))
            tk_d  = self._array_to_tk(depth_to_rgb(depth))
            self.root.after(0, self._update_canvases, tk_il, tk_ir, tk_d)

        # Stream ended
        try:
            pipeline.stop()
        except Exception:
            pass
        self.root.after(0, self._on_stream_stopped)

    def _update_canvases(self, tk_il, tk_ir, tk_d):
        # Hold references so GC doesn't collect them
        self.canvas_ir_left .configure(image=tk_il); self.canvas_ir_left ._img = tk_il
        self.canvas_ir_right.configure(image=tk_ir); self.canvas_ir_right._img = tk_ir
        self.canvas_depth   .configure(image=tk_d);  self.canvas_depth   ._img = tk_d

    def _on_stream_stopped(self):
        self.btn_start.configure(state="normal")
        self.btn_stop .configure(state="disabled")
        self.btn_cap  .configure(state="disabled")
        self.status_var.set("Stopped.")

    # ------------------------------------------------------------------

    def stop_stream(self):
        self.running = False
        self.status_var.set("Stopping…")

    def request_capture(self):
        self.capture_requested = True
        self.status_var.set("Capture queued…")

    def _save_capture(self, ir_left, ir_right, depth):
        import os
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        Image.fromarray(ir_left) .save(f"data/capture_{ts}_ir_left.png")
        Image.fromarray(ir_right).save(f"data/capture_{ts}_ir_right.png")
        Image.fromarray(depth_to_rgb(depth)).save(f"data/capture_{ts}_depth.png")
        np.save(f"data/capture_{ts}_depth_raw.npy", depth)

        msg = f"Saved data/capture_{ts}_*.png"
        print(msg)
        self.root.after(0, lambda: self.status_var.set(msg))

    # ------------------------------------------------------------------

    def on_close(self):
        self.running = False
        time.sleep(0.1)
        try:
            pipeline.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealSenseApp(root)
    root.mainloop()
