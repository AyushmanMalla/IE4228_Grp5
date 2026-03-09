"""Live face recognition demo GUI.

A real-time webcam feed with bounding boxes, identity labels,
confidence scores, and FPS counter.

Design: Dark-mode monitoring dashboard (OLED-inspired)
Stack: Tkinter + OpenCV (no extra dependencies)

Usage:
    python -m facerec.gui [--gallery-dir data/gallery] [--device cpu]
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk

from facerec.alignment import align_face
from facerec.config import Config
from facerec.database import GalleryDatabase
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer


# ---------------------------------------------------------------------------
# Design tokens  (Dark Mode OLED + status indicators)
# ---------------------------------------------------------------------------

class Theme:
    BG_PRIMARY = "#0A0E1A"       # deep navy-black
    BG_SECONDARY = "#111827"     # panel background
    BG_CARD = "#1E293B"          # card / sidebar panel
    BORDER = "#334155"           # subtle borders
    TEXT_PRIMARY = "#F1F5F9"     # near-white
    TEXT_SECONDARY = "#94A3B8"   # muted grey
    TEXT_DIM = "#64748B"         # dimmer text
    ACCENT_BLUE = "#3B82F6"     # primary accent
    ACCENT_GREEN = "#22C55E"    # known / success
    ACCENT_RED = "#EF4444"      # unknown / alert
    ACCENT_AMBER = "#F59E0B"    # warning / medium conf
    ACCENT_CYAN = "#06B6D4"     # info highlight
    FONT_FAMILY = "Helvetica Neue"
    FONT_MONO = "SF Mono"


# Colours for OpenCV drawing (BGR)
CV_GREEN = (94, 197, 34)        # known face
CV_RED = (68, 68, 239)          # unknown face
CV_AMBER = (11, 158, 245)      # medium confidence
CV_WHITE = (241, 245, 249)
CV_BG_OVERLAY = (26, 14, 10)   # dark overlay


import queue
import threading
from dataclasses import dataclass


@dataclass
class TrackedFace:
    """Represents a face tracked across frames."""
    bbox: np.ndarray      # [x1, y1, x2, y2]
    name: str = "Unknown"
    score: float = 0.0
    tracker: cv2.Tracker | None = None
    frames_since_detect: int = 0
    needs_embedding: bool = True
    aligned_crop: np.ndarray | None = None
    
    @property
    def w(self) -> float: return self.bbox[2] - self.bbox[0]
    @property
    def h(self) -> float: return self.bbox[3] - self.bbox[1]


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class FaceRecognitionGUI:
    """Tkinter-based real-time face recognition dashboard with Async Inference."""

    WINDOW_TITLE = "IE4228 · Face Recognition System"
    VIDEO_W, VIDEO_H = 800, 600
    SIDEBAR_W = 280
    
    # Optimization tweaks
    DETECT_EVERY_N_FRAMES = 10
    DETECT_SCALE = 0.5   # Downscale frame for detection

    def __init__(
        self,
        gallery_dir: Path,
        device: str = "cpu",
        camera_index: int = 0,
    ) -> None:
        self._gallery_dir = gallery_dir
        self._device = device
        self._similarity_threshold = 0.35
        self._camera_index = camera_index
        
        # Async threading state
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._is_running = True
        
        # UI state
        self._cap: cv2.VideoCapture | None = None
        self._frame_times: list[float] = []
        self._fps = 0.0
        self._last_results: list[dict] = []
        self._total_detections = 0
        self._total_known = 0
        self._total_unknown = 0
        
        # Build UI first
        self._build_window()
        
        # Start ML worker thread
        self._ml_thread = threading.Thread(target=self._ml_worker_loop, daemon=True)
        self._ml_thread.start()

    # ==================================================================
    # UI Construction (unchanged)
    # ==================================================================

    def _build_window(self) -> None:
        self._root = tk.Tk()
        self._root.title(self.WINDOW_TITLE)
        self._root.configure(bg=Theme.BG_PRIMARY)
        self._root.resizable(False, False)

        self._font_title = tkfont.Font(family=Theme.FONT_FAMILY, size=16, weight="bold")
        self._font_heading = tkfont.Font(family=Theme.FONT_FAMILY, size=12, weight="bold")
        self._font_body = tkfont.Font(family=Theme.FONT_FAMILY, size=11)
        self._font_small = tkfont.Font(family=Theme.FONT_FAMILY, size=10)
        self._font_mono = tkfont.Font(family=Theme.FONT_MONO, size=11)

        self._build_header()
        body = tk.Frame(self._root, bg=Theme.BG_PRIMARY)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._build_video_panel(body)
        self._build_sidebar(body)

    def _build_header(self) -> None:
        header = tk.Frame(self._root, bg=Theme.BG_PRIMARY, height=56)
        header.pack(fill=tk.X, padx=12, pady=(12, 8))
        tk.Label(
            header, text="◉  Face Recognition", font=self._font_title,
            bg=Theme.BG_PRIMARY, fg=Theme.ACCENT_CYAN,
        ).pack(side=tk.LEFT)
        self._status_label = tk.Label(
            header, text="● INITING...", font=self._font_small,
            bg=Theme.BG_PRIMARY, fg=Theme.ACCENT_AMBER,
        )
        self._status_label.pack(side=tk.RIGHT, padx=(0, 8))
        self._fps_label = tk.Label(
            header, text="0.0 FPS", font=self._font_mono,
            bg=Theme.BG_PRIMARY, fg=Theme.TEXT_DIM,
        )
        self._fps_label.pack(side=tk.RIGHT, padx=(0, 16))

    def _build_video_panel(self, parent: tk.Frame) -> None:
        video_frame = tk.Frame(
            parent, bg=Theme.BG_SECONDARY, highlightbackground=Theme.BORDER, highlightthickness=1
        )
        video_frame.pack(side=tk.LEFT, padx=(0, 8))
        self._canvas = tk.Canvas(
            video_frame, width=self.VIDEO_W, height=self.VIDEO_H,
            bg=Theme.BG_SECONDARY, highlightthickness=0,
        )
        self._canvas.pack(padx=2, pady=2)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        sidebar = tk.Frame(parent, bg=Theme.BG_SECONDARY, width=self.SIDEBAR_W)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Gallery
        self._section_header(sidebar, "GALLERY")
        self._gallery_frame = tk.Frame(sidebar, bg=Theme.BG_SECONDARY)
        self._gallery_frame.pack(fill=tk.X)
        tk.Label(self._gallery_frame, text="Loading...", bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack()

        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=12)

        # Stats
        self._section_header(sidebar, "LIVE STATS")
        self._stat_faces = self._stat_row(sidebar, "Faces Detected")
        self._stat_known = self._stat_row(sidebar, "Known")
        self._stat_unknown = self._stat_row(sidebar, "Unknown")

        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=12)

        # Current
        self._section_header(sidebar, "CURRENT FRAME")
        self._detections_frame = tk.Frame(sidebar, bg=Theme.BG_SECONDARY)
        self._detections_frame.pack(fill=tk.X, padx=12, pady=4)

        tk.Frame(sidebar, bg=Theme.BG_SECONDARY).pack(fill=tk.BOTH, expand=True)

        # Controls
        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=4)
        ctrl = tk.Frame(sidebar, bg=Theme.BG_SECONDARY)
        ctrl.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(ctrl, text="Threshold", font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor=tk.W)
        self._threshold_var = tk.DoubleVar(value=self._similarity_threshold)
        tk.Scale(
            ctrl, from_=0.1, to=0.8, resolution=0.05, orient=tk.HORIZONTAL, variable=self._threshold_var,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY, troughcolor=Theme.BG_CARD, highlightthickness=0,
            font=self._font_small, length=self.SIDEBAR_W - 40, command=self._on_threshold_change,
        ).pack(fill=tk.X, pady=4)

    def _section_header(self, parent, text):
        tk.Label(parent, text=text, font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.ACCENT_BLUE).pack(anchor=tk.W, padx=16, pady=(12, 4))

    def _stat_row(self, parent, label):
        row = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(row, text=label, font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side=tk.LEFT)
        val = tk.Label(row, text="0", font=self._font_mono, bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY)
        val.pack(side=tk.RIGHT)
        return val

    def _on_threshold_change(self, val: str) -> None:
        self._similarity_threshold = float(val)

    # ==================================================================
    # Async Background ML Worker
    # ==================================================================
    
    def _ml_worker_loop(self) -> None:
        """Runs in separate thread: loads models and does heavy inference."""
        # 1. Load models
        detector = FaceDetector(device=self._device, det_thresh=0.5)
        recognizer = FaceRecognizer(device=self._device)
        gallery = GalleryDatabase(self._gallery_dir)
        if (self._gallery_dir / "gallery.json").exists():
            gallery.load()
            
        # Update UI with gallery contents safely
        def _update_gallery_ui():
            for w in self._gallery_frame.winfo_children(): w.destroy()
            names = gallery.list_identities()
            if not names:
                tk.Label(self._gallery_frame, text="No identities", bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack(anchor=tk.W, padx=16)
            for name in names:
                row = tk.Frame(self._gallery_frame, bg=Theme.BG_SECONDARY)
                row.pack(fill=tk.X, padx=16, pady=2)
                tk.Label(row, text="●", font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.ACCENT_GREEN).pack(side=tk.LEFT, padx=(0, 6))
                tk.Label(row, text=name.replace("_", " "), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side=tk.LEFT)
            self._status_label.config(text="● LIVE", fg=Theme.ACCENT_GREEN)
                
        self._root.after(0, _update_gallery_ui)

        tracked_faces: list[TrackedFace] = []
        frame_idx = 0

        while self._is_running:
            try:
                # Get latest frame
                frame = self._frame_queue.get(timeout=0.1)
                frame_idx += 1
            except queue.Empty:
                continue
                
            results = []
            
            # --- Tracking Phase ---
            # Update trackers on intermediate frames
            alive_faces = []
            for face in tracked_faces:
                face.frames_since_detect += 1
                if face.tracker is not None:
                    # OpenCV tracker expects uint8 BGR
                    ok, bbox = face.tracker.update(frame)
                    if ok:
                        # Tracker output is (x, y, w, h)
                        x, y, w, h = bbox
                        face.bbox = np.array([x, y, x+w, y+h])
                        
                        # Only keep if box is somewhat sane
                        if w > 20 and h > 20 and x >= -w and y >= -h:
                            alive_faces.append(face)
            
            tracked_faces = alive_faces

            # --- Detection Phase (every N frames) ---
            if frame_idx % self.DETECT_EVERY_N_FRAMES == 0 or not tracked_faces:
                # Downscale for speed
                small_frame = cv2.resize(frame, (0, 0), fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)
                dets = detector.detect(small_frame)
                
                new_tracked_faces = []
                for det in dets:
                    # Scale bbox back up
                    box = det.bbox / self.DETECT_SCALE
                    
                    # Match bounding box to existing trackers using IoU
                    matched_face = None
                    best_iou = 0.3
                    
                    for face in tracked_faces:
                        # Calculate IoU
                        xA = max(box[0], face.bbox[0])
                        yA = max(box[1], face.bbox[1])
                        xB = min(box[2], face.bbox[2])
                        yB = min(box[3], face.bbox[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        boxAArea = (box[2] - box[0]) * (box[3] - box[1])
                        boxBArea = face.w * face.h
                        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
                        
                        if iou > best_iou:
                            best_iou = iou
                            matched_face = face
                            
                    if matched_face is not None:
                        # Update existing face's bbox, keep name
                        matched_face.bbox = box
                        matched_face.frames_since_detect = 0
                        # Re-init tracker with new box
                        self._init_tracker(matched_face, frame)
                        new_tracked_faces.append(matched_face)
                        tracked_faces.remove(matched_face) # prevent multiple matches
                    else:
                        # Brand new face
                        new_face = TrackedFace(bbox=box)
                        # Landmarks scale up
                        lms = det.landmarks / self.DETECT_SCALE
                        # Extract aligned crop for embedding
                        new_face.aligned_crop = align_face(frame, lms, output_size=112)
                        self._init_tracker(new_face, frame)
                        new_tracked_faces.append(new_face)
                        
                tracked_faces = new_tracked_faces
                
            # --- Recognition Phase (only for new faces) ---
            for face in tracked_faces:
                if face.needs_embedding and face.aligned_crop is not None:
                    emb = recognizer.get_embedding(face.aligned_crop)
                    name, score = gallery.query(emb, threshold=self._similarity_threshold)
                    face.name = name
                    face.score = score
                    face.needs_embedding = False
                    face.aligned_crop = None # free mem
                    
                results.append({
                    "bbox": face.bbox,
                    "name": face.name,
                    "score": face.score
                })

            # Send back to UI
            try:
                # Non-blocking put, overwrite if full
                if self._result_queue.full():
                    self._result_queue.get_nowait()
                self._result_queue.put_nowait(results)
            except queue.Full:
                pass

    def _init_tracker(self, face: TrackedFace, frame: np.ndarray) -> None:
        """Initialize OpenCV CSRT tracker for a face bounding box."""
        try:
            # Most modern OpenCV builds
            face.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            # Older/headless OpenCV
            try:
                face.tracker = cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                face.tracker = None
                return
                
        # Format: (x, y, w, h)
        box = face.bbox
        x, y, w, h = max(0, int(box[0])), max(0, int(box[1])), int(box[2] - box[0]), int(box[3] - box[1])
        if w > 0 and h > 0:
            face.tracker.init(frame, (x, y, w, h))

    # ==================================================================
    # Video loop (Main Thread)
    # ==================================================================

    def _start_camera(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._status_label.config(text="● NO CAMERA", fg=Theme.ACCENT_RED)

    def _process_frame(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            self._root.after(100, self._process_frame)
            return

        t0 = time.perf_counter()

        ret, frame = self._cap.read()
        if not ret:
            self._root.after(30, self._process_frame)
            return

        # Resize for display
        display_frame = cv2.resize(frame, (self.VIDEO_W, self.VIDEO_H))
        
        # Dispatch to ML worker (non-blocking)
        try:
            if self._frame_queue.full():
                self._frame_queue.get_nowait()
            self._frame_queue.put_nowait(display_frame.copy())
        except queue.Full:
            pass
            
        # Get latest ML results (non-blocking)
        try:
            self._last_results = self._result_queue.get_nowait()
            
            # Update stats only when we get new results
            self._total_detections += len(self._last_results)
            known = sum(1 for r in self._last_results if r["name"] != "Unknown")
            self._total_known += known
            self._total_unknown += len(self._last_results) - known
            
            self._stat_faces.config(text=str(self._total_detections))
            self._stat_known.config(text=str(self._total_known))
            self._stat_unknown.config(text=str(self._total_unknown))
            self._update_detections_panel(self._last_results)
            
        except queue.Empty:
            pass

        # Draw latest known bounding boxes
        for result in self._last_results:
            self._draw_detection(display_frame, result)

        # --- FPS ---
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self._fps = 1.0 / avg if avg > 0 else 0
        self._fps_label.config(text=f"{self._fps:.1f} FPS")

        # --- Display frame ---
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self._canvas._imgtk = imgtk

        self._root.after(1, self._process_frame)

    def _draw_detection(self, frame: np.ndarray, result: dict) -> None:
        x1, y1, x2, y2 = result["bbox"].astype(int)
        name = result["name"]
        score = result["score"]
        is_known = name != "Unknown"

        color = CV_GREEN if is_known else CV_RED
        if is_known and score < 0.5:
            color = CV_AMBER

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name.replace('_', ' ')}"
        conf_text = f"{score:.0%}"
        label_w = max(len(label), len(conf_text)) * 11 + 16
        label_h = 44

        ly = max(y1 - label_h - 4, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, ly), (x1 + label_w, ly + label_h), CV_BG_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, label, (x1 + 8, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CV_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (x1 + 8, ly + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        corner_len = 12
        for pt in [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]:
            cx, cy = pt
            dx = corner_len if cx == x1 else -corner_len
            dy = corner_len if cy == y1 else -corner_len
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, 3)

    def _update_detections_panel(self, results: list[dict]) -> None:
        for widget in self._detections_frame.winfo_children(): widget.destroy()
        if not results:
            tk.Label(self._detections_frame, text="No faces", font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack(anchor=tk.W)
            return

        for r in results[:6]:
            row = tk.Frame(self._detections_frame, bg=Theme.BG_SECONDARY)
            row.pack(fill=tk.X, pady=1)
            dot_color = Theme.ACCENT_GREEN if r["name"] != "Unknown" else Theme.ACCENT_RED
            tk.Label(row, text="●", font=self._font_small, bg=Theme.BG_SECONDARY, fg=dot_color).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(row, text=r["name"].replace("_", " "), font=self._font_small, bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side=tk.LEFT)
            tk.Label(row, text=f'{r["score"]:.0%}', font=self._font_mono, bg=Theme.BG_SECONDARY, fg=dot_color).pack(side=tk.RIGHT)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self) -> None:
        self._start_camera()
        self._root.after(10, self._process_frame)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_close(self) -> None:
        self._is_running = False
        if self._cap is not None:
            self._cap.release()
        self._root.destroy()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Live face recognition demo")
    parser.add_argument(
        "--gallery-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gallery",
        help="Path to gallery database directory",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    app = FaceRecognitionGUI(
        gallery_dir=args.gallery_dir,
        device=args.device,
        camera_index=args.camera,
    )
    app.run()


if __name__ == "__main__":
    main()
