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


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class FaceRecognitionGUI:
    """Tkinter-based real-time face recognition dashboard."""

    WINDOW_TITLE = "IE4228 · Face Recognition System"
    VIDEO_W, VIDEO_H = 800, 600
    SIDEBAR_W = 280

    def __init__(
        self,
        gallery_dir: Path,
        device: str = "cpu",
        camera_index: int = 0,
    ) -> None:
        # ---- Models ----
        self._detector = FaceDetector(device=device, det_thresh=0.5)
        self._recognizer = FaceRecognizer(device=device)
        self._gallery = GalleryDatabase(gallery_dir)
        if (gallery_dir / "gallery.json").exists():
            self._gallery.load()

        self._similarity_threshold = 0.35
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None

        # ---- FPS tracking ----
        self._frame_times: list[float] = []
        self._fps = 0.0

        # ---- Detection state ----
        self._last_results: list[dict] = []
        self._total_detections = 0
        self._total_known = 0
        self._total_unknown = 0

        # ---- Build UI ----
        self._build_window()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_window(self) -> None:
        self._root = tk.Tk()
        self._root.title(self.WINDOW_TITLE)
        self._root.configure(bg=Theme.BG_PRIMARY)
        self._root.resizable(False, False)

        # Fonts
        self._font_title = tkfont.Font(family=Theme.FONT_FAMILY, size=16, weight="bold")
        self._font_heading = tkfont.Font(family=Theme.FONT_FAMILY, size=12, weight="bold")
        self._font_body = tkfont.Font(family=Theme.FONT_FAMILY, size=11)
        self._font_small = tkfont.Font(family=Theme.FONT_FAMILY, size=10)
        self._font_mono = tkfont.Font(family=Theme.FONT_MONO, size=11)
        self._font_mono_large = tkfont.Font(family=Theme.FONT_MONO, size=22, weight="bold")

        # Main layout: header + body(video + sidebar)
        self._build_header()
        body = tk.Frame(self._root, bg=Theme.BG_PRIMARY)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._build_video_panel(body)
        self._build_sidebar(body)

    def _build_header(self) -> None:
        header = tk.Frame(self._root, bg=Theme.BG_PRIMARY, height=56)
        header.pack(fill=tk.X, padx=12, pady=(12, 8))

        # Title
        tk.Label(
            header, text="◉  Face Recognition", font=self._font_title,
            bg=Theme.BG_PRIMARY, fg=Theme.ACCENT_CYAN,
        ).pack(side=tk.LEFT)

        # Status indicator
        self._status_label = tk.Label(
            header, text="● OFFLINE", font=self._font_small,
            bg=Theme.BG_PRIMARY, fg=Theme.ACCENT_RED,
        )
        self._status_label.pack(side=tk.RIGHT, padx=(0, 8))

        # FPS display
        self._fps_label = tk.Label(
            header, text="0.0 FPS", font=self._font_mono,
            bg=Theme.BG_PRIMARY, fg=Theme.TEXT_DIM,
        )
        self._fps_label.pack(side=tk.RIGHT, padx=(0, 16))

    def _build_video_panel(self, parent: tk.Frame) -> None:
        video_frame = tk.Frame(
            parent, bg=Theme.BG_SECONDARY,
            highlightbackground=Theme.BORDER, highlightthickness=1,
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

        # ---- Gallery section ----
        self._section_header(sidebar, "GALLERY")
        gallery_names = self._gallery.list_identities()
        if gallery_names:
            for name in gallery_names:
                self._gallery_row(sidebar, name)
        else:
            tk.Label(
                sidebar, text="No identities loaded", font=self._font_small,
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM,
            ).pack(anchor=tk.W, padx=16, pady=4)

        # Separator
        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=12)

        # ---- Stats section ----
        self._section_header(sidebar, "LIVE STATS")

        self._stat_faces = self._stat_row(sidebar, "Faces Detected")
        self._stat_known = self._stat_row(sidebar, "Known")
        self._stat_unknown = self._stat_row(sidebar, "Unknown")

        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=12)

        # ---- Current detections ----
        self._section_header(sidebar, "CURRENT FRAME")
        self._detections_frame = tk.Frame(sidebar, bg=Theme.BG_SECONDARY)
        self._detections_frame.pack(fill=tk.X, padx=12, pady=4)

        # Spacer
        tk.Frame(sidebar, bg=Theme.BG_SECONDARY).pack(fill=tk.BOTH, expand=True)

        # ---- Controls ----
        tk.Frame(sidebar, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=12, pady=4)
        ctrl = tk.Frame(sidebar, bg=Theme.BG_SECONDARY)
        ctrl.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(
            ctrl, text="Threshold", font=self._font_small,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(anchor=tk.W)

        self._threshold_var = tk.DoubleVar(value=self._similarity_threshold)
        slider = tk.Scale(
            ctrl, from_=0.1, to=0.8, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self._threshold_var,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_CARD, highlightthickness=0,
            font=self._font_small, length=self.SIDEBAR_W - 40,
            command=self._on_threshold_change,
        )
        slider.pack(fill=tk.X, pady=4)

    # ---- Sidebar helpers ----

    def _section_header(self, parent: tk.Frame, text: str) -> None:
        tk.Label(
            parent, text=text, font=self._font_small,
            bg=Theme.BG_SECONDARY, fg=Theme.ACCENT_BLUE,
        ).pack(anchor=tk.W, padx=16, pady=(12, 4))

    def _gallery_row(self, parent: tk.Frame, name: str) -> None:
        row = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(
            row, text="●", font=self._font_small,
            bg=Theme.BG_SECONDARY, fg=Theme.ACCENT_GREEN,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            row, text=name.replace("_", " "), font=self._font_body,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY, anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X)

    def _stat_row(self, parent: tk.Frame, label: str) -> tk.Label:
        row = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(
            row, text=label, font=self._font_small,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(side=tk.LEFT)
        val = tk.Label(
            row, text="0", font=self._font_mono,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
        )
        val.pack(side=tk.RIGHT)
        return val

    def _on_threshold_change(self, val: str) -> None:
        self._similarity_threshold = float(val)

    # ==================================================================
    # Video loop
    # ==================================================================

    def _start_camera(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if self._cap.isOpened():
            self._status_label.config(text="● LIVE", fg=Theme.ACCENT_GREEN)
        else:
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
        frame = cv2.resize(frame, (self.VIDEO_W, self.VIDEO_H))

        # --- Detection + Recognition ---
        detections = self._detector.detect(frame)
        results: list[dict] = []

        for det in detections:
            aligned = align_face(frame, det.landmarks, output_size=112)
            emb = self._recognizer.get_embedding(aligned)
            name, score = self._gallery.query(emb, threshold=self._similarity_threshold)

            result = {
                "bbox": det.bbox,
                "name": name,
                "score": score,
                "det_conf": det.confidence,
            }
            results.append(result)

            # Draw on frame
            self._draw_detection(frame, result)

        self._last_results = results

        # --- Update stats ---
        self._total_detections += len(results)
        known = sum(1 for r in results if r["name"] != "Unknown")
        unknown = len(results) - known
        self._total_known += known
        self._total_unknown += unknown
        self._stat_faces.config(text=str(self._total_detections))
        self._stat_known.config(text=str(self._total_known))
        self._stat_unknown.config(text=str(self._total_unknown))

        # --- Update current detections sidebar ---
        self._update_detections_panel(results)

        # --- FPS ---
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self._fps = 1.0 / avg if avg > 0 else 0
        self._fps_label.config(text=f"{self._fps:.1f} FPS")

        # --- Display frame ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self._canvas._imgtk = imgtk  # prevent GC

        # Schedule next frame
        self._root.after(1, self._process_frame)

    def _draw_detection(self, frame: np.ndarray, result: dict) -> None:
        """Draw bounding box + label on the frame."""
        x1, y1, x2, y2 = result["bbox"].astype(int)
        name = result["name"]
        score = result["score"]
        is_known = name != "Unknown"

        # Colours
        color = CV_GREEN if is_known else CV_RED
        if is_known and score < 0.5:
            color = CV_AMBER

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{name.replace('_', ' ')}"
        conf_text = f"{score:.0%}"
        label_w = max(len(label), len(conf_text)) * 11 + 16
        label_h = 44

        # Draw label panel above bbox
        ly = max(y1 - label_h - 4, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, ly), (x1 + label_w, ly + label_h), CV_BG_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Text
        cv2.putText(frame, label, (x1 + 8, ly + 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, CV_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (x1 + 8, ly + 36),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Corner accents
        corner_len = 12
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

    def _update_detections_panel(self, results: list[dict]) -> None:
        """Update the sidebar's current-frame detections list."""
        for widget in self._detections_frame.winfo_children():
            widget.destroy()

        if not results:
            tk.Label(
                self._detections_frame, text="No faces", font=self._font_small,
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM,
            ).pack(anchor=tk.W)
            return

        for r in results[:6]:  # show max 6
            row = tk.Frame(self._detections_frame, bg=Theme.BG_SECONDARY)
            row.pack(fill=tk.X, pady=1)
            is_known = r["name"] != "Unknown"
            dot_color = Theme.ACCENT_GREEN if is_known else Theme.ACCENT_RED
            tk.Label(
                row, text="●", font=self._font_small,
                bg=Theme.BG_SECONDARY, fg=dot_color,
            ).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(
                row, text=r["name"].replace("_", " "), font=self._font_small,
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=f'{r["score"]:.0%}', font=self._font_mono,
                bg=Theme.BG_SECONDARY, fg=dot_color,
            ).pack(side=tk.RIGHT)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self) -> None:
        """Start the GUI and webcam loop."""
        self._start_camera()
        self._root.after(10, self._process_frame)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_close(self) -> None:
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
