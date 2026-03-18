"""Live face recognition demo GUI using PySide6.

A real-time webcam feed with entirely decoupled Camera IO and ML inference threads.
Features zero-copy numpy-to-QImage rendering and hardware-accelerated bounding box drawing.

Usage:
    python -m facerec.gui_pyside [--gallery-dir data/gallery] [--device auto]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
    ACCENT_BLUE = "#3B82F6"      # primary accent
    ACCENT_GREEN = "#22C55E"     # known / success
    ACCENT_RED = "#EF4444"       # unknown / alert
    ACCENT_AMBER = "#F59E0B"     # warning / medium conf
    ACCENT_CYAN = "#06B6D4"      # info highlight

    FONT_FAMILY = "Helvetica Neue"
    FONT_MONO = "SF Mono"

def hex_to_qcolor(hex_str: str, alpha: int = 255) -> QColor:
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


# ---------------------------------------------------------------------------
# Common Data Structures
# ---------------------------------------------------------------------------
class TrackedFace:
    """Represents a face tracked across frames."""
    def __init__(self, bbox: np.ndarray):
        self.bbox: np.ndarray = bbox  # [x1, y1, x2, y2]
        self.name: str = "Unknown"
        self.score: float = 0.0
        self.tracker: cv2.Tracker | None = None
        self.frames_since_detect: int = 0
        self.needs_embedding: bool = True
        self.aligned_crop: np.ndarray | None = None

    @property
    def w(self) -> float: return self.bbox[2] - self.bbox[0]
    
    @property
    def h(self) -> float: return self.bbox[3] - self.bbox[1]


# ---------------------------------------------------------------------------
# Core Threads
# ---------------------------------------------------------------------------

class CameraThread(QThread):
    """Dedicated thread for pulling frames from the camera without blocking."""
    frame_ready = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self, camera_index: int = 0, target_fps: int = 60, target_width: int = 1280, target_height: int = 720):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self._is_running = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        
        # Request maximum hardware performance constraints to avoid software clipping
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not cap.isOpened():
            self.error_signal.emit("Failed to open camera.")
            return

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Emit raw frame immediately (no copying)
            # We convert to RGB here because PySide6 QImage prefers RGB888, 
            # and cv2 is BGR by default. This is the ONLY copy made.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame_rgb)
            
        cap.release()

    def stop(self) -> None:
        self._is_running = False
        self.wait()


class MLWorkerThread(QThread):
    """Dedicated thread for InsightFace detection and ArcFace recognition."""
    detections_ready = Signal(list)
    gallery_loaded = Signal(list)

    DETECT_EVERY_N_FRAMES = 3
    DETECT_SCALE = 0.5

    def __init__(self, gallery_dir: Path, device: str, similarity_threshold: float = 0.35):
        super().__init__()
        self.gallery_dir = gallery_dir
        self.device = device
        self.similarity_threshold = similarity_threshold
        self._is_running = True
        
        # 1-deep buffer ensures we only process the absolute newest frame
        self._latest_frame: np.ndarray | None = None
        self._frame_cond = QObject() # Simple synchronization

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        """Called by CameraThread (via Signal), storing the newest frame."""
        self._latest_frame = frame

    def run(self) -> None:
        # Load heavy models inside the thread to prevent UI lockup on startup
        detector = FaceDetector(device=self.device, det_thresh=0.5)
        recognizer = FaceRecognizer(device=self.device)
        
        gallery = GalleryDatabase(self.gallery_dir)
        if (self.gallery_dir / "gallery.json").exists():
            gallery.load()
        self.gallery_loaded.emit(gallery.list_identities())

        tracked_faces: list[TrackedFace] = []
        frame_idx = 0

        while self._is_running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.005)
                continue
                
            self._latest_frame = None # clear so we wait for new one
            frame_idx += 1
            
            # Since frame is RGB from CameraThread, we must convert to BGR 
            # for OpenCV trackers and InsightFace. We only downscale for detection.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            alive_faces = []
            for face in tracked_faces:
                face.frames_since_detect += 1
                if face.tracker is not None:
                    ok, bbox = face.tracker.update(frame_bgr)
                    if ok:
                        x, y, w, h = bbox
                        face.bbox = np.array([x, y, x+w, y+h])
                        if w > 20 and h > 20 and x >= -w and y >= -h:
                            alive_faces.append(face)
            
            tracked_faces = alive_faces

            if frame_idx % self.DETECT_EVERY_N_FRAMES == 0 or not tracked_faces:
                small_frame = cv2.resize(frame_bgr, (0, 0), fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)
                dets = detector.detect(small_frame)
                
                new_tracked_faces = []
                for det in dets:
                    box = det.bbox / self.DETECT_SCALE
                    
                    matched_face = None
                    best_iou = 0.3
                    for face in tracked_faces:
                        xA, yA = max(box[0], face.bbox[0]), max(box[1], face.bbox[1])
                        xB, yB = min(box[2], face.bbox[2]), min(box[3], face.bbox[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        iou = interArea / float(((box[2]-box[0])*(box[3]-box[1])) + (face.w*face.h) - interArea + 1e-5)
                        
                        if iou > best_iou:
                            best_iou = iou
                            matched_face = face
                            
                    if matched_face is not None:
                        matched_face.bbox = box
                        matched_face.frames_since_detect = 0
                        self._init_tracker(matched_face, frame_bgr)
                        new_tracked_faces.append(matched_face)
                        tracked_faces.remove(matched_face)
                    else:
                        new_face = TrackedFace(bbox=box)
                        lms = det.landmarks / self.DETECT_SCALE
                        # Align face based on BGR image
                        new_face.aligned_crop = align_face(frame_bgr, lms, output_size=112)
                        self._init_tracker(new_face, frame_bgr)
                        new_tracked_faces.append(new_face)
                        
                tracked_faces = new_tracked_faces
                
            results = []
            for face in tracked_faces:
                if face.needs_embedding and face.aligned_crop is not None:
                    emb = recognizer.get_embedding(face.aligned_crop)
                    name, score = gallery.query(emb, threshold=self.similarity_threshold)
                    face.name = name
                    face.score = score
                    face.needs_embedding = False
                    face.aligned_crop = None
                    
                results.append({
                    "bbox": face.bbox,
                    "name": face.name,
                    "score": face.score
                })

            self.detections_ready.emit(results)

    def _init_tracker(self, face: TrackedFace, frame_bgr: np.ndarray) -> None:
        try:
            face.tracker = cv2.TrackerKCF_create()
        except AttributeError:
            try: face.tracker = cv2.legacy.TrackerKCF_create()
            except AttributeError:
                face.tracker = None
                return
                
        box = face.bbox
        x, y, w, h = max(0, int(box[0])), max(0, int(box[1])), int(box[2]-box[0]), int(box[3]-box[1])
        if w > 0 and h > 0:
            face.tracker.init(frame_bgr, (x, y, w, h))

    # We allow UI to update the threshold interactively
    @Slot(float)
    def update_threshold(self, threshold: float) -> None:
        self.similarity_threshold = threshold

    def stop(self) -> None:
        self._is_running = False
        self.wait()


# ---------------------------------------------------------------------------
# Qt GUI MainWindow
# ---------------------------------------------------------------------------
class VideoOverlayWidget(QWidget):
    """A custom widget that paints the video frame and then vector graphics on top."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap: QPixmap | None = None
        self.current_detections: list[dict] = []
        
        # UI Font setup
        self.font_main = QFont(Theme.FONT_FAMILY, 14, QFont.Weight.Bold)
        self.font_small = QFont(Theme.FONT_FAMILY, 11, QFont.Weight.Normal)
        
        # Scaling tracker
        self._scale_factor_x = 1.0
        self._scale_factor_y = 1.0

    @Slot(np.ndarray)
    def set_frame(self, frame_rgb: np.ndarray) -> None:
        """Zero-copy conversion from NumPy uint8 RGB to QImage."""
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        # QImage created without copying memory buffer
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.update() # Triggers paintEvent

    @Slot(list)
    def set_detections(self, detections: list[dict]) -> None:
        self.current_detections = detections
        # We don't call self.update() here to save CPU cycles; 
        # the camera feed update (set_frame) happens 60x a second and will draw these.

    def paintEvent(self, event) -> None:
        if self.current_pixmap is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 1. Draw the Video Frame scaled to widget size
        rect = self.rect()
        painter.drawPixmap(rect, self.current_pixmap)
        
        # Calculate scaling coefficients if the video aspect ratio changed
        orig_w = self.current_pixmap.width()
        orig_h = self.current_pixmap.height()
        self._scale_factor_x = rect.width() / orig_w
        self._scale_factor_y = rect.height() / orig_h

        # 2. Draw the bounding boxes and text directly
        for det in self.current_detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            score = det["score"]
            is_known = name != "Unknown"
            
            # Scale coordinates to current window size
            sx1, sy1 = int(x1 * self._scale_factor_x), int(y1 * self._scale_factor_y)
            sx2, sy2 = int(x2 * self._scale_factor_x), int(y2 * self._scale_factor_y)
            sw, sh = sx2 - sx1, sy2 - sy1
            
            # Choose colors
            if is_known:
                base_color = hex_to_qcolor(Theme.ACCENT_AMBER if score < 0.5 else Theme.ACCENT_GREEN)
            else:
                base_color = hex_to_qcolor(Theme.ACCENT_RED)
                
            bg_color = hex_to_qcolor(Theme.BG_PRIMARY, 210) # 82% opacity overlay
            
            # --- Draw Corner Brackets ---
            pen = QPen(base_color)
            pen.setWidth(4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            
            cl = int(min(sw, sh) * 0.2) # corner length 20%
            
            # Top-Left
            painter.drawLine(sx1, sy1, sx1 + cl, sy1)
            painter.drawLine(sx1, sy1, sx1, sy1 + cl)
            # Top-Right
            painter.drawLine(sx2, sy1, sx2 - cl, sy1)
            painter.drawLine(sx2, sy1, sx2, sy1 + cl)
            # Bottom-Left
            painter.drawLine(sx1, sy2, sx1 + cl, sy2)
            painter.drawLine(sx1, sy2, sx1, sy2 - cl)
            # Bottom-Right
            painter.drawLine(sx2, sy2, sx2 - cl, sy2)
            painter.drawLine(sx2, sy2, sx2, sy2 - cl)
            
            # --- Draw Text Overlay Box ---
            label_text = name.replace('_', ' ')
            score_text = f"{score:.0%}"
            
            painter.setFont(self.font_main)
            label_fm = painter.fontMetrics()
            
            painter.setFont(self.font_small)
            score_fm = painter.fontMetrics()
            
            box_w = max(label_fm.horizontalAdvance(label_text), score_fm.horizontalAdvance(score_text)) + 24
            box_h = 56
            box_y = max(sy1 - box_h - 8, 0)
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(sx1, box_y, box_w, box_h, 6, 6)
            
            # --- Draw Text ---
            painter.setPen(hex_to_qcolor(Theme.TEXT_PRIMARY))
            painter.setFont(self.font_main)
            painter.drawText(sx1 + 12, box_y + 24, label_text)
            
            painter.setPen(base_color)
            painter.setFont(self.font_small)
            painter.drawText(sx1 + 12, box_y + 46, score_text)

        painter.end()


class MainWindow(QMainWindow):
    """Main application window bridging threads and UI elements."""
    def __init__(self, gallery_dir: Path, device: str = "auto", camera_index: int = 0):
        super().__init__()
        self.setWindowTitle("IE4228 · PySide6 Hardware-Accelerated Face Recognition")
        self.setStyleSheet(f"background-color: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY};")
        self.setMinimumSize(1000, 700) # Allow resizing!
        
        # FPS Tracking
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        
        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        
        # Left: Video Panel
        self.video_widget = VideoOverlayWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet(f"background-color: {Theme.BG_SECONDARY}; border-radius: 8px;")
        
        # Right: Sidebar (Fixed width)
        sidebar = self._build_sidebar()
        
        main_layout.addWidget(self.video_widget, stretch=3)
        main_layout.addLayout(sidebar, stretch=1)
        
        # Thread Setup
        config = Config() # Resolves specific device fallback
        if device == "auto":
            config.device = "auto"
            config.__post_init__() # forces device auto-resolution if auto
            resolved_device = config.device
        else:
            resolved_device = device
            
        self.status_hw.setText(f"Hardware: {resolved_device.upper()}")

        self.camera_thread = CameraThread(camera_index)
        self.ml_thread = MLWorkerThread(gallery_dir, device=resolved_device)
        
        # Connect Signals
        self.camera_thread.frame_ready.connect(self.video_widget.set_frame)
        self.camera_thread.frame_ready.connect(self.ml_thread.update_frame)
        self.camera_thread.frame_ready.connect(self._calculate_fps)
        
        self.ml_thread.detections_ready.connect(self.video_widget.set_detections)
        self.ml_thread.detections_ready.connect(self._update_stats)
        self.ml_thread.gallery_loaded.connect(self._update_gallery_list)
        
        self.slider.valueChanged.connect(lambda v: self.ml_thread.update_threshold(v / 100.0))
        
        # Start Threads
        self.ml_thread.start()
        self.camera_thread.start()

    def _build_sidebar(self) -> QVBoxLayout:
        sidebar = QVBoxLayout()
        sidebar.setSpacing(12)
        
        # Sidebar config
        panel_style = f"background-color: {Theme.BG_SECONDARY}; border-radius: 8px; padding: 12px;"
        
        # 1. Status Panel
        p_status = QFrame()
        p_status.setStyleSheet(panel_style)
        l_status = QVBoxLayout(p_status)
        
        self.status_lbl = QLabel("● SYSTEM ACTIVE")
        self.status_lbl.setStyleSheet(f"color: {Theme.ACCENT_GREEN}; font-weight: bold;")
        self.status_hw = QLabel("Hardware: INIT")
        self.status_hw.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.status_fps = QLabel("0.0 FPS")
        self.status_fps.setStyleSheet(f"font-family: {Theme.FONT_MONO}; color: {Theme.ACCENT_CYAN}; font-size: 18px; font-weight: bold;")
        
        l_status.addWidget(self.status_lbl)
        l_status.addWidget(self.status_hw)
        l_status.addWidget(self.status_fps)
        sidebar.addWidget(p_status)
        
        # 2. Controls Panel
        p_ctrl = QFrame()
        p_ctrl.setStyleSheet(panel_style)
        l_ctrl = QVBoxLayout(p_ctrl)
        
        l_ctrl.addWidget(QLabel("Similarity Threshold:"))
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(80)
        self.slider.setValue(35)
        self.slider_lbl = QLabel("0.35")
        self.slider_lbl.setStyleSheet(f"font-family: {Theme.FONT_MONO}; color: {Theme.ACCENT_BLUE};")
        
        # Update label immediately on drag
        self.slider.valueChanged.connect(lambda v: self.slider_lbl.setText(f"{v/100.0:.2f}"))
        
        row_sl = QHBoxLayout()
        row_sl.addWidget(self.slider)
        row_sl.addWidget(self.slider_lbl)
        l_ctrl.addLayout(row_sl)
        sidebar.addWidget(p_ctrl)
        
        # 3. Live Stats Panel
        p_stats = QFrame()
        p_stats.setStyleSheet(panel_style)
        l_stats = QVBoxLayout(p_stats)
        
        head = QLabel("LIFETIME STATS")
        head.setStyleSheet(f"color: {Theme.ACCENT_BLUE}; font-weight: bold; font-size: 11px;")
        l_stats.addWidget(head)
        
        self.lbl_faces = QLabel("Faces: 0")
        self.lbl_known = QLabel("Known: 0")
        self.lbl_unknown = QLabel("Unknown: 0")
        self.lbl_known.setStyleSheet(f"color: {Theme.ACCENT_GREEN};")
        self.lbl_unknown.setStyleSheet(f"color: {Theme.ACCENT_RED};")
        
        l_stats.addWidget(self.lbl_faces)
        l_stats.addWidget(self.lbl_known)
        l_stats.addWidget(self.lbl_unknown)
        sidebar.addWidget(p_stats)
        
        # 4. Gallery List
        p_gal = QFrame()
        p_gal.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        p_gal.setStyleSheet(panel_style)
        l_gal = QVBoxLayout(p_gal)
        l_gal.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        head_gal = QLabel("LOADED GALLERY")
        head_gal.setStyleSheet(f"color: {Theme.ACCENT_BLUE}; font-weight: bold; font-size: 11px;")
        l_gal.addWidget(head_gal)
        
        self.gallery_container = QVBoxLayout()
        self.gallery_container.setSpacing(4)
        l_gal.addLayout(self.gallery_container)
        
        sidebar.addWidget(p_gal)
        
        # Tracker Stats
        self._total_det = 0
        self._total_known = 0
        self._total_unknown = 0

        return sidebar

    @Slot()
    def _calculate_fps(self) -> None:
        current_time = time.perf_counter()
        elapsed = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        avg = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg if avg > 0 else 0
        self.status_fps.setText(f"{fps:.1f} FPS")

    @Slot(list)
    def _update_stats(self, detections: list[dict]) -> None:
        self._total_det += len(detections)
        known = sum(1 for d in detections if d["name"] != "Unknown")
        self._total_known += known
        self._total_unknown += (len(detections) - known)
        
        self.lbl_faces.setText(f"Faces: {self._total_det}")
        self.lbl_known.setText(f"Known: {self._total_known}")
        self.lbl_unknown.setText(f"Unknown: {self._total_unknown}")

    @Slot(list)
    def _update_gallery_list(self, names: list[str]) -> None:
        # Clear existing
        while self.gallery_container.count():
            item = self.gallery_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not names:
            lbl = QLabel("No identities found.")
            lbl.setStyleSheet(f"color: {Theme.TEXT_DIM};")
            self.gallery_container.addWidget(lbl)
            return
            
        for name in names:
            lbl = QLabel(f"● {name.replace('_', ' ')}")
            lbl.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
            self.gallery_container.addWidget(lbl)

    def closeEvent(self, event) -> None:
        """Safely clean up threads on exit."""
        self.camera_thread.stop()
        self.ml_thread.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Live face recognition PySide6 demo")
    parser.add_argument(
        "--gallery-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gallery",
        help="Path to gallery database directory",
    )
    # Changed default to auto, so Mac users automatically get mps
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    app = QApplication([])
    window = MainWindow(gallery_dir=args.gallery_dir, device=args.device, camera_index=args.camera)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
