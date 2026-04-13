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
    QPushButton,    
    QInputDialog,   
)

from facerec.alignment import align_face
from facerec.config import Config
from facerec.database import GalleryDatabase
from facerec.detector import FaceDetector
from facerec.recognizer import FaceRecognizer
from facerec.pipeline import FaceRecognitionPipeline


# ---------------------------------------------------------------------------
# Design tokens  (Anthropic Brand Styling)
# ---------------------------------------------------------------------------
class Theme:
    BG_PRIMARY = "#141413"       # Dark Primary
    BG_SECONDARY = "#2a2a29"     # Slightly lighter panel background (custom)
    BG_CARD = "#b0aea5"          # Mid Gray
    BORDER = "#334155"           # Subdued border
    TEXT_PRIMARY = "#faf9f5"     # Light Text
    TEXT_SECONDARY = "#b0aea5"   # Mid Gray
    TEXT_DIM = "#8c8c88"         # custom muted
    ACCENT_ORANGE = "#d97757"    # Anthropic Primary Accent
    ACCENT_BLUE = "#6a9bcc"      # Anthropic Secondary Accent
    ACCENT_GREEN = "#788c5d"     # Anthropic Tertiary Accent

    FONT_HEADING = "Poppins"
    FONT_BODY = "Lora"
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
        self.smooth_bbox: np.ndarray = bbox.copy()
        self.name: str = "Unknown"
        self.score: float = 0.0
        self.tracker: cv2.Tracker | None = None
        self.frames_since_detect: int = 0
        self.needs_embedding: bool = True
        self.aligned_crop: np.ndarray | None = None
        
        # Caches for visualization
        self.last_aligned_crop: np.ndarray | None = None
        self.last_embedding: np.ndarray | None = None

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

    def __init__(self, camera_index: int = 0, target_fps: int = 60, target_width: int = 1920, target_height: int = 1080):
        super().__init__()
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self._is_running = True

    def run(self) -> None:
        # Try AVFoundation backend first for max hardware performance on macOS
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent frame buffering latency

        # If AVFoundation fails or resolution is weirdly low, fallback to default backend and Mac limits
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_WIDTH) < self.target_width:
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
            # Fallback to MacBook default conservative settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

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
    pipeline_stages_ready = Signal(dict)

    DETECT_EVERY_N_FRAMES = 6  # Scaled up for higher FPS target
    DETECT_SCALE = 0.5

    def __init__(self, gallery_dir: Path, device: str, similarity_threshold: float = 0.35):
        super().__init__()
        self.gallery_dir = gallery_dir
        self.device = device
        self.similarity_threshold = similarity_threshold
        self._is_running = True
        self._pending_reg_name: str | None = None
        self._last_dets = []
        
        # 1-deep buffer ensures we only process the absolute newest frame
        self._latest_frame: np.ndarray | None = None
        self._frame_cond = QObject() # Simple synchronization

    @Slot(str)
    def request_registration(self, name: str) -> None:
        """Triggered by UI to register the largest face in the next frame."""
        self._pending_reg_name = name

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

        config = Config()
        config.device = self.device
        pipeline = FaceRecognitionPipeline(config=config, gallery=gallery)

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
            # for OpenCV trackers and InsightFace.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            small_frame = cv2.resize(frame_bgr, (0, 0), fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)

            alive_faces = []
            for face in tracked_faces:
                face.frames_since_detect += 1
                if face.tracker is not None:
                    ok, bbox = face.tracker.update(frame_bgr)
                    if ok:
                        x, y, w, h = bbox
                        target_box = np.array([x, y, x+w, y+h])
                        face.bbox = target_box
                        
                        # Apply EMA smoothing (70% old, 30% new)
                        face.smooth_bbox = (0.7 * face.smooth_bbox) + (0.3 * target_box)
                        
                        if w > 20 and h > 20 and x >= -w and y >= -h:
                            alive_faces.append(face)
            
            tracked_faces = alive_faces

            if frame_idx % self.DETECT_EVERY_N_FRAMES == 0 or not tracked_faces:
                dets = detector.detect(small_frame)
                self._last_dets = dets

                if self._pending_reg_name and dets:
                    pipeline.register_new_identity(
                        frame_bgr, dets[0], self._pending_reg_name, scale=self.DETECT_SCALE
                    )
                    self.gallery_loaded.emit(gallery.list_identities())
                    self._pending_reg_name = None
                    tracked_faces.clear()
                    continue
                
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
                        matched_face.smooth_bbox = (0.7 * matched_face.smooth_bbox) + (0.3 * box)
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
                    face.last_aligned_crop = face.aligned_crop.copy()
                    emb = recognizer.get_embedding(face.aligned_crop)
                    face.last_embedding = emb.copy()
                    name, score = gallery.query(emb, threshold=self.similarity_threshold)
                    face.name = name
                    face.score = score
                    face.needs_embedding = False
                    face.aligned_crop = None
                    
                results.append({
                    "bbox": face.smooth_bbox,
                    "name": face.name,
                    "score": face.score
                })

            stages = self._capture_pipeline_stages(frame_bgr, small_frame, tracked_faces)
            self.pipeline_stages_ready.emit(stages)

            self.detections_ready.emit(results)

    def _capture_pipeline_stages(self, frame_bgr: np.ndarray, small_frame: np.ndarray, tracked_faces: list[TrackedFace]) -> dict:
        stages = {}
        # 1. Raw
        stages["1_raw_crop"] = frame_bgr
        
        # 2. Resized
        stages["2_rescaled"] = small_frame
        
        # 3. Detection (Draw on a copy of small frame)
        det_img = small_frame.copy()
        for det in self._last_dets:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lx, ly in det.landmarks.astype(int):
                cv2.circle(det_img, (lx, ly), 2, (0, 0, 255), -1)
        stages["3_detection"] = det_img
        
        # 4 & 5. Find the largest tracked face that has cached visualizer data
        largest_face = None
        max_area = 0
        for f in tracked_faces:
            if f.last_aligned_crop is not None and f.last_embedding is not None:
                area = f.w * f.h
                if area > max_area:
                    max_area = area
                    largest_face = f
        
        if largest_face:
            stages["4_alignment"] = largest_face.last_aligned_crop
            emb = largest_face.last_embedding
            emb_norm = cv2.normalize(emb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Reshape 512 into 16x32 for barcode/heatmap look
            emb_2d = emb_norm.reshape(16, 32)
            emb_color = cv2.applyColorMap(emb_2d, cv2.COLORMAP_INFERNO)
            stages["5_embedding"] = emb_color
        else:
            stages["4_alignment"] = np.zeros((112, 112, 3), dtype=np.uint8)
            emb_placeholder = np.zeros((16, 32), dtype=np.uint8)
            stages["5_embedding"] = cv2.applyColorMap(emb_placeholder, cv2.COLORMAP_INFERNO)

        return stages

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
# Pipeline Visualizer Widget
# ---------------------------------------------------------------------------
class PipelineVisualizerWidget(QWidget):
    """Displays all intermediate preprocessing stages side-by-side."""

    STAGE_LABELS = {
        "1_raw_crop": "① Raw Feed",
        "2_rescaled": "② Rescaled Feed",
        "3_detection": "③ Detection Model",
        "4_alignment": "④ Alignment Layer",
        "5_embedding": "⑤ ArcFace Embedding",
    }
    STAGE_DESCRIPTIONS = {
        "1_raw_crop": "Full camera feed",
        "2_rescaled": "Downscaled for detector",
        "3_detection": "BBox & 5-point landmarks",
        "4_alignment": "112x112 similarity transform map",
        "5_embedding": "512-dim feature magnitude heatmap",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_labels: dict[str, QLabel] = {}
        self.stage_images: dict[str, QLabel] = {}
        self.stage_descs: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("DL PIPELINE VISUALISER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {Theme.ACCENT_ORANGE}; font-family: {Theme.FONT_HEADING};"
            f" font-size: 18px; font-weight: bold; padding: 8px;"
        )
        outer.addWidget(title)

        subtitle = QLabel("Real-time view of each deep learning pipeline stage")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_BODY}; font-size: 12px; padding-bottom: 12px;"
        )
        outer.addWidget(subtitle)

        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(16)

        col = 0
        row_idx = 0
        for key in self.STAGE_LABELS:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background-color: {Theme.BG_SECONDARY}; border-radius: 10px; }}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(8)

            lbl = QLabel(self.STAGE_LABELS[key])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {Theme.ACCENT_BLUE}; font-family: {Theme.FONT_HEADING};"
                f" font-size: 15px; font-weight: bold;"
            )
            card_layout.addWidget(lbl)
            self.stage_labels[key] = lbl

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            img_lbl.setMinimumSize(240, 240)
            img_lbl.setStyleSheet(
                f"background-color: {Theme.BG_PRIMARY}; border-radius: 8px;"
            )
            card_layout.addWidget(img_lbl)
            self.stage_images[key] = img_lbl

            desc = QLabel(self.STAGE_DESCRIPTIONS[key])
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc.setWordWrap(True)
            desc.setStyleSheet(
                f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_BODY}; font-size: 12px;"
            )
            card_layout.addWidget(desc)
            self.stage_descs[key] = desc

            grid.addWidget(card, row_idx, col)
            
            col += 1
            if col > 2:
                col = 0
                row_idx += 1

        outer.addLayout(grid)
        outer.addStretch()

        self._waiting_label = QLabel("Initializing pipeline visualizer…")
        self._waiting_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._waiting_label.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_BODY}; font-size: 14px;"
        )
        outer.addWidget(self._waiting_label)

    @Slot(dict)
    def update_stages(self, stages: dict[str, np.ndarray]) -> None:
        self._waiting_label.hide()
        for key, img in stages.items():
            if key not in self.stage_images:
                continue

            # Ensure uint8 image
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            display = cv2.resize(img, (240, 240), interpolation=cv2.INTER_NEAREST)

            h, w = display.shape[:2]
            if display.ndim == 2:
                qimg = QImage(display.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                qimg = QImage(display.data, w, h, w * 3, QImage.Format.Format_BGR888)

            self.stage_images[key].setPixmap(QPixmap.fromImage(qimg))


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
        self.font_main = QFont(Theme.FONT_HEADING, 14, QFont.Weight.Bold)
        self.font_small = QFont(Theme.FONT_BODY, 11, QFont.Weight.Normal)
        
        # We need the drawn bounding box coordinates inside the widget relative to the painted video format
        self._painted_rect_w = 1.0
        self._painted_rect_h = 1.0
        self._video_x_offset = 0
        self._video_y_offset = 0

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
        
        # 1. Draw the Video Frame scaled to preserve aspect ratio
        rect = self.rect()
        scaled_pixmap = self.current_pixmap.scaled(
            rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate offsets to center the video if there's letterboxing
        x_offset = (rect.width() - scaled_pixmap.width()) // 2
        y_offset = (rect.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # Save exact dimensions of the painted video area for scaling coordinates safely
        orig_w = self.current_pixmap.width()
        orig_h = self.current_pixmap.height()
        
        scale_x = scaled_pixmap.width() / orig_w
        scale_y = scaled_pixmap.height() / orig_h

        # 2. Draw the bounding boxes and text directly
        for det in self.current_detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            score = det["score"]
            is_known = name != "Unknown"
            
            # Scale coordinates and add the letterbox offset
            sx1 = int(x1 * scale_x) + x_offset
            sy1 = int(y1 * scale_y) + y_offset
            sx2 = int(x2 * scale_x) + x_offset
            sy2 = int(y2 * scale_y) + y_offset
            sw, sh = sx2 - sx1, sy2 - sy1
            
            # Choose colors (Anthropic palette)
            if is_known:
                base_color = hex_to_qcolor(Theme.ACCENT_ORANGE if score < 0.5 else Theme.ACCENT_GREEN)
            else:
                base_color = hex_to_qcolor(Theme.ACCENT_BLUE)
                
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
            box_y = max(sy1 - box_h - 8, y_offset)
            
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
        
        # --- Tab widget: Live Feed + Pipeline Visualizer ---
        from PySide6.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                background-color: {Theme.BG_PRIMARY};
            }}
            QTabBar::tab {{
                background-color: {Theme.BG_SECONDARY};
                color: {Theme.TEXT_SECONDARY};
                font-family: {Theme.FONT_HEADING};
                font-weight: bold;
                padding: 10px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QTabBar::tab:selected {{
                background-color: {Theme.ACCENT_ORANGE};
                color: {Theme.TEXT_PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #3a3a39;
            }}
        """)

        # Left: Video Panel / Live Feed
        self.video_widget = VideoOverlayWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet(f"background-color: {Theme.BG_SECONDARY}; border-radius: 8px;")
        self.tab_widget.addTab(self.video_widget, "⬤  Live Feed")
        
        # Tab 2: Pipeline Visualizer
        self.pipeline_viz = PipelineVisualizerWidget()
        self.tab_widget.addTab(self.pipeline_viz, "⚙  Pipeline Visualiser")
        
        # Right: Sidebar (Fixed width)
        sidebar = self._build_sidebar()
        
        main_layout.addWidget(self.tab_widget, stretch=3)
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
        self.ml_thread.pipeline_stages_ready.connect(self.pipeline_viz.update_stages)
        
        self.slider.valueChanged.connect(lambda v: self.ml_thread.update_threshold(v / 100.0))
        
        # Start Threads
        self.ml_thread.start()
        self.camera_thread.start()

    def _build_sidebar(self) -> QVBoxLayout:
        sidebar = QVBoxLayout()
        sidebar.setSpacing(16)
        
        # Use QFrame selector so children (QLabel) don't inherit the background or padding
        panel_style = f"QFrame {{ background-color: {Theme.BG_SECONDARY}; border-radius: 12px; }}"
        
        # 1. Status Panel
        p_status = QFrame()
        p_status.setStyleSheet(panel_style)
        l_status = QVBoxLayout(p_status)
        l_status.setContentsMargins(16, 16, 16, 16)
        l_status.setSpacing(8)
        
        self.status_lbl = QLabel("● SYSTEM ACTIVE")
        self.status_lbl.setStyleSheet(f"color: {Theme.ACCENT_GREEN}; font-family: {Theme.FONT_HEADING}; font-weight: bold;")
        self.status_hw = QLabel("Hardware: INIT")
        self.status_hw.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-family: {Theme.FONT_BODY};")
        self.status_fps = QLabel("0.0 FPS")
        self.status_fps.setStyleSheet(f"font-family: {Theme.FONT_MONO}; color: {Theme.ACCENT_BLUE}; font-size: 24px; font-weight: bold;")
        
        l_status.addWidget(self.status_lbl)
        l_status.addWidget(self.status_hw)
        l_status.addWidget(self.status_fps)
        sidebar.addWidget(p_status)
        
        # 2. Controls Panel
        p_ctrl = QFrame()
        p_ctrl.setStyleSheet(panel_style)
        l_ctrl = QVBoxLayout(p_ctrl)
        l_ctrl.setContentsMargins(16, 16, 16, 16)
        l_ctrl.setSpacing(12)
        
        lbl_control = QLabel("Similarity Threshold:")
        lbl_control.setStyleSheet(f"font-family: {Theme.FONT_BODY}; color: {Theme.TEXT_PRIMARY};")
        l_ctrl.addWidget(lbl_control)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(10)
        self.slider.setMaximum(80)
        self.slider.setValue(35)
        self.slider_lbl = QLabel("0.35")
        self.slider_lbl.setMinimumWidth(35)
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
        l_stats.setContentsMargins(16, 16, 16, 16)
        l_stats.setSpacing(8)
        
        head = QLabel("CURRENT FRAME STATS")
        head.setStyleSheet(f"color: {Theme.ACCENT_ORANGE}; font-family: {Theme.FONT_HEADING}; font-weight: bold; font-size: 12px;")
        l_stats.addWidget(head)
        
        stat_style = f"font-family: {Theme.FONT_BODY}; font-size: 13px;"
        
        self.lbl_faces = QLabel("Total Faces: 0")
        self.lbl_faces.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; {stat_style}")
        self.lbl_known = QLabel("Known Matches: 0")
        self.lbl_known.setStyleSheet(f"color: {Theme.ACCENT_GREEN}; {stat_style}")
        self.lbl_unknown = QLabel("Unknown Faces: 0")
        self.lbl_unknown.setStyleSheet(f"color: {Theme.ACCENT_BLUE}; {stat_style}")
        
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
        l_gal.setContentsMargins(16, 16, 16, 16)
        l_gal.setSpacing(12)
        
        head_gal = QLabel("LOADED GALLERY")
        head_gal.setStyleSheet(f"color: {Theme.ACCENT_ORANGE}; font-family: {Theme.FONT_HEADING}; font-weight: bold; font-size: 12px;")
        l_gal.addWidget(head_gal)
        
        self.gallery_container = QVBoxLayout()
        self.gallery_container.setSpacing(6)
        l_gal.addLayout(self.gallery_container)

        self.btn_register = QPushButton("Register Unknown Face")
        self.btn_register.setStyleSheet(f"""
            QPushButton {{ background-color: {Theme.ACCENT_BLUE}; color: {Theme.BG_PRIMARY}; 
            border-radius: 6px; padding: 10px; font-weight: bold; }}
        """)
        self.btn_register.clicked.connect(self._on_register_click)
        l_gal.addWidget(self.btn_register)

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
        total = len(detections)
        known = sum(1 for d in detections if d["name"] != "Unknown")
        unknown = total - known
        
        self.lbl_faces.setText(f"Faces in View: {total}")
        self.lbl_known.setText(f"Known Matches: {known}")
        self.lbl_unknown.setText(f"Unknown Faces: {unknown}")

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

    @Slot()
    def _on_register_click(self) -> None:
        name, ok = QInputDialog.getText(self, "Register Face", "Enter name (e.g., John_Doe):")
        if ok and name.strip():
            self.ml_thread.request_registration(name.strip().replace(" ", "_"))

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
