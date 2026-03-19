"""Live face recognition demo GUI using PySide6 (Classical Pipeline).

A real-time webcam feed with entirely decoupled Camera IO and ML inference threads.
Features zero-copy numpy-to-QImage rendering, KCF tracking, and Haar/PCA/LDA inference.
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

from facerec_classical.config import Config
from facerec_classical.detector import FaceAligner
from facerec_classical.pipeline import ClassicalFaceRecPipeline
from facerec_classical.preprocessor import preprocess_face


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
        self.name: str = "Unknown"
        self.score: float = 0.0 # Will store SED distance
        self.tracker: cv2.Tracker | None = None
        self.frames_since_detect: int = 0
        self.needs_recognition: bool = True

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
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame_rgb)
            
        cap.release()

    def stop(self) -> None:
        self._is_running = False
        self.wait()


class ClassicalMLWorkerThread(QThread):
    """Dedicated thread for Haar detection and PCA/LDA recognition."""
    detections_ready = Signal(list)
    gallery_loaded = Signal(list)

    DETECT_EVERY_N_FRAMES = 5
    DETECT_SCALE = 0.5

    def __init__(self, gallery_dir: Path, sed_threshold: float = 0.45):
        super().__init__()
        self.gallery_dir = gallery_dir
        self.sed_threshold = sed_threshold
        self._is_running = True
        self._latest_frame: np.ndarray | None = None

    @Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        self._latest_frame = frame

    def run(self) -> None:
        config = Config()
        config.sed_threshold = self.sed_threshold
        self.pipeline = ClassicalFaceRecPipeline(config)
        self.aligner = FaceAligner()
        
        try:
            print(f"Training PCA/LDA on {self.gallery_dir}...")
            self.pipeline.train(str(self.gallery_dir))
            print("Training complete.")
        except Exception as e:
            print(f"Warning: Training failed. {e}")

        # Find identities manually since A1 doesn't have a structured Gallery dict
        identities = []
        if self.gallery_dir.exists():
            identities = [p.name for p in self.gallery_dir.iterdir() if p.is_dir()]
        self.gallery_loaded.emit(identities)

        tracked_faces: list[TrackedFace] = []
        frame_idx = 0

        while self._is_running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.005)
                continue
                
            self._latest_frame = None
            frame_idx += 1
            
            # Frame comes in as RGB. Convert to BGR for trackers, GRAY for Haar.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            alive_faces = []
            for face in tracked_faces:
                face.frames_since_detect += 1
                if face.tracker is not None:
                    ok, bbox = face.tracker.update(frame_bgr)
                    if ok:
                        x, y, w, h = bbox
                        face.bbox = np.array([x, y, x+w, y+h])
                        # boundary protections
                        H, W = frame.shape[:2]
                        face.bbox[0] = max(0, face.bbox[0])
                        face.bbox[1] = max(0, face.bbox[1])
                        face.bbox[2] = min(W, face.bbox[2])
                        face.bbox[3] = min(H, face.bbox[3])
                        
                        if w > 20 and h > 20:
                            alive_faces.append(face)
            
            tracked_faces = alive_faces

            if frame_idx % self.DETECT_EVERY_N_FRAMES == 0 or not tracked_faces:
                small_gray = cv2.resize(gray, (0, 0), fx=self.DETECT_SCALE, fy=self.DETECT_SCALE)
                dets = self.pipeline._detector.detect(small_gray)
                
                new_tracked_faces = []
                for det in dets:
                    # scale box back up
                    x, y, w, h = det.bbox
                    x = x / self.DETECT_SCALE
                    y = y / self.DETECT_SCALE
                    w = w / self.DETECT_SCALE
                    h = h / self.DETECT_SCALE
                    box = np.array([x, y, x+w, y+h])
                    
                    matched_face = None
                    best_iou = 0.45
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
                        self._init_tracker(new_face, frame_bgr)
                        new_tracked_faces.append(new_face)
                        
                tracked_faces = new_tracked_faces
                
            results = []
            for face in tracked_faces:
                if face.needs_recognition:
                    # Crop and recognize sequentially to avoid GIL contention
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    face_crop = gray[y1:y2, x1:x2]
                    
                    if face_crop.size > 0 and self.pipeline._recognizer.is_fitted:
                        face_aligned = self.aligner.align(face_crop)
                        face_pre = preprocess_face(face_aligned, target_size=config.target_size)
                        face_vector = face_pre.flatten()
                        name, dist = self.pipeline._recognizer.predict(face_vector)
                        
                        face.name = name
                        face.score = dist
                    
                    face.needs_recognition = False
                    
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

    @Slot(float)
    def update_threshold(self, threshold: float) -> None:
        self.sed_threshold = threshold
        if hasattr(self, 'pipeline'):
            self.pipeline._recognizer._sed_threshold = threshold

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
        self.font_main = QFont(Theme.FONT_HEADING, 14, QFont.Weight.Bold)
        self.font_small = QFont(Theme.FONT_BODY, 11, QFont.Weight.Normal)
        
    @Slot(np.ndarray)
    def set_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.update()

    @Slot(list)
    def set_detections(self, detections: list[dict]) -> None:
        self.current_detections = detections

    def paintEvent(self, event) -> None:
        if self.current_pixmap is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        rect = self.rect()
        scaled_pixmap = self.current_pixmap.scaled(
            rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        x_offset = (rect.width() - scaled_pixmap.width()) // 2
        y_offset = (rect.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        orig_w = self.current_pixmap.width()
        orig_h = self.current_pixmap.height()
        
        scale_x = scaled_pixmap.width() / orig_w
        scale_y = scaled_pixmap.height() / orig_h

        # 2. Draw the bounding boxes and text directly
        for det in self.current_detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            score = det["score"]  # This is the SED Distance
            is_known = name != "Unknown"
            
            sx1 = int(x1 * scale_x) + x_offset
            sy1 = int(y1 * scale_y) + y_offset
            sx2 = int(x2 * scale_x) + x_offset
            sy2 = int(y2 * scale_y) + y_offset
            sw, sh = sx2 - sx1, sy2 - sy1
            
            if is_known:
                base_color = hex_to_qcolor(Theme.ACCENT_GREEN)
            else:
                base_color = hex_to_qcolor(Theme.ACCENT_BLUE)
                
            bg_color = hex_to_qcolor(Theme.BG_PRIMARY, 210) 
            
            pen = QPen(base_color)
            pen.setWidth(4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            
            cl = int(min(sw, sh) * 0.2)
            
            painter.drawLine(sx1, sy1, sx1 + cl, sy1)
            painter.drawLine(sx1, sy1, sx1, sy1 + cl)
            painter.drawLine(sx2, sy1, sx2 - cl, sy1)
            painter.drawLine(sx2, sy1, sx2, sy1 + cl)
            painter.drawLine(sx1, sy2, sx1 + cl, sy2)
            painter.drawLine(sx1, sy2, sx1, sy2 - cl)
            painter.drawLine(sx2, sy2, sx2 - cl, sy2)
            painter.drawLine(sx2, sy2, sx2, sy2 - cl)
            
            label_text = name.replace('_', ' ')
            score_text = f"Dist: {score:.2f}" if score > 0 else "Dist: --"
            
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
            
            painter.setPen(hex_to_qcolor(Theme.TEXT_PRIMARY))
            painter.setFont(self.font_main)
            painter.drawText(sx1 + 12, box_y + 24, label_text)
            
            painter.setPen(base_color)
            painter.setFont(self.font_small)
            painter.drawText(sx1 + 12, box_y + 46, score_text)

        painter.end()


class MainWindow(QMainWindow):
    """Main application window bridging threads and UI elements."""
    def __init__(self, gallery_dir: Path, camera_index: int = 0):
        super().__init__()
        self.setWindowTitle("IE4228 · PySide6 Classical Face Recognition")
        self.setStyleSheet(f"background-color: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY};")
        self.setMinimumSize(1000, 700)
        
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)
        
        self.video_widget = VideoOverlayWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet(f"background-color: {Theme.BG_SECONDARY}; border-radius: 8px;")
        
        sidebar = self._build_sidebar()
        
        main_layout.addWidget(self.video_widget, stretch=3)
        main_layout.addLayout(sidebar, stretch=1)
        
        self.status_hw.setText("Hardware: CPU (Classical)")

        self.camera_thread = CameraThread(camera_index)
        self.ml_thread = ClassicalMLWorkerThread(gallery_dir)
        
        self.camera_thread.frame_ready.connect(self.video_widget.set_frame)
        self.camera_thread.frame_ready.connect(self.ml_thread.update_frame)
        self.camera_thread.frame_ready.connect(self._calculate_fps)
        
        self.ml_thread.detections_ready.connect(self.video_widget.set_detections)
        self.ml_thread.detections_ready.connect(self._update_stats)
        self.ml_thread.gallery_loaded.connect(self._update_gallery_list)
        
        # Slider mapped to Cosine Distance Threshold (0.10 -> 1.0)
        self.slider.valueChanged.connect(lambda v: self.ml_thread.update_threshold(float(v) / 100.0))
        
        self.ml_thread.start()
        self.camera_thread.start()

    def _build_sidebar(self) -> QVBoxLayout:
        sidebar = QVBoxLayout()
        sidebar.setSpacing(16)
        
        panel_style = f"QFrame {{ background-color: {Theme.BG_SECONDARY}; border-radius: 12px; }}"
        
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
        
        # Controls Panel
        p_ctrl = QFrame()
        p_ctrl.setStyleSheet(panel_style)
        l_ctrl = QVBoxLayout(p_ctrl)
        l_ctrl.setContentsMargins(16, 16, 16, 16)
        l_ctrl.setSpacing(12)
        
        lbl_control = QLabel("Cosine Distance:")
        lbl_control.setStyleSheet(f"font-family: {Theme.FONT_BODY}; color: {Theme.TEXT_PRIMARY};")
        l_ctrl.addWidget(lbl_control)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(10)   # 0.10
        self.slider.setMaximum(100)  # 1.00
        self.slider.setValue(45)     # default 0.45
        self.slider_lbl = QLabel("0.45")
        self.slider_lbl.setMinimumWidth(35)
        self.slider_lbl.setStyleSheet(f"font-family: {Theme.FONT_MONO}; color: {Theme.ACCENT_BLUE};")
        
        self.slider.valueChanged.connect(lambda v: self.slider_lbl.setText(f"{v / 100.0:.2f}"))
        
        row_sl = QHBoxLayout()
        row_sl.addWidget(self.slider)
        row_sl.addWidget(self.slider_lbl)
        l_ctrl.addLayout(row_sl)
        sidebar.addWidget(p_ctrl)
        
        # Live Stats Panel
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
        
        # Gallery List
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
        
        sidebar.addWidget(p_gal)

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
        self.camera_thread.stop()
        self.ml_thread.stop()
        super().closeEvent(event)
