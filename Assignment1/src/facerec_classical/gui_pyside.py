"""Live face recognition demo GUI using PySide6 (Classical Pipeline).

A real-time webcam feed with entirely decoupled Camera IO and ML inference threads.
Features zero-copy numpy-to-QImage rendering, KCF tracking, Haar/PCA/LDA inference,
and a Pipeline Visualizer tab showing all intermediate preprocessing stages.
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
    pipeline_stages_ready = Signal(dict)  # intermediate preprocessing visualizations

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
            stage_images = self._capture_pipeline_stages(frame_bgr, gray, config.target_size)
            
            for idx, face in enumerate(tracked_faces):
                if face.needs_recognition:
                    # Crop and recognize sequentially to avoid GIL contention
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    face_crop = gray[y1:y2, x1:x2]
                    
                    if face_crop.size > 0 and self.pipeline._recognizer.is_fitted:
                        face_aligned = self.aligner.align(face_crop, target_size=config.target_size)
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
            if stage_images:
                self.pipeline_stages_ready.emit(stage_images)

    def _capture_pipeline_stages(
        self, full_frame_bgr: np.ndarray, full_frame_gray: np.ndarray, target_size: tuple[int, int]
    ) -> dict[str, np.ndarray]:
        """Extract intermediate images for the pipeline visualiser using the full frame."""
        from facerec_classical.preprocessor import clahe, resize_face
        from skimage.feature import hog, local_binary_pattern
        import cv2

        stages: dict[str, np.ndarray] = {}

        # Resize for visualization to keep it fast
        resized_bgr = cv2.resize(full_frame_bgr, target_size)
        resized_gray = cv2.resize(full_frame_gray, target_size)

        # Stage 1: Raw Feed
        stages["1_raw_crop"] = resized_bgr

        # Stage 2: Grayscale
        stages["2_aligned"] = resized_gray

        # Stage 3: CLAHE
        clahe_img = clahe(resized_gray)
        stages["3_clahe"] = clahe_img.copy()

        # Stage 4: HOG gradient visualisation
        _, hog_image = hog(
            clahe_img,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
            feature_vector=True,
        )
        # Normalise HOG map to 0-255 uint8 for display
        hog_norm = (hog_image / (hog_image.max() + 1e-8) * 255).astype(np.uint8)
        stages["4_hog"] = hog_norm

        # Stage 5: LBP texture map
        lbp_map = local_binary_pattern(clahe_img, 8, 1, method="uniform")
        lbp_norm = (lbp_map / (lbp_map.max() + 1e-8) * 255).astype(np.uint8)
        stages["5_lbp"] = lbp_norm

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

    @Slot(float)
    def update_threshold(self, threshold: float) -> None:
        """Update SVM Probability threshold."""
        if hasattr(self, 'pipeline'):
            self.pipeline._recognizer._svm_prob_threshold = threshold

    @Slot(float)
    def update_recon_threshold(self, threshold: float) -> None:
        """Update PCA reconstruction error threshold."""
        if hasattr(self, 'pipeline'):
            self.pipeline._recognizer._recon_threshold = threshold



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
        "2_aligned": "② Grayscale",
        "3_clahe": "③ CLAHE",
        "4_hog": "④ HOG Gradients",
        "5_lbp": "⑤ LBP Texture",
    }
    STAGE_DESCRIPTIONS = {
        "1_raw_crop": "Full camera feed",
        "2_aligned": "Grayscale converted",
        "3_clahe": "Adaptive histogram equalisation",
        "4_hog": "Gradient orientation map (edges)",
        "5_lbp": "Local Binary Pattern (micro-texture)",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_labels: dict[str, QLabel] = {}
        self.stage_images: dict[str, QLabel] = {}
        self.stage_descs: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("PIPELINE VISUALISER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {Theme.ACCENT_ORANGE}; font-family: {Theme.FONT_HEADING};"
            f" font-size: 18px; font-weight: bold; padding: 8px;"
        )
        outer.addWidget(title)

        subtitle = QLabel("Real-time view of each classical preprocessing stage for the entire camera feed")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_BODY}; font-size: 12px; padding-bottom: 12px;"
        )
        outer.addWidget(subtitle)

        from PySide6.QtWidgets import QGridLayout
        # 3 items per row grid
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

            # Stage name
            lbl = QLabel(self.STAGE_LABELS[key])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {Theme.ACCENT_BLUE}; font-family: {Theme.FONT_HEADING};"
                f" font-size: 15px; font-weight: bold;"
            )
            card_layout.addWidget(lbl)
            self.stage_labels[key] = lbl

            # Image placeholder
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            img_lbl.setMinimumSize(240, 240)
            img_lbl.setStyleSheet(
                f"background-color: {Theme.BG_PRIMARY}; border-radius: 8px;"
            )
            card_layout.addWidget(img_lbl)
            self.stage_images[key] = img_lbl

            # Description
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

        # Waiting label shown when no face detected
        self._waiting_label = QLabel("Initializing pipeline visualizer…")
        self._waiting_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._waiting_label.setStyleSheet(
            f"color: {Theme.TEXT_DIM}; font-family: {Theme.FONT_BODY}; font-size: 14px;"
        )
        outer.addWidget(self._waiting_label)

    @Slot(dict)
    def update_stages(self, stages: dict[str, np.ndarray]) -> None:
        """Receive intermediate images and render them."""
        self._waiting_label.hide()
        for key, img in stages.items():
            if key not in self.stage_images:
                continue

            # Ensure uint8 grayscale
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Resize for display (uniform 240x240)
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
        self.setMinimumSize(1100, 750)
        
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        
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

        # Tab 1: Live video feed
        self.video_widget = VideoOverlayWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setStyleSheet(f"background-color: {Theme.BG_SECONDARY}; border-radius: 8px;")
        self.tab_widget.addTab(self.video_widget, "⬤  Live Feed")

        # Tab 2: Pipeline Visualizer
        self.pipeline_viz = PipelineVisualizerWidget()
        self.tab_widget.addTab(self.pipeline_viz, "⚙  Pipeline Visualiser")
        
        sidebar = self._build_sidebar()
        
        main_layout.addWidget(self.tab_widget, stretch=3)
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
        self.ml_thread.pipeline_stages_ready.connect(self.pipeline_viz.update_stages)
        
        # Sliders mapped to two-stage triage thresholds
        self.slider.valueChanged.connect(lambda v: self.ml_thread.update_threshold(float(v) / 100.0))
        self.recon_slider.valueChanged.connect(lambda v: self.ml_thread.update_recon_threshold(float(v)))
        
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
        
        lbl_recon = QLabel("Recon Error Threshold:")
        lbl_recon.setStyleSheet(f"font-family: {Theme.FONT_BODY}; color: {Theme.TEXT_PRIMARY};")
        l_ctrl.addWidget(lbl_recon)
        
        self.recon_slider = QSlider(Qt.Orientation.Horizontal)
        self.recon_slider.setMinimum(100)
        self.recon_slider.setMaximum(50000)
        self.recon_slider.setValue(5000)
        self.recon_slider_lbl = QLabel("5000")
        self.recon_slider_lbl.setMinimumWidth(45)
        self.recon_slider_lbl.setStyleSheet(f"font-family: {Theme.FONT_MONO}; color: {Theme.ACCENT_BLUE};")
        
        self.recon_slider.valueChanged.connect(lambda v: self.recon_slider_lbl.setText(f"{v}"))
        
        row_recon = QHBoxLayout()
        row_recon.addWidget(self.recon_slider)
        row_recon.addWidget(self.recon_slider_lbl)
        l_ctrl.addLayout(row_recon)
        
        lbl_mahal = QLabel("SVM Prob Threshold:")
        lbl_mahal.setStyleSheet(f"font-family: {Theme.FONT_BODY}; color: {Theme.TEXT_PRIMARY};")
        l_ctrl.addWidget(lbl_mahal)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(60)
        self.slider_lbl = QLabel("0.60")
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
