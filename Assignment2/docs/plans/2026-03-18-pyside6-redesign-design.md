# Face Recognition PySide6 Redesign

## Objective
Migrate the existing Tkinter-based face recognition GUI to a modern, decoupled, and highly optimized PySide6 architecture. The new architecture must utilize the M-series GPU (`mps`) where available, run smoothly at max camera FPS (e.g., 60fps), and eliminate unnecessary memory copies of video frames.

## 1. Zero-Copy & Low-Copy Optimization Strategy

The current implementation copies numpy arrays 5-6 times per frame (resizing, `.copy()` into queues, `.copy()` for drawing overlays, `addWeighted` for transparency). We will drastically optimize this:

1. **Read-Only Frame References**: The ML Worker Thread will receive a *read-only reference* to the original numpy array straight from the Camera Thread. It will not copy it.
2. **PySide6 Native Overlays**: Instead of modifying the numpy array with `cv2.rectangle` and `cv2.putText` (which requires copying the frame multiple times for transparency), we will convert the clean frame straight to a `QImage` and display it. The bounding boxes and labels will be rendered *on top of the widget* using standard PySide6 `QPainter`. PySide6 uses GPU acceleration for vector drawing and transparencies natively.
3. **No Intermediate Resizing**: We will tell `cv2.VideoCapture` to request exactly `1280x720` (or `1920x1080` if preferred) directly from the hardware, avoiding software resizing.

## 2. Core Components (QThreads)

### Component A: `CameraThread` (QThread)
- **Role**: Dedicated solely to hardware I/O.
- **Loop**: `cap.read()`
- **Output**: Emits a `frame_ready(np.ndarray)` Signal.

### Component B: `MLWorkerThread` (QThread)
- **Role**: Runs PyTorch/ONNX inference async. Releases GIL.
- **Input**: Connected to `CameraThread.frame_ready()`. (Note: To prevent buffer backup if ML drops below camera FPS, we keep a size-1 variable in the ML thread that just grabs the *latest* emitted frame).
- **Execution**:
  - Validates `mps` (Apple Silicon GPU) backend availability via `torch.backends.mps.is_available()`.
  - Runs detection on a heavily downscaled version `cv2.resize(frame, (0,0), fx=0.3, fy=0.3)` just for detection.
  - Tracker runs on original frame.
  - Crops aligned faces, runs recognition.
- **Output**: Emits a `detections_ready(list[dict])` Signal containing only metadata (x, y, w, h, name, score).

### Component C: `MainWindow` (Main UI Thread)
- **Role**: Responsive UI rendering.
- **Receives `frame_ready`**: Quickly converts `np.ndarray` to `QImage` -> `QPixmap` -> `QLabel`.
- **Receives `detections_ready`**: Updates the internal list of bounding boxes.
- **PaintEvent**: When Qt repaints, it draws the video pixmap, and then iterates the bounding box list and draws sleek vector shapes (`QPainter`) with true drop shadows and native alpha channels directly over the video, at 60fps.

## 3. Dependency Management
- All dependencies will be installed into local `.venv`.
- `pyside6` must be added to `requirements.txt` or `pyproject.toml`.

## 4. Hardware Support
- We will patch `src/facerec/config.py` explicitly to map macOS silicon to the PyTorch `mps` device.

