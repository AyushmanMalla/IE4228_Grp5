# IE4228 Assignment 1: Classical Face Detection and Recognition System
**Draft Report**

## 1. Introduction
This report outlines the methodology, data pipeline, architectural design choices, and evaluation of the human face recognition system developed for Assignment 1. The system detects and recognizes faces in a live camera feed using classical machine learning techniques, explicitly avoiding deep learning models for feature extraction and recognition, in adherence to the assignment constraints. The core of the system is built upon a robust Histogram of Oriented Gradients (HOG) detector combined with a PCA-LDA (Eigenfaces to Fisherfaces) dimension reduction cascade and a Two-Stage triage multi-class Support Vector Machine (SVM) pipeline for open-set recognition.

---

## 2. In-Depth Pre-processing & Inference Pipeline
The data path is highly sequential, transitioning from high-definition webcam matrices through sequential geometric transformations to a flattened mathematical feature vector.

1. **Hardware Ingestion:** The `cv2.VideoCapture` hooks the physical webcam, configuring hardware to pull frames at `1280x720` resolution and `60 FPS`. OpenCV natively reads physical sensor data as a 3-channel `BGR` matrix.
2. **Base Color Conversions (`CameraThread` & `ClassicalMLWorkerThread`):** The raw BGR frame is immediately isolated and cast to `RGB` for GUI UI rendering. Simultaneously, the ML processing thread clones the exact frame and casts it to single-channel `Grayscale` via `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)`. **All facial detection, alignment, and texture extractions physically operate *only* on the 2D Grayscale matrix.**
3. **Downscaled HOG Boundary Detection (`pipeline._detector.detect`):** To preserve CPU cycles, the grayscale matrix is temporarily reduced by 50% (`DETECT_SCALE = 0.5`). The `DlibHOGFaceDetector` structurally analyzes the gradient orientations of 16x16 pixel grids across this compressed image. An underlying Linear SVM maps those gradients to human face structures, drawing native coordinates that are scaled back up to map directly to the `1280x720` original matrix.
4. **Physical Cropping:** An array slice `face_crop = gray[y1:y2, x1:x2]` isolates just the pixels of the human face from the massive 720p 2D array matrix.
5. **Landmark Affine Alignment (`FaceAligner.align`):** The cropped array is passed to Dlib's `shape_predictor_68_face_landmarks`, physically locating the mathematical center of both eyes. The pipeline uses `cv2.estimateAffinePartial2D` to compute and apply a similarity transform matrix (rotation, uniform scale, translation). This geometrically morphs the array so the left and right eyes are rigidly pinned to exact coordinates `(30, 35)` and `(70, 35)` over a theoretical `100x100` canvas.
6. **Feature Preprocessing (`preprocess_face`):** The rigidly aligned grayscale crop is passed through Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE().apply(gray)`), which normalizes extreme local lighting contrasts or shadows cast by eyebrows/noses. The matrix is then strictly resized to `100x100` (`cv2.resize`). 
7. **Feature Fusion Extraction & Flattening:** Two distinct structural descriptors are extracted simultaneously from the `100x100` matrix:
   - **HOG Array (`skimage.feature.hog`):** A 3,872-dimensional vector capturing the macro-shape outlines and edge directions.
   - **LBP Array (`skimage.feature.local_binary_pattern`):** A ~1,500-dimensional matrix counting micro-texture variances (pores, wrinkles, skin roughness) in a grid context.
   Both are mathematically concatenated into a massive 1D Array shape feature vector.
8. **Dimensionality Reduction (`PCALDARecognizer.predict`):** The 5,300+ dimension vector is linearly transformed via `_pca.transform()` against 50 trained orthogonal principal components. It is then projected using `_lda.transform()` into the Fisherface sub-space, massively reducing noise while concentrating the data across class boundaries.

---

## 3. Open-Set Recognition (Two-Stage Triage)
To solve "Unknown Class" false positives commonly caused by using single generic distances like Cosine Similarity, the pipeline enforces a dual-stage mathematical bounds check, executed entirely in `recognizer.py`:

**Stage 1 (Reconstruction Error Bound):** 
Before the identity is judged, the vector's `MSE` mathematical magnitude is evaluated via `PCALDARecognizer.reconstruction_error()`. The pipeline forcibly reconstructs the vector backwards from its 50D PCA state via `pca.inverse_transform` back to the 5,300D raw feature space. If the deviation between the original input and the attempted reconstruction exceeds the `reconstruction_threshold` (e.g., 5000), it implies the physical architecture of the face is alien to the mathematical manifold. The system immediately rejects the identity as `"Unknown"`.

**Stage 2 (SVM Probability Confidence):**
If the vector's physical geometry satisfies Stage 1, it enters the non-linear LDA space to query identity. An RBF-kernel multi-class `SVC(probability=True)` parses the coordinates in the Fisherface space. It outputs a 0.0-1.0 probability threshold array mapped to each trained teammate. If the highest likelihood falls strictly below `svm_prob_threshold`, it triggers a soft reject, labelling the identity `"Unknown"`.

---

## 4. Concurrency Architecture & Inference Strategies
The live graphical layer (`gui_pyside.py`) utilizes a heavily decoupled architecture optimized for hardware and zero UI-blocking. 

- **Multi-Threading Separation:** The system uses true PyQt/PySide concurrency. The primary `QApplication` only handles window GUI painting. A dedicated `QThread` (`CameraThread`) monopolizes USB/Hardware polling, yielding the `1280x720` arrays at `60 FPS`. A third independent `QThread` (`ClassicalMLWorkerThread`) consumes these arrays for inference. 
- **Zero-Copy Architecture:** To prevent the massive RAM overhead of cloning matrices between Python and native C++ Qt components during `60 FPS` rendering, the application executes a strict pointer-handoff. The RGB Numpy array's raw memory address is bound directly to the PySide UI painter via `QImage(frame_rgb.data...)`. This allows instantaneous repaints.
- **KCF Frame Skipping Tracker Strategy:** Heavy Dlib HOG SVM detections consume excessive CPU cycles and would brutally cap the framerate to `~10-15 FPS`. To bypass this, `ClassicalMLWorkerThread` implements a frame-skipping `cv2.TrackerKCF` continuity hook. Standard HOG detection is skipped. Mapped to `DETECT_EVERY_N_FRAMES = 5`, the structural Dlib analysis only triggers twice a second. During the intervening 4 frames, ultra-lightweight CPU-driven Kernelized Correlation Filters (`TrackerKCF`) passively shift the bounding box mathematically matching simple spatial correlations. This logic securely holds bounding box alignment while letting the camera thread sprint at native speed (`~60 FPS`).
- **Pipelined Matrix Multiplication (BLAS / Accelerate):** During training and inference, the matrices are coerced to `C_CONTIGUOUS float` tensors, explicitly chaining the underlying `numpy` layer to Apple's `Accelerate` framework. This offloads the dense PCA/LDA matmuls away from the physical CPU cores, delegating operations natively to the silicon matrix coprocessors for near-GPU performance limits.
