# Dlib HOG+SVM Face Detector Design Document
**Date**: 2026-03-19
**Topic**: Classical Detection Bottlenecks

## 1. Context & Problem
Currently, the pipeline's foundational anchor—the physical detection of the face in the 1080p camera feed—is exclusively relying on OpenCV's native `haarcascade_frontalface_default.xml`. 
This implements the 2001 **Viola-Jones** algorithm, which mathematically scans incoming arrays for flat, 2D "Haar" shadow rectangles (e.g., verifying if the eye region is darker than the cheek region). Because it operates strictly on these pixel-value differences, the detector is highly sensitive to rotation, lighting shadows, glasses, or partial obstructions. 

If the Haar bounding box jitters or completely clips half the users face off, the downstream mathematically-perfect PCA Eigenface manifold is flooded with corrupt, misaligned data, destroying Open-Set recognition.

## 2. Approach: Dlib HOG + Linear SVM
We will physically rip out the Haar Cascades and upgrade the architecture to the legendary **2005 Histogram of Oriented Gradients (HOG)** detector backed by a Convex Optimization **Linear Support Vector Machine (SVM)**.

Unlike flat shadows, HOG computes the local structural "gradient" (the edge directions) of every 16x16 pixel cell on your face. This creates a dense mathematical map of edge orientations that are practically immune to lighting variations. The SVM is trained universally on these edge structures to perfectly lock a tight bounding box on human faces under extreme angles.
Because it relies strictly on Gradients and an SVM, it perfectly satisfies the "No Deep Learning" constraint of Assignment 1.

## 3. Architecture Updates
**Component 1: `requirements.txt`**
- Inject `dlib` into the native dependency list.

**Component 2: `detector.py`**
- Create a new class `DlibHOGFaceDetector` that securely implements the structural `FaceDetector` abstract interface natively.
- Import `dlib.get_frontal_face_detector()`.
- The native dlib `rect` coordinates (`left`, `top`, `right`, `bottom`) will be aggressively cast to standard `[x1, y1, x2, y2]` Numpy bound boxes and returned as fully compliant `Detection` dataclass arrays to prevent breaking the downstream tracker logic!

**Component 3: `pipeline.py`**
- Switch the injected detector interface from `HaarFaceDetector` to the new `DlibHOGFaceDetector`.
