# Feature Fusion + Two-Stage Triage Design Document

**Date**: 2026-03-23  
**Topic**: Improving Open-Set Rejection + Closed-Set Discrimination

## 1. Context & Problem

The current pipeline has two critical failure modes:

1. **False positives on unknowns**: Random people/objects get confidently labelled as teammates because OneClassSVM is trained on LDA-distorted space (where unknowns can overlap with known clusters).
2. **Misidentification between similar-looking teammates**: HOG-only features (3,872-dim) miss micro-texture differences. Two people with similar bone structure produce near-identical HOG vectors.

## 2. Approved Approach: Two-Stage Pragmatic

### Feature Engineering Changes

| Change | Current | New |
|--------|---------|-----|
| Histogram eq | Global `cv2.equalizeHist` | CLAHE (`clipLimit=2.0, tileGridSize=(8,8)`) |
| Features | HOG only (3,872-dim) | HOG + LBP concatenated (~5,347-dim) |
| Alignment | Haar eye cascade, rotation-only | dlib 68-point landmarks, similarity transform (rotation + scale + translation) |

### Recognition Architecture

```
Input → CLAHE → dlib Landmark Align → Resize(100×100) → HOG + LBP → Concatenate → PCA(50)
  ↓
Stage 1: PCA Reconstruction Error → reject if MSE > τ₁ (Unknown)
  ↓
Stage 2: LDA → Per-Class Mahalanobis (pooled within-class covariance, Ledoit-Wolf shrinkage) → Identity + Rejection at τ₂
```

- **Stage 1** catches background objects and non-gallery faces cheaply
- **Stage 2** replaces both OneClassSVM and cosine NN with a single principled distance metric

## 3. Architecture Updates

### preprocessor.py
- Replace `equalize_histogram()` with `clahe()` using `cv2.createCLAHE`
- Add `compute_lbp_histogram()` using `skimage.feature.local_binary_pattern`
- Update `preprocess_face()` to return concatenated `[HOG | LBP]` vector

### detector.py
- Upgrade `FaceAligner` to use `dlib.shape_predictor` with 68-point landmarks
- Implement similarity transform pinning eyes to fixed target coordinates `(30, 35)` and `(70, 35)` on the 100×100 crop

### recognizer.py
- Remove OneClassSVM entirely
- Add `_reconstruct_error()` method for PCA reconstruction MSE check
- Replace cosine NN in `predict()` with per-class Mahalanobis distance using `sklearn.covariance.LedoitWolf`
- Store per-class means and pooled shrinkage covariance at training time

### config.py
- Add `reconstruction_threshold: float` and `mahalanobis_threshold: float`
- Remove `svm_nu`

### gui_pyside.py
- Add a second slider for reconstruction error threshold
- Update `update_threshold` to support both thresholds

### pipeline.py
- Pass new config params through to recognizer
