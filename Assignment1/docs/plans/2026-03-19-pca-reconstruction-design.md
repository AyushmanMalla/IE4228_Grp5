# PCA Reconstruction Error Design Document
**Date**: 2026-03-19
**Topic**: Open-Set Recognition Rejection (Classical ML)

## 1. Context & Problem
Currently, the pipeline forces every cropped face into the trained PCA/LDA subspace and uses Cosine Distance to map it to the closest gallery identity angle. Because Cosine Distance only evaluates angle and ignores physical vector magnitude, random background objects or "Unknown" people are confidently assigned to whichever gallery member they happen to vaguely align with mathematically. This causes massive False Positives in open-cast live streams.

## 2. Approach: PCA Reconstruction Error
When we project an image into the 50-dimensional PCA subspace, we inevitably lose information. However, if we take that 50D vector and physically reconstruct the 3872-length HOG vector using `pca.inverse_transform`, a face that is "Known" (or very similar to the gallery) will reconstruct with minimal loss. An "Unknown" face will reconstruct terribly because the principal components do not describe its structure.

We will calculate the **Reconstruction Error** (Squared Euclidean Distance between the input vector and its reconstructed output). If `Error > Threshold`, the face is definitively `Unknown`, bypassing the Cosine check entirely.

## 3. Hardware Acceleration Strategy
The user requested leveraging the MacBook's NPU/GPU. Since deep learning frameworks (`mlx`, `torch`) are banned by the classic ML requirement, we will strictly rely on `numpy` and `scikit-learn`. 
**Optimization**: On M-series Macs, `numpy` is compiled against Apple's `Accelerate` framework. `Accelerate` natively offloads `BLAS` (Basic Linear Algebra Subprograms) matrix multiplications directly to the AMX (Apple Matrix Coprocessor), which is fundamentally an NPU. 
- We will explicitly convert vectors to `float32` and `C_CONTIGUOUS` arrays before performing the matrix inversion to ensure the Accelerate framework achieves maximum AMX throughput, yielding near-GPU speeds for the reconstruction matmuls.

## 4. Architecture Updates
**Component 1: `PCALDARecognizer`**
- Add `reconstruction_threshold` to `__init__`.
- Implement `reconstruct()` method utilizing `self._pca.inverse_transform`.
- Modify `predict()` to calculate the SED reconstruction loss. If it exceeds the threshold, return `("Unknown", 1.0)`.

**Component 2: `gui_pyside.py`**
- Add a second `QSlider` dynamically mapped to the `reconstruction_threshold` so the user can physically dial in the "Unknown" rejection sensitivity live alongside the generic Cosine distance.
