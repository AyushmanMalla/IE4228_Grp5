# Hybrid Distance Rejection Design Document
**Date**: 2026-03-19
**Topic**: Solving "Unknown Class" False Positives via Dual-Axis Metrics

## 1. Context & Problem
We recently replaced Squared Euclidean Distance (SED) with Cosine Similarity to solve catastrophic label jittering. Cosine perfectly ignores lighting intensity scaling because it only tracks the *Angle* of the matrices.

However, this created a massive vulnerability in **Open-Set Recognition** (rejecting unknown faces). Because Cosine *only* checks the angle, an entirely random face passing the camera might have a structural ratio that accidentally angles towards the "Ayushman" subspace. The system gives it a 99% confident lock, completely ignoring the fact that the actual numerical distance (magnitude) of the face is mathematically thousands of units outside the gallery!

## 2. Approach: Dual-Axis Envelope
We will implement **Approach 3: Hybrid Euclidean + Cosine Thresholding**. 

We will calculate *both* distances simultaneously in the LDA space. 
- **Cosine** will strictly lock the *Identity* (Who is this?) completely ignoring lighting.
- **Euclidean** will strictly lock the *Probability bounds* (Do I actually know you?) measuring absolute spatial magnitude from the cluster epicenter.

For a face to be classified as a known teammate, it must satisfy **both** bounds simultaneously. If it fails either, we trivially reject the detection as `"Unknown"`.

## 3. Architecture Updates
**Component 1: `config.py`**
- Interleave a new parameter: `euclidean_threshold: float = 3000.0` (TBD baseline constraint).

**Component 2: `recognizer.py`**
- Modify `PCALDARecognizer.predict()`. 
- Pull `scipy.spatial.distance.euclidean` natively down alongside `cosine`.
- Compare the query subspace projection against the target centroid using both functions.
- Write the interception: `if best_cosine > cos_thresh or best_euclidean > euc_thresh: return "Unknown"`.

**Component 3: `gui_pyside.py`**
- Draw a secondary UI Slider directly alongside the Cosine threshold to permit live, real-time tweaking of the Euclidean Spatial Boundary by the user, immediately exposing the Open-Set strictness factor.
