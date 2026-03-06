# A1 — AR Tag Detection and Overlay

## 1. Implementation Approach (Overview)

Pipeline steps implemented in `utils.py` and orchestrated from `main.py`:

- Frame acquisition: webcam or input video (via `--video`).
- Grayscale conversion: simple weighted sum of BGR channels.
- Automatic thresholding: Kittler–Illingworth method to separate foreground/background.
- Binary conversion and morphological erosion to clean noise.
- Custom contour detection to extract borders and hierarchical information without relying on OpenCV high-level detectors.
- Candidate extraction: perimeter-based filtering, polygon approximation (Douglas–Peucker), corner merging, convexity checks.
- Corner ordering: consistent clockwise ordering for homography.
- Homography computation: build linear system and solve via SVD for Ah=0, normalize by h33.
- Warp & sampling: inverse mapping from output square to image using homography, vectorized coordinate transforms.
- Tag identification: sample an 8x8 grid, detect orientation marker, read inner 2x2 payload to compute 4-bit ID.
- Overlay: 2D template overlay via inverse-homography sampling; 3D overlay by decomposing homography and projecting an OBJ model with given camera matrix.

## 2. Key Design Decisions and Justifications

- Custom contour detection: avoided dependence on OpenCV's high-level `findContours` to keep algorithmic control and reproduce academic-style border following suitable for nested tags and holes.
- Kittler threshold: chosen for robust automatic separation on variable lighting over naive fixed thresholds.
- Vectorized operations (NumPy): used heavily (histograms, warping, sampling) for performance and clarity.
- SVD-based homography: numerically stable solution for Ah = 0 with normalization.
- Inverse mapping for warping: avoids holes/pixel overlap and preserves source image fidelity.
- Clamping variances and small epsilons in Kittler implementation to avoid numerical issues and log-of-zero errors.

## 3. Challenges and Resolutions

- Noisy / fragmented contours: addressed with an erosion step and a `merge_broken_corners` routine to join near-duplicate corner points.
- Small/degenerate segments during Douglas–Peucker: used perpendicular distance computation robust to degenerate line segments.
- Numerical instability in Kittler: added epsilons and clamped variances to minimum values to avoid log(0) and negative variances.
- Frame size / output video mismatch: `main.py` contains a `frame_size` variable — the README explains adjusting this or dynamically setting the writer.

## 4. Resources Consulted

- OpenCV documentation (image I/O, camera calibration, chessboard detection)
- NumPy documentation (vectorized indexing, meshgrid, linear algebra)
- Kittler & Illingworth - "Minimum Error Thresholding" (algorithm background)
- Zhang, Z. — Camera calibration method papers and tutorials
- Various StackOverflow threads for homography computation, SVD usage, and numpy indexing tips

## 5. Assumptions

- Tags follow an 8x8 grid with an inner 4x4 payload and a single white orientation marker.
- Input images are BGR 8-bit images from OpenCV.
- Camera intrinsics for 3D rendering are available at `camera_matrix.txt` when using `--model`.
- `.obj` models are triangulated and in a simple vertex/face ASCII format; complex materials/textures are not supported by the renderer.

## 6. How to Reproduce / Run

1. (Optional) Capture chessboard images with `get_images.py` and calibrate with `caliberate_camera.py`.
2. Place calibration files in `A1/`.
3. Run detection and overlay with `python main.py --video path --template template.png` or `--model model.obj`.

## 7. Next Improvements (possible enhancements)

- Add unit tests for core utilities (homography, warping, tag identification).
- Add GPU-accelerated sampling for real-time performance.
- Improve 3D rendering (shading, textures) and support more OBJ features.
- Add command-line option to auto-detect frame size for the video writer.

## 8. Output Video Drive Link

- https://drive.google.com/drive/folders/1JzRQCrtomU9hH1DAnTLarBTS8PtSp7Ws


