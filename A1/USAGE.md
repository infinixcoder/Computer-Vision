# AR Tag Detection & Overlay (A1)

This folder contains an AR tag detection and overlay project implemented in Python using NumPy and OpenCV. The codebase implements tag detection, homography estimation, 2D template overlay, and 3D model rendering using a custom pipeline.

## Requirements
- Python 3.8+ (3.9/3.10/3.11 recommended)
- numpy
- opencv-python

Install dependencies:

```bash
pip install numpy opencv-python
```

## Files
- `main.py` - main runner: processes video/webcam, detects tags, overlays template or 3D model.
- `utils.py` - core image processing and AR utilities (thresholding, contour detection, homography, warp, tag identification, rendering helpers).
- `caliberate_camera.py` - camera calibration utility to generate `camera_matrix.txt` and `distortion_coefficients.txt` in `output/`.
- `get_images.py` - helper for capturing chessboard images for calibration.

## Usage

- Run with webcam (default):

```bash
python main.py
```

- Run with a video file:

```bash
python main.py --video /path/to/video.mp4
```

- Overlay a 2D template image (task 2):

```bash
python main.py --video /path/to/video.mp4 --template /path/to/template.png
```

- Use a 3D `.obj` model (task 3):

1. Generate or place a camera matrix at `output/camera_matrix.txt` (use `caliberate_camera.py` to create it).
2. Run:

```bash
python main.py --video /path/to/video.mp4 --model /path/to/model.obj
```

## Outputs
- `AR_Detection_Output.mp4` — saved processed video.

## Notes & Troubleshooting
- If the output video appears black or wrong size, update `frame_size` in `main.py` to match your camera/video frame dimensions or change the writer logic to use dynamic sizing.
- Ensure `output/camera_matrix.txt` exists before running task 3 (3D rendering).
- The pipeline relies on tags that match the expected 8x8 grid / inner 4x4 payload pattern.


