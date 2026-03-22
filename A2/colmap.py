import os
import subprocess
import cv2
import glob

os.environ["QT_QPA_PLATFORM"] = "offscreen"

video_path = 'split_a_truck-004.mp4'
workspace = 'colmap_project_truck'
image_dir = os.path.join(workspace, 'images')
sparse_dir = os.path.join(workspace, 'sparse')
text_dir = os.path.join(workspace, 'text_output')
db_path = os.path.join(workspace, 'database.db')

for folder in [workspace, image_dir, sparse_dir, text_dir]:
    os.makedirs(folder, exist_ok=True)

existing_images = glob.glob(os.path.join(image_dir, '*.jpg'))

if len(existing_images) > 0:
    print(f"Found {len(existing_images)} existing images. Skipping video extraction!")
    # We still need to grab the video resolution once to calculate intrinsics
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W = frame.shape[:2]
    cap.release()
else:
    print("No existing images found. Extracting frames from video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        exit()

    H, W = frame.shape[:2]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        # Instantly skip to the next frame without decoding pixels
        ret = cap.grab() 
        if not ret:
            break
            
        if frame_count % 300 == 0:
            # ONLY decode the pixels on the 300th frame
            ret, frame = cap.retrieve() 
            if ret:
                img_path = os.path.join(image_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                saved_count += 1
                
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames.")


fx = 0.7 * W
fy = 0.7 * W
cx = W / 2.0
cy = H / 2.0
camera_params = f"{fx},{fy},{cx},{cy}"

subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", db_path,
    "--image_path", image_dir,
    "--ImageReader.camera_model", "PINHOLE",
    "--ImageReader.camera_params", camera_params,
    "--ImageReader.single_camera", "1"
], check=True)

subprocess.run([
    "colmap", "exhaustive_matcher",
    "--database_path", db_path,
], check=True)

os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
subprocess.run([
    "colmap", "mapper",
    "--database_path", db_path,
    "--image_path", image_dir,
    "--output_path", sparse_dir
], check=True)

subprocess.run([
    "colmap", "model_converter",
    "--input_path", os.path.join(sparse_dir, "0"),
    "--output_path", text_dir,
    "--output_type", "TXT"
], check=True)

print("COLMAP pipeline finished successfully.") 