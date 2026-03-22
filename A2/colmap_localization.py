import cv2
import numpy as np

def refine_pose_gauss_newton(rvec_init, tvec_init, pts_3d, pts_2d, K_mat, iterations=10):
    rvec = rvec_init.copy()
    tvec = tvec_init.copy()
    for _ in range(iterations):
        proj, jacobian = cv2.projectPoints(pts_3d, rvec, tvec, K_mat, None)
        proj = proj.reshape(-1, 2)
        residuals = (pts_2d - proj).reshape(-1)
        J = jacobian[:, :6]
        try:
            delta = np.linalg.inv(J.T @ J) @ J.T @ residuals
            rvec += delta[:3].reshape(3, 1)
            tvec += delta[3:6].reshape(3, 1)
        except np.linalg.LinAlgError:
            break
    return rvec, tvec


# ---------------------------------------------------------
# Step 0: Load the COLMAP map data (from Task 5)
# ---------------------------------------------------------
# Replace this with however you loaded your COLMAP database/txt exports
data = np.load('colmap_exported_data.npz') 
colmap_pts3d = np.float32(data['pts3d'])
colmap_descs = np.float32(data['descs']) 
K = data['K']

print(f"Loaded COLMAP Map: {len(colmap_pts3d)} 3D points and descriptors.")
colmap_descs = colmap_descs / (np.sum(np.abs(colmap_descs), axis=1, keepdims=True) + 1e-7)
colmap_descs = np.sqrt(colmap_descs)
# ---------------------------------------------------------
# Step 1: Build the KD-Tree (Descriptor-space search)
# ---------------------------------------------------------
print("Building KD-Tree over COLMAP descriptors...")
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ---------------------------------------------------------
# Step 2: Initialize SfM's built-in extractor (SIFT)
# ---------------------------------------------------------
sift = cv2.SIFT_create()

video_path_B = 'split_b_truck.mp4'
cap_B = cv2.VideoCapture(video_path_B)

frame_idx = 0
total_frames_processed = 0
localization_failures = 0
localized_cameras = []

print("\nStarting Localization on Split B (Pre-existing COLMAP Map)...")

while cap_B.isOpened():
    ret, curr_frame = cap_B.read()
    if not ret:
        break
        
    if frame_idx % 5 != 0: # Adjust to your extraction interval
        frame_idx += 1
        continue
        
    total_frames_processed += 1
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Extract features using the SfM system's own built-in feature extractor
    curr_kp, curr_desc = sift.detectAndCompute(curr_gray, None)
    
    if curr_desc is None:
        print(f"Frame {frame_idx}: Localization failure (No features extracted)")
        localization_failures += 1
        frame_idx += 1
        continue

    curr_desc = np.float32(curr_desc)

    # 1. L1+root OpenCV descriptors to perfectly match the COLMAP scale
    curr_desc = curr_desc / (np.sum(np.abs(curr_desc), axis=1, keepdims=True) + 1e-7)
    curr_desc = np.sqrt(curr_desc)

    # 2. Query the KD-Tree
    matches = flann.knnMatch(curr_desc, colmap_descs, k=2)
    
    valid_matches = []
    
    # 3. Ratio Test (0.80)
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance and m.distance < 0.5:
                valid_matches.append(m)
                
    # 4. SORT the survivors by distance! (Best matches first)
    valid_matches = sorted(valid_matches, key=lambda x: x.distance)
    
    pts_2d = []
    pts_3d = []
    seen_train_idx = set() 
    
    # 5. Deduplicate 3D points
    for m in valid_matches:
        if m.trainIdx not in seen_train_idx:
            seen_train_idx.add(m.trainIdx)
            pts_2d.append(curr_kp[m.queryIdx].pt)
            pts_3d.append(colmap_pts3d[m.trainIdx])
            
    pts_2d = np.float32(pts_2d)
    pts_3d = np.float32(pts_3d)
    
    if len(pts_2d) < 6:
        localization_failures += 1
        frame_idx += 1
        continue
        
    # 6. Robust PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None, 
        reprojectionError=15.0, 
        iterationsCount=10000, 
        confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP 
    )
    
    # 4. Localization failure constraint: PnP inliers below minimum threshold (6)
    if not success or inliers is None or len(inliers) < 6:
        print(f"Frame {frame_idx}: Localization failure (PnP inliers below minimum threshold of 6)")
        localization_failures += 1
        frame_idx += 1
        continue
        
    # 5. Refine by minimizing reprojection error
    inlier_indices = inliers.flatten()
    inlier_pts_3d = pts_3d[inlier_indices]
    inlier_pts_2d = pts_2d[inlier_indices]
    
    try:
        # Assumes you have your refine_pose_gauss_newton function in this file
        rvec_opt, tvec_opt = refine_pose_gauss_newton(rvec, tvec, inlier_pts_3d, inlier_pts_2d, K)
    except Exception:
        rvec_opt, tvec_opt = rvec, tvec
        
    print(f"Frame {frame_idx} Localized! Inliers: {len(inliers)}")
    localized_cameras.append((rvec_opt, tvec_opt))
    
    frame_idx += 1

# ---------------------------------------------------------
# Step 6: Document the failure rate per scene in your report
# ---------------------------------------------------------
if total_frames_processed > 0:
    failure_rate = (localization_failures / total_frames_processed) * 100
    print("\n--- Split B Localization Report ---")
    print(f"Total Frames Processed: {total_frames_processed}")
    print(f"Successful Localizations: {len(localized_cameras)}")
    print(f"Localization Failures: {localization_failures}")
    print(f"Failure Rate: {failure_rate:.2f}%")