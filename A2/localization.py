import numpy as np
import cv2
import matplotlib.pyplot as plt


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
# 1. Load the SIFT-based Map Data
# ---------------------------------------------------------
print("Loading saved SIFT map data...")
data = np.load('custom_map_data.npz')
global_map_pts = data['pts3d']
global_map_descs = data['descs']
desc_to_pt_idx = data['mapping'].tolist()
K = data['K']

print(f"Loaded {len(global_map_pts)} 3D points and {len(global_map_descs)} descriptors.")

# Calculate map boundaries to filter out extreme 3D SfM artifacts
centroid = np.median(global_map_pts, axis=0)
map_distances = np.linalg.norm(global_map_pts - centroid, axis=1)
median_dist = np.median(map_distances)
valid_threshold = median_dist * 5 
print(f"Calculated map boundaries. Rejecting 3D points further than {valid_threshold:.2f}")

# ---------------------------------------------------------
# 2. Initialize SIFT & Matcher
# ---------------------------------------------------------
sift = cv2.SIFT_create()

# SIFT uses L2 (Euclidean) distance, NOT Hamming.
bf_matcher = cv2.BFMatcher() 

# Force map descriptors to float32 (SIFT standard) to prevent type errors
map_descs_f32 = np.float32(global_map_descs)

# ---------------------------------------------------------
# 3. Setup Video & Tracking Variables
# ---------------------------------------------------------
video_path_B = 'split_b_truck.mp4'
cap_B = cv2.VideoCapture(video_path_B)

localized_cameras = []
frame_indices = []
reprojection_errors = []
inlier_ratios = []
frame_idx = 0

print("Starting Localization on Split B (Custom SIFT Map)...")

while cap_B.isOpened():
    ret, curr_frame = cap_B.read()
    if not ret:
        break
        
    if frame_idx % 5 != 0: 
        frame_idx += 1
        continue
        
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Extract SIFT features
    curr_kp, curr_desc = sift.detectAndCompute(curr_gray, None)
    
    if curr_desc is None or len(curr_desc) < 6:
        print(f"Frame {frame_idx}: Localization Failure (Not enough SIFT descriptors)")
        frame_idx += 1
        continue

    curr_desc_f32 = np.float32(curr_desc)

    # ---------------------------------------------------------
    # 4. Standard KNN Matching & Lowe's Ratio Test
    # ---------------------------------------------------------
    knn_matches = bf_matcher.knnMatch(curr_desc_f32, map_descs_f32, k=2)
    
    pts_2d = []
    pts_3d = []
    
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            
            # Lowe's Ratio Test (0.75 is the proven sweet spot for SIFT)
            if m.distance < 0.8 * n.distance:
                pt_idx = desc_to_pt_idx[m.trainIdx]
                
                # Protect against 3D outlier points
                if map_distances[pt_idx] < valid_threshold:
                    pts_2d.append(curr_kp[m.queryIdx].pt)
                    pts_3d.append(global_map_pts[pt_idx])
                    
    pts_2d = np.float32(pts_2d)
    pts_3d = np.float32(pts_3d)
    
    if len(pts_2d) < 6:
        print(f"Frame {frame_idx}: Localization Failure (Only {len(pts_2d)} ratio-tested matches)")
        frame_idx += 1
        continue
        
    # ---------------------------------------------------------
    # 5. Robust PnP Solver
    # ---------------------------------------------------------
    success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None, 
        reprojectionError=12.0,   # SIFT is more accurate, we can tighten this back to 12
        iterationsCount=5000, 
        confidence=0.99,
        flags=cv2.SOLVEPNP_SQPNP
    )
    
    if success and inliers_pnp is not None and len(inliers_pnp) >= 6:
        inlier_count = len(inliers_pnp)
        inlier_ratio = inlier_count / len(pts_2d)
        
        inlier_indices = inliers_pnp.flatten()
        inlier_pts_3d = pts_3d[inlier_indices]
        inlier_pts_2d = pts_2d[inlier_indices]
        
        try:
            # Assumes your custom refiner function is pasted in this file
            rvec_opt, tvec_opt = refine_pose_gauss_newton(rvec, tvec, inlier_pts_3d, inlier_pts_2d, K)
        except Exception:
            rvec_opt, tvec_opt = rvec, tvec
        
        # Calculate final reprojection error
        proj, _ = cv2.projectPoints(inlier_pts_3d, rvec_opt, tvec_opt, K, None)
        proj = proj.reshape(-1, 2)
        err = np.mean(np.linalg.norm(proj - inlier_pts_2d, axis=1))
        
        R_curr, _ = cv2.Rodrigues(rvec_opt)
        
        localized_cameras.append({"R": R_curr, "t": tvec_opt})
        frame_indices.append(frame_idx)
        reprojection_errors.append(err)
        inlier_ratios.append(inlier_ratio)
        
        print(f"Frame {frame_idx} Localized: Error={err:.3f}px, Inliers={inlier_count}/{len(pts_2d)} ({inlier_ratio:.1%})")
        
    else:
        print(f"Frame {frame_idx}: Localization Failure (PnP RANSAC rejected points. Total matches: {len(pts_2d)})")
        
    frame_idx += 1

cap_B.release()

# ---------------------------------------------------------
# 6. Final Evaluation Metrics & Plotting
# ---------------------------------------------------------
if len(reprojection_errors) > 0:
    mean_err = np.mean(reprojection_errors)
    std_err = np.std(reprojection_errors)
    mean_inlier_ratio = np.mean(inlier_ratios)
    
    print("\n--- Split B Localization Summary ---")
    print(f"Successfully localized: {len(localized_cameras)} frames.")
    print(f"Mean Reprojection Error: {mean_err:.4f} pixels (Std Dev: {std_err:.4f})")
    print(f"Mean Inlier Ratio: {mean_inlier_ratio:.2%}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(frame_indices, reprojection_errors, marker='o', color='b')
    ax1.set_title("Temporal Reprojection Error (Drift Analysis)")
    ax1.set_ylabel("Error (pixels)")
    ax1.grid(True)
    
    ax2.plot(frame_indices, inlier_ratios, marker='s', color='g')
    ax2.set_title("Inlier Ratio Stability")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Inlier Ratio")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('task6_localization_metrics.png', dpi=300)
    print("Saved metrics plot to task6_localization_metrics.png")
else:
    print("Failed to localize any frames in Split B.")