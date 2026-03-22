import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import open3d as o3d
import sqlite3

video_path = 'split_a_truck-004.mp4'
cap = cv2.VideoCapture(video_path)

ret1, frame1 = cap.read()
for _ in range(10):
    cap.read()
ret2, frame2 = cap.read()

if not ret1 or not ret2:
    print("Could not read frames.")
    exit()

H, W = frame1.shape[:2]
f = 0.7 * W
cx = W / 2
cy = H / 2
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0,  1]], dtype=np.float64)

akaze = cv2.AKAZE_create()
sift = cv2.SIFT_create()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

kp1, desc1 = sift.detectAndCompute(gray1, None)
kp2, desc2 = sift.detectAndCompute(gray2, None)

map_data = {
    "frame1": {"keypoints": kp1, "descriptors": desc1},
    "frame2": {"keypoints": kp2, "descriptors": desc2}
}

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass an empty dictionary {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

desc1_f32 = np.float32(desc1)
desc2_f32 = np.float32(desc2)

matches = flann.knnMatch(desc1_f32, desc2_f32, k=2)

good_matches = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
E1_raw = K.T @ F @ K
U, S, Vt = np.linalg.svd(E1_raw)
S_rank2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
E1 = U @ S_rank2 @ Vt

E2, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

if len(good_matches) >= 10:
    sample_indices = random.sample(range(len(good_matches)), 10)
    sample_pts1 = pts1[sample_indices]
    sample_pts2 = pts2[sample_indices]

    lines1 = cv2.computeCorrespondEpilines(sample_pts1.reshape(-1, 1, 2), 1, F)
    lines1 = lines1.reshape(-1, 3)

    img1_E1 = frame1.copy()
    img2_E1 = frame2.copy()
    
    for r, pt1, pt2 in zip(lines1, sample_pts1, sample_pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [W, -(r[2]+r[0]*W)/r[1]])
        
        img2_E1 = cv2.line(img2_E1, (x0, y0), (x1, y1), color, 1)
        img2_E1 = cv2.circle(img2_E1, tuple(map(int, pt2)), 5, color, -1)
        img1_E1 = cv2.circle(img1_E1, tuple(map(int, pt1)), 5, color, -1)

    #cv2.imshow("Epipolar Lines (E1 - All points)", img2_E1)
    #cv2.waitKey(0)

#--------------------------------------------------------------------------------------------------------------
"""TASK 2"""


inlier_mask = mask.ravel() == 1
inliers1 = pts1[inlier_mask]
inliers2 = pts2[inlier_mask]

R1, R2, t = cv2.decomposeEssentialMat(E2)
candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

print("Candidate Poses (R, t):")
for i, (R, t_vec) in enumerate(candidates):
    print(f"Candidate {i+1}:\nR:\n{R}\nt:\n{t_vec}\n")

def linear_triangulation(P1, P2, pt1, pt2):
    A = np.zeros((4, 4))
    A[0] = pt1[0] * P1[2, :] - P1[0, :]
    A[1] = pt1[1] * P1[2, :] - P1[1, :]
    A[2] = pt2[0] * P2[2, :] - P2[0, :]
    A[3] = pt2[1] * P2[2, :] - P2[1, :]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

best_pose = None
best_P2 = None
best_points_3d_initial = []
max_positive_depths = -1

for idx, (R, t_vec) in enumerate(candidates):
    P2 = K @ np.hstack((R, t_vec))
    current_points_3d = []
    positive_depth_count = 0
    
    for p1, p2 in zip(inliers1, inliers2):
        X = linear_triangulation(P1, P2, p1, p2)
        current_points_3d.append(X)
        
        Z_A = X[2]
        X_camB = R @ X + t_vec.flatten()
        Z_B = X_camB[2]
        
        if Z_A > 0 and Z_B > 0:
            positive_depth_count += 1
            
    print(f"Candidate {idx+1} valid cheirality points: {positive_depth_count}/{len(inliers1)}")
    
    if positive_depth_count > max_positive_depths:
        max_positive_depths = positive_depth_count
        best_pose = (R, t_vec)
        best_P2 = P2
        best_points_3d_initial = np.array(current_points_3d)

print(f"\nSelected Pose maximizes positive depth with {max_positive_depths} points.")

def compute_reprojection_error(points_3d, P1, P2, pts1, pts2):
    total_error = 0
    for X, p1, p2 in zip(points_3d, pts1, pts2):
        y1 = P1 @ np.append(X, 1.0)
        y2 = P2 @ np.append(X, 1.0)
        p1_prime = y1[:2] / y1[2]
        p2_prime = y2[:2] / y2[2]
        total_error += np.linalg.norm(p1_prime - p1) + np.linalg.norm(p2_prime - p2)
    return total_error / (2 * len(points_3d))

def refine_point_fixed_iterations(X_init, P1, P2, pt1, pt2, iterations=10):
    X = X_init.copy()
    for _ in range(iterations):
        J = np.zeros((4, 3))
        r = np.zeros(4)
        
        for i, (P, pt) in enumerate([(P1, pt1), (P2, pt2)]):
            y = P @ np.append(X, 1.0)
            y1, y2, y3 = y
            
            p_prime = np.array([y1/y3, y2/y3])
            r[2*i:2*i+2] = p_prime - pt
            
            m1, m2, m3 = P[0,:3], P[1,:3], P[2,:3]
            J[2*i] = (y3 * m1 - y1 * m3) / (y3**2)
            J[2*i+1] = (y3 * m2 - y2 * m3) / (y3**2)
            
        try:
            delta = np.linalg.inv(J.T @ J) @ J.T @ r
            X -= delta
        except np.linalg.LinAlgError:
            break
            
    return X

initial_error = compute_reprojection_error(best_points_3d_initial, P1, best_P2, inliers1, inliers2)

refined_points_3d = []
for X_init, p1, p2 in zip(best_points_3d_initial, inliers1, inliers2):
    X_opt = refine_point_fixed_iterations(X_init, P1, best_P2, p1, p2, iterations=10)
    refined_points_3d.append(X_opt)
    
refined_points_3d = np.array(refined_points_3d)
refined_error = compute_reprojection_error(refined_points_3d, P1, best_P2, inliers1, inliers2)

print(f"\nAverage Reprojection Error (Initial): {initial_error:.4f} pixels")
print(f"Average Reprojection Error (Refined): {refined_error:.4f} pixels")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(best_points_3d_initial[:, 0], best_points_3d_initial[:, 1], best_points_3d_initial[:, 2], c='b', marker='.', s=5)
ax1.set_title('Initial Sparse 3D Point Cloud')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(refined_points_3d[:, 0], refined_points_3d[:, 1], refined_points_3d[:, 2], c='g', marker='.', s=5)
ax2.set_title('Refined Sparse 3D Point Cloud')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

plt.tight_layout()
plt.savefig('task2_3d_point_cloud.png', dpi=300)
print("Plot saved as task2_3d_point_cloud.png")


#---------------------------------------------------------------------------------------------------------
"""TASK 3"""

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

global_map_pts = list(refined_points_3d)
global_map_descs = []
desc_to_pt_idx = []

camera_indices_list = []
point_indices_list = []
points_2d_list = []

good_matches_inliers = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
for i, m in enumerate(good_matches_inliers):
    global_map_descs.append(desc1[m.queryIdx])
    desc_to_pt_idx.append(i)
    global_map_descs.append(desc2[m.trainIdx])
    desc_to_pt_idx.append(i)
    
    camera_indices_list.extend([0, 1])
    point_indices_list.extend([i, i])
    points_2d_list.extend([kp1[m.queryIdx].pt, kp2[m.trainIdx].pt])

global_map_descs = np.array(global_map_descs)

cameras = [{"P": P1, "R": np.eye(3), "t": np.zeros((3,1))}, 
           {"P": best_P2, "R": best_pose[0], "t": best_pose[1]}]

prev_kp = kp2
prev_desc = desc2
prev_P = best_P2

frame_count = 2
max_frames_to_process = 5

while cap.isOpened() and frame_count < max_frames_to_process:
    for _ in range(10):
        cap.read()
    ret, curr_frame = cap.read()
    if not ret:
        break
        
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_kp, curr_desc = sift.detectAndCompute(curr_gray, None)
    
    matches_2d3d = flann.knnMatch(curr_desc, global_map_descs, k=2)
    good_2d3d = []
    pts_2d = []
    pts_3d = []
    matched_pt_indices = []
    
    for m, n in matches_2d3d:
        if m.distance < 0.75 * n.distance:
            good_2d3d.append(m)
            pts_2d.append(curr_kp[m.queryIdx].pt)
            pt_idx = desc_to_pt_idx[m.trainIdx]
            pts_3d.append(global_map_pts[pt_idx])
            matched_pt_indices.append(pt_idx)
            
    pts_2d = np.float32(pts_2d)
    pts_3d = np.float32(pts_3d)
    
    if len(pts_2d) < 6:
        break
        
    success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None, reprojectionError=8.0, flags=cv2.SOLVEPNP_EPNP
    )
    
    if success and inliers_pnp is not None:
        inlier_indices = inliers_pnp.flatten()
        inlier_pts_3d = pts_3d[inlier_indices]
        inlier_pts_2d = pts_2d[inlier_indices]
        
        rvec_opt, tvec_opt = refine_pose_gauss_newton(rvec, tvec, inlier_pts_3d, inlier_pts_2d, K)
        
        R_curr, _ = cv2.Rodrigues(rvec_opt)
        P_curr = K @ np.hstack((R_curr, tvec_opt))
        
        cam_idx = len(cameras)
        cameras.append({"P": P_curr, "R": R_curr, "t": tvec_opt})
        
        for idx in inlier_indices:
            camera_indices_list.append(cam_idx)
            point_indices_list.append(matched_pt_indices[idx])
            points_2d_list.append(pts_2d[idx])
        
        matches_exp = flann.knnMatch(prev_desc, curr_desc, k=2)
        new_pts_1 = []
        new_pts_2 = []
        new_desc_1 = []
        new_desc_2 = []
        
        for m, n in matches_exp:
            if m.distance < 0.75 * n.distance:
                new_pts_1.append(prev_kp[m.queryIdx].pt)
                new_pts_2.append(curr_kp[m.trainIdx].pt)
                new_desc_1.append(prev_desc[m.queryIdx])
                new_desc_2.append(curr_desc[m.trainIdx])
                
        new_pts_1 = np.float32(new_pts_1)
        new_pts_2 = np.float32(new_pts_2)
        
        for idx in range(len(new_pts_1)):
            X_init = linear_triangulation(prev_P, P_curr, new_pts_1[idx], new_pts_2[idx])
            X_opt = refine_point_fixed_iterations(X_init, prev_P, P_curr, new_pts_1[idx], new_pts_2[idx], iterations=10)
            
            Z_A = (cameras[-2]["R"] @ X_opt + cameras[-2]["t"].flatten())[2]
            Z_B = (R_curr @ X_opt + tvec_opt.flatten())[2]
            
            if Z_A > 0 and Z_B > 0:
                err = compute_reprojection_error([X_opt], prev_P, P_curr, [new_pts_1[idx]], [new_pts_2[idx]])
                if err < 5.0:
                    pt_index = len(global_map_pts)
                    global_map_pts.append(X_opt)
                    global_map_descs = np.vstack((global_map_descs, new_desc_1[idx], new_desc_2[idx]))
                    desc_to_pt_idx.extend([pt_index, pt_index])
                    
                    camera_indices_list.extend([cam_idx - 1, cam_idx])
                    point_indices_list.extend([pt_index, pt_index])
                    points_2d_list.extend([new_pts_1[idx], new_pts_2[idx]])
                    
        prev_kp = curr_kp
        prev_desc = curr_desc
        prev_P = P_curr
        frame_count += 1
    else:
        break

camera_indices = np.array(camera_indices_list, dtype=int)
point_indices = np.array(point_indices_list, dtype=int)
points_2d = np.array(points_2d_list, dtype=np.float64)

map_pts_arr = np.array(global_map_pts)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(map_pts_arr[:, 0], map_pts_arr[:, 1], map_pts_arr[:, 2], c='g', marker='.', s=2)

for i, cam in enumerate(cameras):
    C = -np.linalg.inv(cam["R"]) @ cam["t"]
    ax.scatter(C[0], C[1], C[2], c='r', marker='^', s=50)
    ax.text(C[0][0], C[1][0], C[2][0], f'C{i}', color='red')

ax.set_title(f'Incremental Map: {len(cameras)} Cameras, {len(map_pts_arr)} Points')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.savefig('task3.png', dpi=300)
print("Plot saved as task2_3d_point_cloud.png")


#-----------------------------------------------------------------------------------------------------------------------------
"""TASK 4"""

ply_path = 'Truck.ply'

pcd = o3d.io.read_point_cloud(ply_path)
downsampled_pcd = pcd.uniform_down_sample(every_k_points=1000)
ground_truth_pc = np.asarray(downsampled_pcd.points)

n_cameras = len(cameras)
n_points = len(global_map_pts)

camera_params = np.empty((n_cameras, 6))
for i, cam in enumerate(cameras):
    rvec, _ = cv2.Rodrigues(cam["R"])
    camera_params[i, :3] = rvec.flatten()
    camera_params[i, 3:] = cam["t"].flatten()

points_3d = np.array(global_map_pts)

def project_vectorized(points, camera_params, K):
    projections = np.empty((len(points), 2))
    for i in range(len(points)):
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:]
        proj, _ = cv2.projectPoints(points[i:i+1], rvec, tvec, K, None)
        projections[i] = proj.ravel()
    return projections

def ba_objective(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    projs = project_vectorized(points_3d[point_indices], camera_params[camera_indices], K)
    return (projs - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
        
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        
    return A

x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = ba_objective(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K)
initial_mean_error = np.mean(np.linalg.norm(f0.reshape(-1, 2), axis=1))

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

res = least_squares(ba_objective, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-3, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K))

optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))

f_opt = ba_objective(res.x, n_cameras, n_points, camera_indices, point_indices, points_2d, K)
optimized_mean_error = np.mean(np.linalg.norm(f_opt.reshape(-1, 2), axis=1))

print(f"Mean Reprojection Error (Before BA): {initial_mean_error:.4f} pixels")
print(f"Mean Reprojection Error (After BA): {optimized_mean_error:.4f} pixels")

global_map_pts = list(optimized_points_3d)

for i in range(n_cameras):
    rvec = optimized_camera_params[i, :3]
    tvec = optimized_camera_params[i, 3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    cameras[i]["R"] = R
    cameras[i]["t"] = tvec
    cameras[i]["P"] = K @ np.hstack((R, tvec))

def filter_pc(pc):
    centroid = np.median(pc, axis=0)
    centered = pc - centroid
    distances = np.linalg.norm(centered, axis=1)
    median_dist = np.median(distances)
    valid_mask = distances < (median_dist * 5)
    return centered[valid_mask]

print("Filtering extreme outliers...")
gt_centered = filter_pc(ground_truth_pc)
recon_centered = filter_pc(optimized_points_3d)

print(f"Cleaned Reconstruction Points: {recon_centered.shape[0]}")

# Calculate ONE global scale factor (the larger of the two)
scale_gt = np.max(np.linalg.norm(gt_centered, axis=1))
scale_recon = np.max(np.linalg.norm(recon_centered, axis=1))
global_scale = max(scale_gt, scale_recon)
if global_scale == 0: global_scale = 1.0

print(f"Normalizing BOTH clouds by global scale: {global_scale:.4f}")
gt_normalized = gt_centered / global_scale
recon_normalized = recon_centered / global_scale

print("Building KD-Trees and calculating Chamfer Distance...")
tree_gt = cKDTree(gt_normalized)
tree_recon = cKDTree(recon_normalized)

dist_recon_to_gt, _ = tree_gt.query(recon_normalized)
dist_gt_to_recon, _ = tree_recon.query(gt_normalized)
chamfer_dist = np.mean(dist_recon_to_gt) + np.mean(dist_gt_to_recon)

print(f"Final Rule-Compliant Chamfer Distance: {chamfer_dist:.6f}")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(recon_normalized[:, 0], recon_normalized[:, 1], recon_normalized[:, 2], c='g', s=1, label='Reconstruction')
ax1.set_title('Normalized Reconstructed Point Cloud')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(gt_normalized[:, 0], gt_normalized[:, 1], gt_normalized[:, 2], c='b', s=1, label='Ground Truth')
ax2.set_title('Normalized Ground Truth Point Cloud')
ax2.legend()

plt.savefig('task4_bundle_adjustment_comparison.png', dpi=300)
print("Saved Task 4 visualization to task4_bundle_adjustment_comparison.png")

np.savez('custom_map_data.npz', 
         pts3d=optimized_points_3d,   # Use the refined BA points!
         descs=global_map_descs, 
         mapping=desc_to_pt_idx, 
         K=K)

print("Saved Task 1-4 map variables to custom_map_data.npz!")

#-----------------------------------------------------------------------------------------------------------------------
"""TASK 5"""

db_path = 'colmap_project/database.db'
points3D_path = 'colmap_project/text_output/points3D.txt'
images_path = 'colmap_project/text_output/images.txt'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

image_id_map = {}
with open(images_path, 'r') as f:
    lines = f.readlines()
    for i in range(4, len(lines), 2):
        elems = lines[i].split()
        image_id = int(elems[0])
        image_name = elems[-1]
        cursor.execute("SELECT image_id FROM images WHERE name=?", (image_name,))
        db_image_id = cursor.fetchone()[0]
        image_id_map[image_id] = db_image_id

colmap_3d_points = []
colmap_descriptors = []
desc_to_colmap_pt_idx = []

with open(points3D_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        elems = line.split()
        x, y, z = map(float, elems[1:4])
        
        img_id_txt = int(elems[8])
        pt2d_idx = int(elems[9])
        
        db_img_id = image_id_map.get(img_id_txt)
        if db_img_id is None:
            continue

        cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (db_img_id,))
        row = cursor.fetchone()
        if row is None:
            continue
            
        rows, cols, data = row
        descs = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        
        if pt2d_idx < rows:
            colmap_3d_points.append([x, y, z])
            colmap_descriptors.append(descs[pt2d_idx])
            desc_to_colmap_pt_idx.append(len(colmap_3d_points) - 1)

conn.close()

colmap_3d_points = np.array(colmap_3d_points)
colmap_descriptors = np.float32(colmap_descriptors)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
colmap_flann = cv2.FlannBasedMatcher(index_params, search_params)

colmap_flann.add([colmap_descriptors])
colmap_flann.train()

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    scale = np.max(np.linalg.norm(pc_centered, axis=1))
    return pc_centered / scale

gt_normalized = normalize_point_cloud(ground_truth_pc)
colmap_recon_normalized = normalize_point_cloud(colmap_3d_points)

tree_gt = cKDTree(gt_normalized)
tree_colmap = cKDTree(colmap_recon_normalized)

dist_recon_to_gt, _ = tree_gt.query(colmap_recon_normalized)
dist_gt_to_recon, _ = tree_colmap.query(gt_normalized)
colmap_chamfer_dist = np.mean(dist_recon_to_gt) + np.mean(dist_gt_to_recon)

print(f"COLMAP Chamfer Distance: {colmap_chamfer_dist:.6f}")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(colmap_recon_normalized[:, 0], colmap_recon_normalized[:, 1], colmap_recon_normalized[:, 2], c='g', s=1)
ax1.set_title('COLMAP Normalized Reconstruction')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(gt_normalized[:, 0], gt_normalized[:, 1], gt_normalized[:, 2], c='b', s=1)
ax2.set_title('Ground Truth Point Cloud')

plt.savefig('task5', dpi=300)
print("Plot saved as task5.png")

np.savez('colmap_exported_data.npz', 
         pts3d=colmap_3d_points, 
         descs=colmap_descriptors,
         K=K)

print(f"Successfully saved {len(colmap_3d_points)} COLMAP points and descriptors to colmap_exported_data.npz!")

#-----------------------------------------------------------------------------------------------------------------------
"""TASK 6"""

cap.release()
cv2.destroyAllWindows()