import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import open3d as o3d

ply_path = 'Meetingroom.ply'

pcd = o3d.io.read_point_cloud(ply_path)
ground_truth_pc = np.asarray(pcd.points)

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

res = least_squares(ba_objective, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
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

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc_centered = pc - centroid
    scale = np.max(np.linalg.norm(pc_centered, axis=1))
    return pc_centered / scale

gt_normalized = normalize_point_cloud(ground_truth_pc)
recon_normalized = normalize_point_cloud(optimized_points_3d)
print("Building KD-Trees and calculating Chamfer Distance (this might take a minute)...")
tree_gt = cKDTree(gt_normalized)
tree_recon = cKDTree(recon_normalized)

dist_recon_to_gt, _ = tree_gt.query(recon_normalized)
dist_gt_to_recon, _ = tree_recon.query(gt_normalized)
chamfer_dist = np.mean(dist_recon_to_gt) + np.mean(dist_gt_to_recon)

print(f"Chamfer Distance: {chamfer_dist:.6f}")

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(recon_normalized[:, 0], recon_normalized[:, 1], recon_normalized[:, 2], c='g', s=1, label='Reconstruction')
ax1.set_title('Normalized Reconstructed Point Cloud')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(gt_normalized[:, 0], gt_normalized[:, 1], gt_normalized[:, 2], c='b', s=1, label='Ground Truth')
ax2.set_title('Normalized Ground Truth Point Cloud')
ax2.legend()

plt.savefig('task4.png', dpi=300)
print("Plot saved as task2_3d_point_cloud.png")