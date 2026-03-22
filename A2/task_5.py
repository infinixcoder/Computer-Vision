import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import open3d as o3d
import sqlite3

"""TASK 5"""

db_path = 'colmap_project_truck/database.db'
points3D_path = 'colmap_project_truck/text_output/points3D.txt'
images_path = 'colmap_project_truck/text_output/images.txt'

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