import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import open3d as o3d
import os, sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FRAME_SKIP  = 20
RATIO_INIT  = 0.75
RATIO_INCR  = 0.85
OUTLIER_K   = 5.0
MIN_PNP     = 8
MAX_BA_OBS  = 50000

# ==========================================
# 1. CACHING & SIFT EXTRACTION
# ==========================================

def pack_keypoints(keypoints):
    if not keypoints: return np.empty((0, 7), dtype=np.float32)
    return np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                     for kp in keypoints], dtype=np.float32)

def unpack_keypoints(kp_array):
    if kp_array.size == 0: return []
    return [cv2.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3], 
                         response=row[4], octave=int(row[5]), class_id=int(row[6])) 
            for row in kp_array]

def process_video_on_the_fly(path, skip):
    name = os.path.splitext(os.path.basename(path))[0]
    cache_file = f"cache_sift_{name}_skip{skip}.npz"
    
    if os.path.exists(cache_file):
        print(f"  [Cache] Found cached features! Loading {cache_file}...")
        data = np.load(cache_file)
        K = data['K']
        num_frames = sum(1 for k in data.files if k.startswith('desc_'))
        kps_list = [unpack_keypoints(data[f'kp_{i}']) for i in range(num_frames)]
        desc_list = [data[f'desc_{i}'] for i in range(num_frames)]
        print(f"  [Cache] Loaded SIFT features for {len(kps_list)} frames.")
        return K, kps_list, desc_list

    cap, det, kps_list, desc_list, K, n = cv2.VideoCapture(path), cv2.SIFT_create(), [], [], None, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [Compute] Extracting from {total_frames} frames (FRAME_SKIP={skip})...")
    
    while cap.grab():
        if n % skip == 0:
            ret, f = cap.retrieve()
            if ret and f is not None:
                if K is None:
                    h, w = f.shape[:2]
                    K = np.array([[max(h, w), 0, w / 2], [0, max(h, w), h / 2], [0, 0, 1]], dtype=np.float64)
                
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                kp, d = det.detectAndCompute(g, None)
                kps_list.append(kp)
                desc_list.append(d if d is not None else np.zeros((0, 128), np.float32))
                print(f"\r  Scanning frame {n}/{total_frames} | Kept features for {len(kps_list)} frames", end="", flush=True)
        n += 1
        
    cap.release()
    print("\n  Extraction complete! Saving cache...")
    
    save_dict = {'K': K}
    for i, (kps, desc) in enumerate(zip(kps_list, desc_list)):
        save_dict[f'kp_{i}'] = pack_keypoints(kps)
        save_dict[f'desc_{i}'] = desc
    np.savez(cache_file, **save_dict)
    return K, kps_list, desc_list

# ==========================================
# 2. CORE MATH & GEOMETRY
# ==========================================

def flann_match(d1, d2, ratio):
    if len(d1) < 2 or len(d2) < 2: return []
    fl = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    raw = fl.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    return [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]

def triangulate(P1, P2, pts1, pts2):
    h = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
    return (h[:3] / h[3]).T

def reproject_P(P, X):
    x = P @ np.vstack((X.T, np.ones((1, len(X)))))
    return (x[:2] / x[2]).T

def reproject(K, R, t, X):
    return reproject_P(K @ np.hstack([R, t]), X)

def outlier_mask(pts3d):
    if len(pts3d) < 4: return np.ones(len(pts3d), dtype=bool)
    c = np.median(pts3d, axis=0)
    d = np.linalg.norm(pts3d - c, axis=1)
    return d < OUTLIER_K * np.median(d)

# ==========================================
# 3. NON-LINEAR OPTIMIZERS (TASKS 2 & 3)
# ==========================================

def refine_initial_3d(pts3d, P1, P2, p1, p2):
    """Task 2: Refines initial 3D points holding cameras fixed."""
    def res_3d(x, P1, P2, p1, p2):
        X = x.reshape(-1, 3)
        return np.hstack([(reproject_P(P1, X) - p1).ravel(), (reproject_P(P2, X) - p2).ravel()])
    
    res = least_squares(res_3d, pts3d.ravel(), args=(P1, P2, p1, p2), method='trf', loss='huber')
    return res.x.reshape(-1, 3)

def refine_single_pose(R, t, pts3d, pts2d, K):
    """Task 3: Refines a single newly estimated PnP pose."""
    def res_pose(x, X, p, K):
        R_ref = cv2.Rodrigues(x[:3])[0]
        t_ref = x[3:6].reshape(3, 1)
        return (reproject(K, R_ref, t_ref, X) - p).ravel()
    
    x0 = np.hstack([cv2.Rodrigues(R)[0].ravel(), t.ravel()])
    res = least_squares(res_pose, x0, args=(pts3d, pts2d, K), method='trf', loss='huber')
    return cv2.Rodrigues(res.x[:3])[0], res.x[3:6].reshape(3, 1)

def ba_residuals(x, nc, obs_c, obs_p, obs_uv, K):
    cp = x[:nc * 6].reshape(nc, 6)
    pts = x[nc * 6:].reshape(-1, 3)
    Rs = [(cv2.Rodrigues(cp[i, :3])[0], cp[i, 3:6].reshape(3, 1)) for i in range(nc)]
    
    res = []
    for ci, pi, uv in zip(obs_c, obs_p, obs_uv):
        R, t = Rs[ci]
        res.append((reproject(K, R, t, pts[pi:pi+1])[0] - uv))
    return np.array(res).ravel()

def build_sparsity(nc, np_, obs_c, obs_p):
    A = lil_matrix((2 * len(obs_c), nc * 6 + np_ * 3), dtype=int)
    for i, (ci, pi) in enumerate(zip(obs_c, obs_p)):
        A[2*i:2*i+2, ci*6:(ci+1)*6] = 1
        A[2*i:2*i+2, nc*6+pi*3:nc*6+(pi+1)*3] = 1
    return A

def run_ba(cams_list, pts3d, obs_c, obs_p, obs_uv, K, verbose=0, max_iter=30):
    nc, np_ = len(cams_list), len(pts3d)
    x0 = []
    for R, t in cams_list:
        x0 += list(cv2.Rodrigues(R)[0].ravel()) + list(t.ravel())
    x0 = np.array(x0 + list(pts3d.ravel()))
    
    A = build_sparsity(nc, np_, obs_c, obs_p)
    res = least_squares(ba_residuals, x0, jac_sparsity=A, args=(nc, obs_c, obs_p, obs_uv, K),
                        method='trf', loss='huber', max_nfev=max_iter, ftol=1e-3, xtol=1e-3, verbose=verbose)
    
    cp = res.x[:nc * 6].reshape(nc, 6)
    return [(cv2.Rodrigues(cp[i, :3])[0], cp[i, 3:6].reshape(3, 1)) for i in range(nc)], res.x[nc * 6:].reshape(np_, 3)

def chamfer_distance(A, B, n=8000):
    """Calculates the Chamfer Distance between two point clouds."""
    rng = np.random.default_rng(42)
    if len(A) > n: A = A[rng.choice(len(A), n, replace=False)]
    if len(B) > n: B = B[rng.choice(len(B), n, replace=False)]
    d_ab = cKDTree(B).query(A, k=1)[0].mean()
    d_ba = cKDTree(A).query(B, k=1)[0].mean()
    return (d_ab + d_ba) / 2.0

# ==========================================
# 4. VISUALIZATIONS & OUTPUTS
# ==========================================

def draw_epipolar_lines(img1, img2, pts1, pts2, F, out_path):
    r, c = img1.shape[:2]
    img1_color, img2_color = img1.copy(), img2.copy()
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)); axes[0].set_title("Epipolar Lines (Frame A)")
    axes[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)); axes[1].set_title("Epipolar Lines (Frame B)")
    plt.savefig(out_path); plt.close()

def save_map_plot(pts3d, cams_list, title, path):
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection='3d')
    p = np.array(pts3d)
    if len(p) > 0: ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.4, c='green', alpha=0.5)
    for R, t in cams_list:
        pos = (-R.T @ t).ravel()
        ax.scatter(*pos, c='red', s=50, marker='^', zorder=5)
    ax.set_title(title); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()

# ==========================================
# 5. MAIN PIPELINE
# ==========================================

def run_pipeline(video_path, ply_path=None):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = f"results_{name}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}\nScene : {name}\n{'='*55}")

    K, kps, desc = process_video_on_the_fly(video_path, FRAME_SKIP)
    NF = len(kps)
    
    # ------------------------------------------
    # TASK 1: Essential Matrix & Epipolar Geometry
    # ------------------------------------------
    m01 = flann_match(desc[0], desc[1], RATIO_INIT)
    p0r = np.float32([kps[0][m.queryIdx].pt for m in m01])
    p1r = np.float32([kps[1][m.trainIdx].pt for m in m01])

    # Compute E1 (All correspondences, NO robust estimation)
    F_1, _ = cv2.findFundamentalMat(p0r, p1r, cv2.FM_8POINT)
    E_1 = K.T @ F_1 @ K
    
    # Compute E2 (RANSAC)
    E_2, mask_E2 = cv2.findEssentialMat(p0r, p1r, K, cv2.RANSAC, 0.999, 1.0)
    mE = mask_E2.ravel().astype(bool)
    
    # Extract 10 random matches for Epipolar Line plotting
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); _, img0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_SKIP); _, img1 = cap.read()
    cap.release()
    
    idx10 = np.random.choice(len(p0r), min(10, len(p0r)), replace=False)
    draw_epipolar_lines(img0, img1, p0r[idx10], p1r[idx10], F_1, f"{out_dir}/epipolar_E1_NonRobust.png")
    
    F_2 = np.linalg.inv(K).T @ E_2 @ np.linalg.inv(K)
    draw_epipolar_lines(img0, img1, p0r[idx10], p1r[idx10], F_2, f"{out_dir}/epipolar_E2_RANSAC.png")
    print("\n[Task 1] E1 (Non-Robust) and E2 (RANSAC) epipolar plots saved.")

    p0, p1, m01f = p0r[mE], p1r[mE], [m for m, v in zip(m01, mE) if v]

    # ------------------------------------------
    # TASK 2: Decomposition, Cheirality, Refinement
    # ------------------------------------------
    print("\n[Task 2] Decomposing E matrix into 4 Candidate Poses:")
    U, S, Vt = np.linalg.svd(E_2)
    if np.linalg.det(U) < 0: U = -U
    if np.linalg.det(Vt) < 0: Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    cands = [
        (U @ W @ Vt, U[:, 2].reshape(3, 1)),
        (U @ W @ Vt, -U[:, 2].reshape(3, 1)),
        (U @ W.T @ Vt, U[:, 2].reshape(3, 1)),
        (U @ W.T @ Vt, -U[:, 2].reshape(3, 1))
    ]
    
    best_cand, best_pts, max_pos = None, None, -1
    P0_mat = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    for i, (R, t) in enumerate(cands):
        print(f"  Candidate {i+1}: R=\n{R}\n  t=\n{t}")
        P_cand = K @ np.hstack([R, t])
        pts3d_cand = triangulate(P0_mat, P_cand, p0, p1)
        
        # Check Z > 0 in Cam A and Cam B
        Z_A = pts3d_cand[:, 2] > 0
        X_B = (R @ pts3d_cand.T + t).T
        Z_B = X_B[:, 2] > 0
        valid = Z_A & Z_B
        
        print(f"    Candidate {i+1} valid cheirality points: {valid.sum()}/{len(valid)}")
        if valid.sum() > max_pos:
            max_pos = valid.sum()
            best_cand = (R, t)
            best_pts = pts3d_cand[valid]
            p0_v, p1_v = p0[valid], p1[valid]
            m01_v = [m for m, v in zip(m01f, valid) if v]

    R1, t1 = best_cand
    P1_mat = K @ np.hstack([R1, t1])
    print(f"\n[Task 2] Selected Pose maximizes positive depth with {max_pos} points.")
    
    # Boundary filter mapping
    vm = outlier_mask(best_pts)
    i3d = best_pts[vm]
    p0, p1 = p0_v[vm], p1_v[vm]
    m01f = [m for m, v in zip(m01_v, vm) if v]

    # Initial 3D Refinement
    err_pre = np.mean(np.linalg.norm(reproject_P(P1_mat, i3d) - p1, axis=1))
    print(f"  Initial 3D Pre-Refinement Error: {err_pre:.4f}px")
    i3d = refine_initial_3d(i3d, P0_mat, P1_mat, p0, p1)
    err_post = np.mean(np.linalg.norm(reproject_P(P1_mat, i3d) - p1, axis=1))
    print(f"  Initial 3D Post-Refinement Error: {err_post:.4f}px")

    # Tracking & Descriptors Setup
    map_pts = list(i3d)
    map_desc = np.array([desc[1][m.trainIdx] for m in m01f]) # TASK 3: Storing Descriptors
    
    cams_reg = {0: (np.eye(3), np.zeros((3, 1))), 1: (R1, t1)}
    k2p = [{} for _ in range(NF)]
    obs_c, obs_p, obs_uv = [], [], []

    for idx, m in enumerate(m01f):
        k2p[0][m.queryIdx] = idx
        k2p[1][m.trainIdx] = idx
        obs_c += [0, 1]; obs_p += [idx, idx]
        obs_uv += [kps[0][m.queryIdx].pt, kps[1][m.trainIdx].pt]

    # ------------------------------------------
    # TASK 3: Incremental Mapping
    # ------------------------------------------
    print("\n[Task 3] Starting Incremental PnP Mapping...")
    for i in range(2, NF):
        prev_keys = sorted(k for k in cams_reg if k < i)
        if not prev_keys: continue
        
        # Lookback matching to prevent PnP failure
        p2d, p3d, trk = [], [], []
        for pv in prev_keys[-3:]: # Search last 3 successful frames
            mts = flann_match(desc[pv], desc[i], RATIO_INCR)
            for m in mts:
                if m.queryIdx in k2p[pv]:
                    pi = k2p[pv][m.queryIdx]
                    if pi < len(map_pts) and pi not in [t[1] for t in trk]:
                        p2d.append(kps[i][m.trainIdx].pt)
                        p3d.append(map_pts[pi])
                        trk.append((m.trainIdx, pi, pv))
                        
        if len(p3d) < MIN_PNP:
            print(f"  Frame {i+1}: skipped (Not enough overlap)")
            continue

        ok, rv, tv, inl = cv2.solvePnPRansac(np.array(p3d, dtype=np.float64), np.array(p2d, dtype=np.float64),
                                             K, None, reprojectionError=8.0, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < MIN_PNP:
            continue

        Ri, ti_vec = cv2.Rodrigues(rv)[0], tv.reshape(3, 1)
        
        # Task 3: Refine estimated PnP Pose
        p2d_inl, p3d_inl = np.array(p2d)[inl.ravel()], np.array(p3d)[inl.ravel()]
        Ri, ti_vec = refine_single_pose(Ri, ti_vec, p3d_inl, p2d_inl, K)
        cams_reg[i] = (Ri, ti_vec)
        
        iset = set(inl.ravel())
        for j, (kpi, mpi, pv) in enumerate(trk):
            if j in iset:
                k2p[i][kpi] = mpi
                obs_c.append(i); obs_p.append(mpi); obs_uv.append(kps[i][kpi].pt)

        # Map Expansion (Triangulating new points)
        pv = prev_keys[-1]
        mts_all = flann_match(desc[pv], desc[i], RATIO_INCR)
        unm = [m for m in mts_all if m.queryIdx not in k2p[pv]]
        
        if unm:
            Rp, tp = cams_reg[pv]
            Pp_mat, Pi_mat = K @ np.hstack([Rp, tp]), K @ np.hstack([Ri, ti_vec])
            up = np.float32([kps[pv][m.queryIdx].pt for m in unm])
            ui = np.float32([kps[i][m.trainIdx].pt for m in unm])
            
            n3d = triangulate(Pp_mat, Pi_mat, up, ui)
            pos_mask = n3d[:, 2] > 0
            
            if pos_mask.sum() >= 4:
                n3d_pos = n3d[pos_mask]
                vm2 = outlier_mask(n3d_pos)
                n3d_cl = n3d_pos[vm2]
                unm_cl = [m for m, v in zip([m for m, v in zip(unm, pos_mask) if v], vm2) if v]
                
                base = len(map_pts)
                map_pts.extend(list(n3d_cl))
                
                # TASK 3: Store descriptors for new map points
                new_desc = np.array([desc[i][m.trainIdx] for m in unm_cl])
                map_desc = np.vstack([map_desc, new_desc])
                
                for j2, m in enumerate(unm_cl):
                    pi = base + j2
                    k2p[pv][m.queryIdx] = pi; k2p[i][m.trainIdx] = pi
                    obs_c += [pv, i]; obs_p += [pi, pi]
                    obs_uv += [kps[pv][m.queryIdx].pt, kps[i][m.trainIdx].pt]

        active = [cams_reg[k] for k in sorted(cams_reg)]
        print(f"  Frame {i+1}: {len(active)} cams, {len(map_pts)} pts")

        # ==========================================
        # PERIODIC LOCAL BUNDLE ADJUSTMENT
        # ==========================================
        # if len(active) % 10 == 0:
        #     print(f"  [Local BA] Stabilizing map...")
        #     s_keys = sorted(cams_reg.keys())
        #     c_rmp = {k: idx for idx, k in enumerate(s_keys)}
        #     u_pts = sorted(set(obs_p))
        #     p_rmp = {old: new for new, old in enumerate(u_pts)}
            
        #     filt = [(ci, pi, uv) for ci, pi, uv in zip(obs_c, obs_p, obs_uv) if ci in c_rmp and pi in p_rmp]
        #     o_c, o_p, o_u = [c_rmp[ci] for ci,_,_ in filt], [p_rmp[pi] for _,pi,_ in filt], [uv for _,_,uv in filt]
            
        #     c_curr = [cams_reg[k] for k in s_keys]
        #     p_curr = np.array(map_pts)[u_pts]
            
        #     n_cams, n_pts = run_ba(c_curr, p_curr, o_c, o_p, o_u, K, verbose=0, max_iter=20)
            
        #     for idx, key in enumerate(s_keys): cams_reg[key] = n_cams[idx]
        #     for n_idx, o_idx in enumerate(u_pts): map_pts[o_idx] = n_pts[n_idx]

    # ------------------------------------------
    # TASK 4: Global Bundle Adjustment
    # ------------------------------------------
    sorted_keys = sorted(cams_reg.keys())
    c_remap = {k: idx for idx, k in enumerate(sorted_keys)}
    used_pts = sorted(set(obs_p))
    p_remap = {old: new for new, old in enumerate(used_pts)}

    filt = [(ci, pi, uv) for ci, pi, uv in zip(obs_c, obs_p, obs_uv) if ci in c_remap and pi in p_remap]
    obs_c_ba, obs_p_ba, obs_uv_ba = [c_remap[ci] for ci,_,_ in filt], [p_remap[pi] for _,pi,_ in filt], [uv for _,_,uv in filt]

    ba_cams = [cams_reg[k] for k in sorted_keys]
    ba_pts = np.array(map_pts)[used_pts]
    ba_desc = map_desc[used_pts] # Filter descriptors to match used points

    pre_errs = [np.linalg.norm(reproject(K, ba_cams[ci][0], ba_cams[ci][1], ba_pts[pi:pi+1])[0] - np.array(uv))
                for ci, pi, uv in zip(obs_c_ba, obs_p_ba, obs_uv_ba)]
    print(f"\n[Task 4] Global BA Pre-Error: {np.mean(pre_errs):.4f}px")

    print("  Running Global Bundle Adjustment (this may take a minute)...")
    new_ba_cams, new_ba_pts = run_ba(ba_cams, ba_pts, obs_c_ba, obs_p_ba, obs_uv_ba, K, verbose=2, max_iter=40)

    post_errs = [np.linalg.norm(reproject(K, new_ba_cams[ci][0], new_ba_cams[ci][1], new_ba_pts[pi:pi+1])[0] - np.array(uv))
                 for ci, pi, uv in zip(obs_c_ba, obs_p_ba, obs_uv_ba)]
    print(f"  Global BA Post-Error: {np.mean(post_errs):.4f}px")

    vm_f = outlier_mask(new_ba_pts)
    final_pts = new_ba_pts[vm_f]
    final_desc = ba_desc[vm_f]
    print(f"  Cleaned Map Points: {len(final_pts)}")

    if ply_path and os.path.exists(ply_path):
        print("\n[Task 4] Evaluating against Ground Truth...")
        gt_raw = np.asarray(o3d.io.read_point_cloud(ply_path).points)
        
        # Zero-center and normalize both clouds using the SAME scale (Per Assignment constraints)
        gs = np.max(np.linalg.norm(gt_raw - gt_raw.mean(0), axis=1))
        rs = np.max(np.linalg.norm(final_pts - final_pts.mean(0), axis=1))
        scale = (gs + rs) / 2.0
        
        gt_n = (gt_raw - gt_raw.mean(0)) / scale
        rec_n = (final_pts - final_pts.mean(0)) / scale
        
        cd = chamfer_distance(rec_n, gt_n)
        print(f"  Final Chamfer Distance: {cd:.6f}")

        # Qualitative Visualization Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
        axes[0].scatter(rec_n[:, 0], rec_n[:, 1], rec_n[:, 2], s=0.3, c='green', alpha=0.5)
        axes[0].set_title("Normalized Reconstructed Point Cloud")
        
        step = max(1, len(gt_n) // 5000) # Subsample GT for plotting speed
        axes[1].scatter(gt_n[::step, 0], gt_n[::step, 1], gt_n[::step, 2], s=0.3, c='blue', alpha=0.5)
        axes[1].set_title("Normalized Ground Truth Point Cloud")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/task4_comparison.png", dpi=100)
        plt.close()
        print(f"  Comparison plot saved to {out_dir}/task4_comparison.png")

    # ------------------------------------------
    # 6. EXPORT DELIVERABLES (Task 6 prep & Ply)
    # ------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    o3d.io.write_point_cloud(f"{out_dir}/reconstructed_cloud.ply", pcd)
    
    # Save Custom Map specifically for Task 6
    np.savez(f"{out_dir}/custom_map_task6.npz", 
             pts3d=final_pts, 
             desc3d=final_desc, 
             cams=np.array([(cv2.Rodrigues(R)[0], t) for R, t in new_ba_cams], dtype=object))
             
    save_map_plot(final_pts, new_ba_cams, f"Final Map", f"{out_dir}/final_reconstruction.png")
    print(f"\n✅ Pipeline Complete! Deliverables saved to: {out_dir}/")
    return final_pts, new_ba_cams, K

if __name__ == "__main__":
    run_pipeline('split_a_meetingroom-001.mp4', 'Meetingroom.ply')