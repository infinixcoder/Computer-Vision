import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
import open3d as o3d
import os, glob, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

FRAME_SKIP  = 20
RATIO_INIT  = 0.75
RATIO_INCR  = 0.80
OUTLIER_K   = 5.0
MIN_PNP     = 8
MAX_BA_OBS  = 50000


def pack_keypoints(keypoints):
    """Converts OpenCV KeyPoints into a dense numpy array for saving."""
    if not keypoints:
        return np.empty((0, 7), dtype=np.float32)
    return np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                     for kp in keypoints], dtype=np.float32)

def unpack_keypoints(kp_array):
    """Rebuilds OpenCV KeyPoints from loaded numpy arrays."""
    if kp_array.size == 0:
        return []
    return [cv2.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3], 
                         response=row[4], octave=int(row[5]), class_id=int(row[6])) 
            for row in kp_array]

def process_video_on_the_fly(path, skip):
    name = os.path.splitext(os.path.basename(path))[0]
    cache_file = f"cache_sift_{name}_skip{skip}.npz"  # Switched to .npz
    
    # --- CHECK FOR CACHE FIRST ---
    if os.path.exists(cache_file):
        print(f"  [Cache] Found cached features! Loading {cache_file}...")
        data = np.load(cache_file)
        K = data['K']
        
        # Count how many frames we saved by counting 'desc_X' keys
        num_frames = sum(1 for k in data.files if k.startswith('desc_'))
        
        kps_list = [unpack_keypoints(data[f'kp_{i}']) for i in range(num_frames)]
        desc_list = [data[f'desc_{i}'] for i in range(num_frames)]
        
        print(f"  [Cache] Loaded SIFT features for {len(kps_list)} frames.")
        return K, kps_list, desc_list

    # --- IF NO CACHE, COMPUTE FROM SCRATCH ---
    cap = cv2.VideoCapture(path)
    det = cv2.SIFT_create()
    kps_list, desc_list = [], []
    K = None
    n = 0

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Starting extraction from {total_frames} total frames (using FRAME_SKIP={skip})...")
    
    while cap.grab():
        if n % skip == 0:
            ret, f = cap.retrieve()
            if ret and f is not None:
                # Build K using the very first valid frame
                if K is None:
                    h, w = f.shape[:2]
                    f_max = max(h, w)
                    K = np.array([[f_max, 0, w / 2],
                                  [0, f_max, h / 2],
                                  [0, 0,     1]], dtype=np.float64)
                
                # Extract SIFT and immediately discard the frame
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                kp, d = det.detectAndCompute(g, None)
                kps_list.append(kp)
                desc_list.append(d if d is not None else np.zeros((0, 128), np.float32))
                
                # Print progress dynamically on the same line
                print(f"\r  Scanning frame {n}/{total_frames} | Kept SIFT descriptors for {len(kps_list)} frames", end="", flush=True)
        n += 1
        
    cap.release()
    print("\n  Extraction complete!") # Move to a new line once done

    # --- SAVE TO CACHE FOR NEXT TIME ---
    print(f"  [Cache] Saving computed features to {cache_file} (compressing)...")
    
    # Pack into a dictionary where each frame gets its own key
    save_dict = {'K': K}
    for i, (kps, desc) in enumerate(zip(kps_list, desc_list)):
        save_dict[f'kp_{i}'] = pack_keypoints(kps)
        save_dict[f'desc_{i}'] = desc
        
    # np.savez_compressed writes arrays iteratively, avoiding memory spikes
    np.savez_compressed(cache_file, **save_dict)

    return K, kps_list, desc_list


def build_K(frame):
    h, w = frame.shape[:2]
    f = max(h, w)
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0,     1]], dtype=np.float64)



def flann_match(d1, d2, ratio):
    if len(d1) < 2 or len(d2) < 2:
        return []
    fl = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    raw = fl.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    good = []
    for pair in raw:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def triangulate(P1, P2, pts1, pts2):
    h = cv2.triangulatePoints(P1, P2,
                               pts1.T.astype(np.float64),
                               pts2.T.astype(np.float64))
    return (h[:3] / h[3]).T


def outlier_mask(pts3d):
    if len(pts3d) < 4:
        return np.ones(len(pts3d), dtype=bool)
    c = np.median(pts3d, axis=0)
    d = np.linalg.norm(pts3d - c, axis=1)
    return d < OUTLIER_K * np.median(d)


def reproject(K, R, t, X):
    x = K @ (R @ X.T + t)
    return (x[:2] / x[2]).T


def mean_repr_error(K, cams_dict, pts3d_arr, obs_c, obs_p, obs_uv, c_remap, p_remap):
    errs = []
    for ci, pi, uv in zip(obs_c, obs_p, obs_uv):
        if ci not in c_remap or pi not in p_remap:
            continue
        R, t = cams_dict[ci]
        proj = reproject(K, R, t, pts3d_arr[p_remap[pi]:p_remap[pi]+1])[0]
        errs.append(np.linalg.norm(proj - np.array(uv)))
    return float(np.mean(errs)) if errs else 0.0


def ba_residuals(x, nc, obs_c, obs_p, obs_uv, K):
    cp  = x[:nc * 6].reshape(nc, 6)
    pts = x[nc * 6:].reshape(-1, 3)
    Rs  = [(cv2.Rodrigues(cp[i, :3])[0], cp[i, 3:6].reshape(3, 1)) for i in range(nc)]
    r   = []
    for ci, pi, uv in zip(obs_c, obs_p, obs_uv):
        R, t = Rs[ci]
        proj = reproject(K, R, t, pts[pi:pi+1])[0]
        r   += list(proj - uv)
    return np.array(r)


def build_sparsity(nc, np_, obs_c, obs_p):
    A = lil_matrix((2 * len(obs_c), nc * 6 + np_ * 3), dtype=int)
    for i, (ci, pi) in enumerate(zip(obs_c, obs_p)):
        A[2*i:2*i+2, ci*6:(ci+1)*6] = 1
        A[2*i:2*i+2, nc*6+pi*3:nc*6+(pi+1)*3] = 1
    return A


def run_ba(cams_list, pts3d, obs_c, obs_p, obs_uv, K):
    nc, np_ = len(cams_list), len(pts3d)
    x0 = []
    for R, t in cams_list:
        rv, _ = cv2.Rodrigues(R)
        x0 += list(rv.ravel()) + list(t.ravel())
    x0 += list(pts3d.ravel())
    x0  = np.array(x0)
    A   = build_sparsity(nc, np_, obs_c, obs_p)
    res = least_squares(ba_residuals, x0, jac_sparsity=A,
                        args=(nc, obs_c, obs_p, obs_uv, K),
                        method='trf', loss='huber', max_nfev=150, ftol = 1e-3, xtol = 1e-3, verbose=2)
    cp = res.x[:nc * 6].reshape(nc, 6)
    new_cams = [(cv2.Rodrigues(cp[i, :3])[0], cp[i, 3:6].reshape(3, 1)) for i in range(nc)]
    new_pts  = res.x[nc * 6:].reshape(np_, 3)
    return new_cams, new_pts


def chamfer_distance(A, B, n=8000):
    rng = np.random.default_rng(42)
    if len(A) > n: A = A[rng.choice(len(A), n, replace=False)]
    if len(B) > n: B = B[rng.choice(len(B), n, replace=False)]
    d_ab = cKDTree(B).query(A, k=1)[0].mean()
    d_ba = cKDTree(A).query(B, k=1)[0].mean()
    return (d_ab + d_ba) / 2.0


def save_map_plot(pts3d, cams_list, title, path):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    p   = np.array(pts3d)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.4, c='green', alpha=0.5)
    for R, t in cams_list:
        pos = (-R.T @ t).ravel()
        ax.scatter(*pos, c='red', s=50, marker='^', zorder=5)
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def run_pipeline(video_path, ply_path=None):
    name    = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = f"results_{name}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*55}\nScene : {name}\n{'='*55}")

# Process everything on the fly to save RAM
    K, kps_list, desc = process_video_on_the_fly(video_path, FRAME_SKIP)
    NF = len(kps_list)
    print(f"  Frames Processed: {NF}")
    
    if NF < 2:
        print("  Not enough frames — try reducing FRAME_SKIP.")
        return None

    m01 = flann_match(desc[0], desc[1], RATIO_INIT)
    if len(m01) < 10:
        print("  Not enough initial matches.")
        return None

    p0r = np.float32([kps_list[0][m.queryIdx].pt for m in m01])
    p1r = np.float32([kps_list[1][m.trainIdx].pt for m in m01])

    E, mE = cv2.findEssentialMat(p0r, p1r, K, cv2.RANSAC, 0.999, 1.0)
    mE    = mE.ravel().astype(bool)
    p0, p1, m01f = p0r[mE], p1r[mE], [m for m, v in zip(m01, mE) if v]

    _, R1, t1, mC = cv2.recoverPose(E, p0, p1, K)
    mC = mC.ravel() > 0
    p0, p1, m01f = p0[mC], p1[mC], [m for m, v in zip(m01f, mC) if v]

    R0, t0   = np.eye(3), np.zeros((3, 1))
    P0_mat   = K @ np.hstack([R0, t0])
    P1_mat   = K @ np.hstack([R1, t1])
    i3d      = triangulate(P0_mat, P1_mat, p0, p1)
    vm       = outlier_mask(i3d)
    i3d      = i3d[vm]
    p0       = p0[vm]
    p1       = p1[vm]  
    m01f     = [m for m, v in zip(m01f, vm) if v]

    map_pts  = list(i3d)
    cams_reg = {0: (R0, t0), 1: (R1, t1)}
    k2p      = [{} for _ in range(NF)]
    obs_c, obs_p, obs_uv = [], [], []

    for idx, m in enumerate(m01f):
        k2p[0][m.queryIdx] = idx
        k2p[1][m.trainIdx] = idx
        obs_c  += [0, 1]
        obs_p  += [idx, idx]
        obs_uv += [kps_list[0][m.queryIdx].pt, kps_list[1][m.trainIdx].pt]

    e0 = np.mean(np.linalg.norm(reproject(K, R1, t1, i3d) - p1, axis=1))
    print(f"  Init: {len(map_pts)} pts, MRE={e0:.4f}px")
    save_map_plot(map_pts, [cams_reg[0], cams_reg[1]],
                  f"Incremental Map (Frame 2): 2 Cameras, {len(map_pts)} Points",
                  f"{out_dir}/map_frame_02.png")

    for i in range(2, NF):
        prev_keys = sorted(k for k in cams_reg if k < i)
        if not prev_keys:
            continue
        pv    = prev_keys[-1]
        ratio = RATIO_INCR if i >= 5 else RATIO_INIT
        mts   = flann_match(desc[pv], desc[i], ratio)

        p2d, p3d, trk, unm = [], [], [], []
        for m in mts:
            qi = m.queryIdx
            ti = m.trainIdx
            if qi in k2p[pv]:
                pi = k2p[pv][qi]
                if pi < len(map_pts):
                    p2d.append(kps_list[i][ti].pt)
                    p3d.append(map_pts[pi])
                    trk.append((ti, pi))
                    continue
            unm.append(m)

        if len(p3d) < MIN_PNP:
            print(f"  Frame {i+1}: skip ({len(p3d)} 2D-3D pairs)")
            continue

        ok, rv, tv, inl = cv2.solvePnPRansac(
            np.array(p3d, dtype=np.float64),
            np.array(p2d, dtype=np.float64),
            K, None,
            iterationsCount=300,
            reprojectionError=4.0,
            flags=cv2.SOLVEPNP_SQPNP)

        if not ok or inl is None or len(inl) < MIN_PNP:
            print(f"  Frame {i+1}: PnP failed")
            continue

        Ri, ti_vec = cv2.Rodrigues(rv)[0], tv.reshape(3, 1)
        cams_reg[i] = (Ri, ti_vec)
        iset        = set(inl.ravel())

        for j, (kpi, mpi) in enumerate(trk):
            if j in iset:
                k2p[i][kpi] = mpi
                obs_c.append(i); obs_p.append(mpi)
                obs_uv.append(kps_list[i][kpi].pt)

        Rp, tp   = cams_reg[pv]
        Pp_mat   = K @ np.hstack([Rp, tp])
        Pi_mat   = K @ np.hstack([Ri, ti_vec])

        if unm:
            up  = np.float32([kps_list[pv][m.queryIdx].pt for m in unm])
            ui  = np.float32([kps_list[i][m.trainIdx].pt  for m in unm])
            n3d = triangulate(Pp_mat, Pi_mat, up, ui)

            pos_mask = n3d[:, 2] > 0
            if pos_mask.sum() >= 4:
                n3d_pos  = n3d[pos_mask]
                vm2      = outlier_mask(n3d_pos)
                n3d_cl   = n3d_pos[vm2]
                unm_cl   = [m for m, v in zip(
                                [m for m, v in zip(unm, pos_mask) if v], vm2) if v]
                base     = len(map_pts)
                map_pts.extend(list(n3d_cl))
                for j2, m in enumerate(unm_cl):
                    pi = base + j2
                    k2p[pv][m.queryIdx] = pi
                    k2p[i][m.trainIdx]  = pi
                    obs_c  += [pv, i]
                    obs_p  += [pi, pi]
                    obs_uv += [kps_list[pv][m.queryIdx].pt, kps_list[i][m.trainIdx].pt]

        active = [cams_reg[k] for k in sorted(cams_reg)]
        print(f"  Frame {i+1}: {len(active)} cams, {len(map_pts)} pts")
        save_map_plot(map_pts, active,
                      f"Incremental Map (Frame {i+1}): {len(active)} Cameras, {len(map_pts)} Points",
                      f"{out_dir}/map_frame_{i+1:02d}.png")

    sorted_keys = sorted(cams_reg.keys())
    c_remap     = {k: idx for idx, k in enumerate(sorted_keys)}
    used_pts    = sorted(set(obs_p))
    p_remap     = {old: new for new, old in enumerate(used_pts)}

    filt = [(ci, pi, uv) for ci, pi, uv in zip(obs_c, obs_p, obs_uv)
            if ci in c_remap and pi in p_remap]
    if len(filt) > MAX_BA_OBS:
        idx  = np.random.default_rng(0).choice(len(filt), MAX_BA_OBS, replace=False)
        filt = [filt[k] for k in sorted(idx)]

    obs_c_ba  = [c_remap[ci] for ci, _, _  in filt]
    obs_p_ba  = [p_remap[pi] for _, pi, _  in filt]
    obs_uv_ba = [uv          for _, _,  uv in filt]

    ba_cams = [cams_reg[k] for k in sorted_keys]
    ba_pts  = np.array(map_pts)[used_pts]

    pre_errs = [np.linalg.norm(
                    reproject(K, ba_cams[ci][0], ba_cams[ci][1], ba_pts[pi:pi+1])[0]
                    - np.array(uv))
                for ci, pi, uv in zip(obs_c_ba, obs_p_ba, obs_uv_ba)]
    print(f"\n  Mean Reprojection Error (Before BA): {np.mean(pre_errs):.4f} pixels")

    print("  Running Bundle Adjustment ...")
    new_ba_cams, new_ba_pts = run_ba(ba_cams, ba_pts, obs_c_ba, obs_p_ba, obs_uv_ba, K)

    post_errs = [np.linalg.norm(
                     reproject(K, new_ba_cams[ci][0], new_ba_cams[ci][1], new_ba_pts[pi:pi+1])[0]
                     - np.array(uv))
                 for ci, pi, uv in zip(obs_c_ba, obs_p_ba, obs_uv_ba)]
    print(f"  Mean Reprojection Error (After BA):  {np.mean(post_errs):.4f} pixels")

    vm_f    = outlier_mask(new_ba_pts)
    final_pts = new_ba_pts[vm_f]
    print(f"  Cleaned Reconstruction Points: {len(final_pts)}")

    if ply_path and os.path.exists(ply_path):
        gt_raw = np.asarray(o3d.io.read_point_cloud(ply_path).points)
        gs     = np.max(np.linalg.norm(gt_raw  - gt_raw.mean(0),  axis=1))
        rs     = np.max(np.linalg.norm(final_pts - final_pts.mean(0), axis=1))
        scale  = (gs + rs) / 2.0
        print(f"  Normalizing BOTH clouds by global scale: {scale:.4f}")
        gt_n  = (gt_raw   - gt_raw.mean(0))   / gs
        rec_n = (final_pts - final_pts.mean(0)) / rs
        cd    = chamfer_distance(rec_n, gt_n)
        print(f"  Final Rule-Compliant Chamfer Distance: {cd:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
        axes[0].scatter(rec_n[:, 0], rec_n[:, 1], rec_n[:, 2], s=0.3, c='green', alpha=0.5)
        axes[0].set_title("Normalized Reconstructed Point Cloud")
        step = max(1, len(gt_n) // 5000)
        axes[1].scatter(gt_n[::step, 0], gt_n[::step, 1], gt_n[::step, 2], s=0.3, c='blue', alpha=0.5)
        axes[1].set_title("Normalized Ground Truth Point Cloud")
        plt.tight_layout()
        cmp_path = f"{out_dir}/comparison.png"
        plt.savefig(cmp_path, dpi=100)
        plt.close()
        print(f"    Saved: {cmp_path}")

    final_path = f"{out_dir}/final_reconstruction.png"
    save_map_plot(final_pts, new_ba_cams,
                  f"Final Reconstruction: {len(new_ba_cams)} Cameras, {len(final_pts)} Points",
                  final_path)
    return final_pts, new_ba_cams, K


if __name__ == "__main__":
    # exts  = ["*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV"]
    # vids  = []
    # for pat in exts:
    #     vids += glob.glob(f"*split_a*{pat[1:]}") + glob.glob(pat)
    # vids  = sorted(set(vids))
    # plys  = sorted(glob.glob("*.ply"))

    vids = ['split_a_truck-004.mp4']
    plys = ['Truck.ply']

    if not vids:
        print("No video files found in current directory.")
        sys.exit(1)

    seen = set()
    for i, v in enumerate(vids):
        if v in seen:
            continue
        seen.add(v)
        ply = plys[i] if i < len(plys) else None
        run_pipeline(v, ply)