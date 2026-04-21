import json
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def visualize_batch_grid(images_input, num_cols=5, axes_pad=0.1, filename="batch_grid.png", show=True):
    """Renders a batch of images in a Matplotlib ImageGrid."""
    if torch.is_tensor(images_input):
        images = images_input.detach().float().clamp(0, 1).cpu().numpy()
    else:
        images = images_input

    num_views = len(images)
    nrows, ncols = math.ceil(num_views / num_cols), num_cols
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)

    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metrics(history: dict, filename="training_metrics.png", show=True):
    """Graphs the Mask Loss and Smoothness Loss over the iterations."""
    iters = range(len(history['total_loss']))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Convert tensor values to numpy, moving to CPU if needed
    mask_loss_vals = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in history['mask_loss']]
    smooth_loss_vals = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in history['smooth_loss']]

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mask BCE Loss', color=color)
    ax1.plot(iters, mask_loss_vals, color=color, label='Mask Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Smoothness Loss', color=color)
    ax2.plot(iters, smooth_loss_vals, color=color, label='Smoothness', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Voxel Optimization Metrics')
    fig.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_2d_mask_metrics(phi_grid, seg_result, cams, args, device, threshold: float = 0.5) -> dict:
    """Render phi_grid masks at each SAM-accepted camera and compare against
    seg_result.masks. Returns per-view + mean precision/recall/F1/IoU.

    Assumes cams has already been filtered to valid_indices, so
    len(cams.viewmats) == len(seg_result.masks).
    """
    num_views = len(cams.viewmats)
    H, W = args.height, args.width
    per_view = []

    with torch.no_grad():
        for i in range(num_views):
            pred = phi_grid.render_mask(i, cams, args.num_samples).view(H, W)
            pred_bin = (pred >= threshold)
            target = torch.from_numpy(seg_result.masks[i]).to(device=device, dtype=torch.bool)

            tp = (pred_bin & target).sum().item()
            fp = (pred_bin & ~target).sum().item()
            fn = (~pred_bin & target).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

            per_view.append({
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            })

    def _mean(key: str) -> float:
        return sum(v[key] for v in per_view) / num_views if num_views > 0 else 0.0

    mean = {
        "precision": _mean("precision"),
        "recall": _mean("recall"),
        "f1": _mean("f1"),
        "iou": _mean("iou"),
    }
    return {
        "mean": mean,
        "per_view": per_view,
        "num_views": num_views,
        "threshold": threshold,
    }


def load_off_mesh(path: str):
    """Load a ModelNet-style .off mesh. Returns (verts (V,3), faces (F,3)).

    Only handles triangle / quad faces (quads are triangulated as fans).
    """
    with open(path, "r") as f:
        header = f.readline().strip()
        if header.startswith("OFF") and len(header) > 3:
            # Some ModelNet files have "OFF" glued to the counts line
            counts_line = header[3:].strip()
        else:
            if header != "OFF":
                raise ValueError(f"{path}: expected OFF header, got '{header}'")
            counts_line = f.readline().strip()
        parts = counts_line.split()
        n_verts, n_faces = int(parts[0]), int(parts[1])

        verts = np.empty((n_verts, 3), dtype=np.float64)
        for i in range(n_verts):
            verts[i] = [float(x) for x in f.readline().split()[:3]]

        faces = []
        for _ in range(n_faces):
            tokens = f.readline().split()
            k = int(tokens[0])
            idx = [int(x) for x in tokens[1:1 + k]]
            for j in range(1, k - 1):
                faces.append([idx[0], idx[j], idx[j + 1]])
        faces = np.asarray(faces, dtype=np.int64)

    return verts, faces


def sample_points_on_mesh(verts: np.ndarray, faces: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    """Area-weighted uniform sampling of points on triangle mesh surface."""
    rng = np.random.default_rng(seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("Mesh has zero total surface area")
    probs = areas / total
    face_idx = rng.choice(len(faces), size=num_points, p=probs)

    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    a = 1.0 - sqrt_r1
    b = sqrt_r1 * (1.0 - r2)
    c = sqrt_r1 * r2

    return (a[:, None] * v0[face_idx] + b[:, None] * v1[face_idx] + c[:, None] * v2[face_idx])


def _normalize_to_unit(points: np.ndarray):
    """Center at centroid and scale so max distance from centroid equals 1.
    Returns (normalized, center, scale)."""
    center = points.mean(axis=0)
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        scale = 1.0
    return centered / scale, center, scale


def _best_fit_rigid(src: np.ndarray, dst: np.ndarray):
    """Kabsch: find R, t minimizing ||R@src + t - dst||^2 over matched pairs."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    return R, t


def _icp(src: np.ndarray, dst: np.ndarray, max_iters: int = 30, tol: float = 1e-6) -> np.ndarray:
    """Rigid ICP (no scale). Returns src aligned to dst."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    current = src.copy()
    prev_err = np.inf
    for _ in range(max_iters):
        dists, nn_idx = tree.query(current, k=1)
        R, t = _best_fit_rigid(current, dst[nn_idx])
        current = current @ R.T + t
        err = float(np.mean(dists))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return current


def compute_chamfer(pred_points: np.ndarray, gt_points: np.ndarray, align: bool = True) -> dict:
    """Symmetric Chamfer distance between two point clouds.

    Both clouds are centered and unit-scaled before comparison; when align=True
    the pred cloud is then further aligned to GT with rigid ICP. Metrics are
    reported in the unit-normalized frame (scale-invariant, comparable across
    objects with different physical sizes).
    """
    from scipy.spatial import cKDTree

    pred_n, _, _ = _normalize_to_unit(pred_points)
    gt_n, _, _ = _normalize_to_unit(gt_points)

    if align:
        pred_n = _icp(pred_n, gt_n)

    tree_gt = cKDTree(gt_n)
    tree_pred = cKDTree(pred_n)
    d_pred_to_gt, _ = tree_gt.query(pred_n, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_n, k=1)

    chamfer_l2 = float(d_pred_to_gt.mean() + d_gt_to_pred.mean())
    chamfer_l2_sq = float((d_pred_to_gt ** 2).mean() + (d_gt_to_pred ** 2).mean())

    # F-score at fixed thresholds (fraction of points within tau of the other cloud)
    fscores = {}
    for tau in (0.01, 0.02, 0.05):
        precision = float((d_pred_to_gt < tau).mean())
        recall = float((d_gt_to_pred < tau).mean())
        f = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fscores[f"f_at_{tau}"] = f

    return {
        "chamfer_l2": chamfer_l2,
        "chamfer_l2_squared": chamfer_l2_sq,
        "num_pred_points": int(len(pred_points)),
        "num_gt_points": int(len(gt_points)),
        "aligned": bool(align),
        **fscores,
    }


def compute_geometry_metrics(phi_grid, cams, gt_mesh_path: str, num_samples: int = 100_000, align: bool = True) -> dict:
    """Extract phi_grid mesh, sample both predicted and GT surfaces, compute Chamfer."""
    mesh = phi_grid.extract_mesh(cams)
    if mesh is None:
        return {"error": "phi_grid.extract_mesh returned None (no isosurface)"}
    verts_pred, faces_pred, _ = mesh

    pred_points = sample_points_on_mesh(verts_pred, faces_pred, num_samples)

    gt_verts, gt_faces = load_off_mesh(gt_mesh_path)
    gt_points = sample_points_on_mesh(gt_verts, gt_faces, num_samples, seed=1)

    metrics = compute_chamfer(pred_points, gt_points, align=align)
    metrics["gt_mesh"] = gt_mesh_path
    return metrics


def save_2d_mask_metrics(metrics: dict, out_path: str) -> None:
    """Write the metrics dict to JSON and print a short summary."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    m = metrics["mean"]
    print(f"[Metrics] Saved to {out_path}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  IoU:       {m['iou']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  Views:     {metrics['num_views']}")