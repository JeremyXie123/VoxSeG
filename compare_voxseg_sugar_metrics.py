#!/usr/bin/env python3
"""
Compare VoxSeG vs a SuGaR-centers baseline (Poisson reconstruction from Gaussian centres).

NOTE: This is NOT full SuGaR. Full SuGaR (the CVPR 2024 paper pipeline) needs a
COLMAP source dataset (images + sparse reconstruction) plus a trained vanilla
3DGS checkpoint directory with cameras.json. The `splats/{airplane,car,guitar,
toilet,bathtub}/train/*/point_cloud.ply` files in this repo are valid 3DGS PLYs
but ship with no images and no COLMAP data, and `splats/archive/` only contains
ModelNet40 `.off` ground-truth meshes — so the full SuGaR extractor cannot be
invoked on them. See the top-level comparison discussion for options (rendering
views from ModelNet40 meshes, OmniObject3D, etc.).

What this script DOES do is reproduce SuGaR's `--use_centers_to_extract_mesh`
path standalone: take Gaussian means, prune by opacity, estimate normals, run
Screened Poisson Surface Reconstruction, and compute Chamfer + F-scores
against ModelNet40 ground-truth meshes for the five categories used in VoxSeG.
This is the simplest SuGaR-family baseline and a reasonable approximation when
the full image-in-the-loop pipeline is not available, but numbers here are
NOT paper-comparable to full SuGaR.

Usage:
    python compare_voxseg_sugar_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from core.splat_io import load_ply
from stages.evaluation import (
    load_off_mesh,
    sample_points_on_mesh,
    compute_chamfer,
)

MODELNET_ROOT = "splats/archive/ModelNet40"

OBJECTS = {
    "airplane": {
        "splat": "splats/airplane/train/airplane_0001/point_cloud.ply",
        "gt": f"{MODELNET_ROOT}/airplane/train/airplane_0001.off",
    },
    "car": {
        "splat": "splats/car/train/car_0001/point_cloud.ply",
        "gt": f"{MODELNET_ROOT}/car/train/car_0001.off",
    },
    "guitar": {
        "splat": "splats/guitar/train/guitar_0001/point_cloud.ply",
        "gt": f"{MODELNET_ROOT}/guitar/train/guitar_0001.off",
    },
    "bathtub": {
        "splat": "splats/bathtub/train/bathtub_0001/point_cloud.ply",
        "gt": f"{MODELNET_ROOT}/bathtub/train/bathtub_0001.off",
    },
    "toilet": {
        "splat": "splats/toilet/train/toilet_0001/point_cloud.ply",
        "gt": f"{MODELNET_ROOT}/toilet/train/toilet_0001.off",
    },
}

NUM_SAMPLES = 100_000
OPACITY_THRESHOLD = 0.5  # SuGaR uses 0.5 in low_opacity_gaussian_pruning_threshold
POISSON_DEPTH = 7  # SuGaR uses 10 for real, 6-7 for synthetic scenes
DENSITY_QUANTILE = 0.0  # SuGaR recommends 0.0 for synthetic scenes


def sugar_centers_mesh_from_gaussians(
    means: np.ndarray,
    poisson_depth: int = POISSON_DEPTH,
    density_quantile: float = DENSITY_QUANTILE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Reproduce SuGaR's center-based mesh extraction:
    1. Build point cloud from Gaussian centres
    2. Estimate normals
    3. Run Screened Poisson Surface Reconstruction
    4. Prune low-density vertices
    5. Return (vertices, faces)
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))

    # Estimate normals (SuGaR uses sugar.get_normals(estimate_from_points=True))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)

    # Statistical outlier removal (SuGaR: nb_neighbors=20, std_ratio=20)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)

    # Screened Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    if density_quantile > 0.0:
        vertices_to_remove = densities < np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Clean mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if len(verts) < 100 or len(faces) < 100:
        return None

    return verts, faces


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voxseg_results = {}
    sugar_results = {}

    for name, paths in OBJECTS.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        # ---- Load VoxSeG metrics (already computed) ----
        voxseg_file = f"logs/{name}/mask_metrics.json"
        if os.path.isfile(voxseg_file):
            with open(voxseg_file) as f:
                data = json.load(f)
            voxseg_results[name] = {
                "f1": data["mean"]["f1"],
                "iou": data["mean"]["iou"],
                "chamfer_l2": data["geometry"]["chamfer_l2"],
                "f_at_0.02": data["geometry"]["f_at_0.02"],
                "f_at_0.05": data["geometry"]["f_at_0.05"],
            }
            print(f"  VoxSeG — Chamfer: {voxseg_results[name]['chamfer_l2']:.4f}, "
                  f"F@0.05: {voxseg_results[name]['f_at_0.05']:.4f}")
        else:
            print(f"  WARNING: VoxSeG metrics not found at {voxseg_file}")
            voxseg_results[name] = None

        # ---- SuGaR-style baseline ----
        splats = load_ply(paths["splat"], device)
        n_total = splats.means.shape[0]
        print(f"  Total Gaussians: {n_total:,}")

        # Filter by opacity (SuGaR prunes at 0.5)
        mask = splats.opacities > OPACITY_THRESHOLD
        means_np = splats.means[mask].detach().cpu().numpy()
        n_kept = means_np.shape[0]
        print(f"  After opacity > {OPACITY_THRESHOLD}: {n_kept:,} ({100*n_kept/n_total:.1f}%)")

        t0 = time.time()
        mesh_result = sugar_centers_mesh_from_gaussians(means_np)
        t1 = time.time()

        if mesh_result is None:
            print(f"  SKIP — Poisson reconstruction failed")
            sugar_results[name] = {"error": "mesh extraction failed"}
            continue

        verts, faces = mesh_result
        print(f"  SuGaR-centers mesh: {len(verts):,} verts, {len(faces):,} faces ({t1-t0:.1f}s)")

        # Sample points on the SuGaR mesh
        pred_points = sample_points_on_mesh(verts.astype(np.float32), faces, NUM_SAMPLES, seed=42)

        # Load GT mesh and sample points
        gt_verts, gt_faces = load_off_mesh(paths["gt"])
        gt_points = sample_points_on_mesh(gt_verts, gt_faces, NUM_SAMPLES, seed=1)

        # Compute Chamfer (same protocol as VoxSeG: unit-normalize + ICP)
        metrics = compute_chamfer(pred_points, gt_points, align=True)
        sugar_results[name] = metrics

        print(f"  SuGaR-centers — Chamfer L2: {metrics['chamfer_l2']:.4f}, "
              f"F@0.02: {metrics['f_at_0.02']:.4f}, F@0.05: {metrics['f_at_0.05']:.4f}")

    # ---- Print comparison table ----
    print(f"\n{'='*90}")
    print("VoxSeG vs SuGaR-centers Baseline (Poisson from Gaussian centres — NOT full SuGaR)")
    print(f"{'='*90}\n")

    header = (
        f"{'Object':10} | {'VoxSeG':>12} | {'SuGaR-ctr':>12} | "
        f"{'VoxSeG':>12} | {'SuGaR-ctr':>12} | "
        f"{'VoxSeG':>12} | {'SuGaR-ctr':>12}"
    )
    subheader = (
        f"{'':10} | {'Chamfer L2':>12} | {'Chamfer L2':>12} | "
        f"{'F@0.02':>12} | {'F@0.02':>12} | "
        f"{'F@0.05':>12} | {'F@0.05':>12}"
    )
    print(subheader)
    print("-" * len(subheader))

    v_chamf, s_chamf = [], []
    v_f02, s_f02 = [], []
    v_f05, s_f05 = [], []

    for name in OBJECTS:
        v = voxseg_results.get(name)
        s = sugar_results.get(name)

        if v is None or s is None or "error" in (s if isinstance(s, dict) else {}):
            print(f"  {name:10} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}")
            continue

        row = (
            f"  {name.capitalize():10} | "
            f"{v['chamfer_l2']:12.4f} | {s['chamfer_l2']:12.4f} | "
            f"{v['f_at_0.02']:12.4f} | {s['f_at_0.02']:12.4f} | "
            f"{v['f_at_0.05']:12.4f} | {s['f_at_0.05']:12.4f}"
        )
        print(row)

        v_chamf.append(v["chamfer_l2"])
        s_chamf.append(s["chamfer_l2"])
        v_f02.append(v["f_at_0.02"])
        s_f02.append(s["f_at_0.02"])
        v_f05.append(v["f_at_0.05"])
        s_f05.append(s["f_at_0.05"])

    if v_chamf:
        n = len(v_chamf)
        mean_row = (
            f"  {'Mean':10} | "
            f"{sum(v_chamf)/n:12.4f} | {sum(s_chamf)/n:12.4f} | "
            f"{sum(v_f02)/n:12.4f} | {sum(s_f02)/n:12.4f} | "
            f"{sum(v_f05)/n:12.4f} | {sum(s_f05)/n:12.4f}"
        )
        print("-" * len(subheader))
        print(mean_row)

    # Save results
    out = {
        "_note": (
            "sugar_centers_baseline reproduces SuGaR's --use_centers_to_extract_mesh "
            "path (Poisson-from-centres). It is NOT the full SuGaR paper pipeline, "
            "which additionally requires COLMAP images and a trained 3DGS checkpoint "
            "with cameras.json — not available for these ModelNet40 splats."
        ),
        "voxseg": voxseg_results,
        "sugar_centers_baseline": {k: v for k, v in sugar_results.items() if not isinstance(v, dict) or "error" not in v},
    }
    out_path = "logs/voxseg_vs_sugar_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
