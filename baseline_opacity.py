#!/usr/bin/env python3
"""
Opacity-threshold baseline: take Gaussian means with opacity > threshold,
treat them as a point cloud, and compute Chamfer distance against GT mesh.

This gives a simple reference for what you get "for free" from the splat
without any volumetric optimization.
"""

import json
import os
import sys
import time
import numpy as np
import torch

# Add project root to path
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
OPACITY_THRESHOLD = 0.5  # keep Gaussians with opacity > 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for name, paths in OBJECTS.items():
        print(f"\n{'='*40}")
        print(f"  {name.upper()}")
        print(f"{'='*40}")

        # Load splat
        splats = load_ply(paths["splat"], device)
        n_total = splats.means.shape[0]
        print(f"  Total Gaussians: {n_total:,}")

        # Threshold by opacity
        mask = splats.opacities > OPACITY_THRESHOLD
        means_filtered = splats.means[mask].detach().cpu().numpy()
        n_kept = means_filtered.shape[0]
        print(f"  After opacity > {OPACITY_THRESHOLD}: {n_kept:,} ({100*n_kept/n_total:.1f}%)")

        if n_kept < 100:
            print(f"  SKIP — too few points after thresholding")
            results[name] = {"error": "too few points"}
            continue

        # Load GT mesh and sample points
        gt_verts, gt_faces = load_off_mesh(paths["gt"])
        gt_points = sample_points_on_mesh(gt_verts, gt_faces, NUM_SAMPLES, seed=1)

        # For the baseline, we use the Gaussian means directly as the "predicted surface"
        # If there are more than NUM_SAMPLES, subsample uniformly
        if n_kept > NUM_SAMPLES:
            idx = np.random.default_rng(0).choice(n_kept, NUM_SAMPLES, replace=False)
            pred_points = means_filtered[idx]
        else:
            pred_points = means_filtered

        print(f"  Pred points: {pred_points.shape[0]:,},  GT points: {gt_points.shape[0]:,}")

        # Compute Chamfer
        metrics = compute_chamfer(pred_points, gt_points, align=True)
        results[name] = metrics

        print(f"  Chamfer L2:  {metrics['chamfer_l2']:.4f}")
        print(f"  F@0.02:      {metrics['f_at_0.02']:.4f}")
        print(f"  F@0.05:      {metrics['f_at_0.05']:.4f}")

    # Print summary table
    print(f"\n{'='*80}")
    print("OPACITY-THRESHOLD BASELINE (opacity > 0.5, Gaussian means as point cloud)")
    print(f"{'='*80}\n")

    print("| Object   | Chamfer L2 | F@0.02 | F@0.05 |")
    print("|----------|------------|--------|--------|")

    chamfer_vals = []
    f02_vals = []
    f05_vals = []

    for name in OBJECTS:
        m = results.get(name, {})
        if "error" in m:
            print(f"| {name.capitalize():8} | {'N/A':>10} | {'N/A':>6} | {'N/A':>6} |")
            continue
        chamfer_vals.append(m["chamfer_l2"])
        f02_vals.append(m["f_at_0.02"])
        f05_vals.append(m["f_at_0.05"])
        print(f"| {name.capitalize():8} | {m['chamfer_l2']:10.4f} | {m['f_at_0.02']:6.4f} | {m['f_at_0.05']:6.4f} |")

    if chamfer_vals:
        avg_c = sum(chamfer_vals) / len(chamfer_vals)
        avg_02 = sum(f02_vals) / len(f02_vals)
        avg_05 = sum(f05_vals) / len(f05_vals)
        print(f"| {'Mean':8} | {avg_c:10.4f} | {avg_02:6.4f} | {avg_05:6.4f} |")

    # Save results
    out_path = "logs/baseline_opacity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
