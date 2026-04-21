#!/usr/bin/env python3
"""
Collect and aggregate F1/IoU metrics from all object sweeps.
Outputs a markdown table suitable for reports.
"""

import json
import os
from pathlib import Path

objects = ["airplane", "car", "guitar", "bathtub", "toilet"]
log_base = Path("logs")

results = {}

# Collect metrics from each object
for obj in objects:
    metrics_file = log_base / obj / "mask_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
                mean_metrics = data.get("mean", {})
                geom = data.get("geometry", {}) or {}
                results[obj] = {
                    "f1": mean_metrics.get("f1", None),
                    "iou": mean_metrics.get("iou", None),
                    "precision": mean_metrics.get("precision", None),
                    "recall": mean_metrics.get("recall", None),
                    "num_views": data.get("num_views", None),
                    "chamfer_l2": geom.get("chamfer_l2", None),
                    "f_at_0.02": geom.get("f_at_0.02", None),
                    "f_at_0.05": geom.get("f_at_0.05", None),
                }
        except Exception as e:
            print(f"[Warning] Could not load {obj} metrics: {e}")
            results[obj] = {"f1": None, "iou": None, "precision": None, "recall": None, "num_views": None, "chamfer_l2": None, "f_at_0.02": None, "f_at_0.05": None}
    else:
        results[obj] = {"f1": None, "iou": None, "precision": None, "recall": None, "num_views": None, "chamfer_l2": None, "f_at_0.02": None, "f_at_0.05": None}

# Print markdown table
print("\n" + "="*80)
print("SEGMENTATION METRICS - F1 & IoU SCORES")
print("="*80 + "\n")

print("| Object   | F1 Score | IoU Score | Precision | Recall | Chamfer L2 | F@0.02 | F@0.05 | Views |")
print("|----------|----------|-----------|-----------|--------|------------|--------|--------|-------|")

for obj in objects:
    data = results[obj]
    f1_str = f"{data['f1']:.4f}" if data['f1'] is not None else "N/A"
    iou_str = f"{data['iou']:.4f}" if data['iou'] is not None else "N/A"
    prec_str = f"{data['precision']:.4f}" if data['precision'] is not None else "N/A"
    rec_str = f"{data['recall']:.4f}" if data['recall'] is not None else "N/A"
    cham_str = f"{data['chamfer_l2']:.4f}" if data['chamfer_l2'] is not None else "N/A"
    fsc02_str = f"{data['f_at_0.02']:.4f}" if data['f_at_0.02'] is not None else "N/A"
    fsc05_str = f"{data['f_at_0.05']:.4f}" if data['f_at_0.05'] is not None else "N/A"
    views_str = str(data['num_views']) if data['num_views'] is not None else "N/A"

    print(f"| {obj.capitalize():8} | {f1_str:8} | {iou_str:9} | {prec_str:9} | {rec_str:6} | {cham_str:10} | {fsc02_str:6} | {fsc05_str:6} | {views_str:5} |")

print("\n" + "="*80 + "\n")

# Plain text summary
print("SUMMARY STATISTICS")
print("-" * 80)

valid_f1s = [data['f1'] for data in results.values() if data['f1'] is not None]
valid_ious = [data['iou'] for data in results.values() if data['iou'] is not None]
valid_chamfer = [data['chamfer_l2'] for data in results.values() if data['chamfer_l2'] is not None]

if valid_f1s:
    avg_f1 = sum(valid_f1s) / len(valid_f1s)
    print(f"Average F1:      {avg_f1:.4f}")
    print(f"F1 Range:        {min(valid_f1s):.4f} - {max(valid_f1s):.4f}")

if valid_ious:
    avg_iou = sum(valid_ious) / len(valid_ious)
    print(f"Average IoU:     {avg_iou:.4f}")
    print(f"IoU Range:       {min(valid_ious):.4f} - {max(valid_ious):.4f}")

if valid_chamfer:
    avg_ch = sum(valid_chamfer) / len(valid_chamfer)
    print(f"Average Chamfer: {avg_ch:.4f}")
    print(f"Chamfer Range:   {min(valid_chamfer):.4f} - {max(valid_chamfer):.4f}")

print("\n" + "="*80 + "\n")
