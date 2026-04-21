"""Recompute 2D F1/IoU for a previous pipeline run from its saved eval bundle.

Usage:
    python eval_mask_metrics.py --bundle logs/truck/eval_bundle.pt
    python eval_mask_metrics.py --bundle logs/truck/eval_bundle.pt --threshold 0.4 \
        --save_json logs/truck/mask_metrics_t04.json
"""

import argparse
import os
import torch

from stages.evaluation import compute_2d_mask_metrics, load_eval_bundle, save_2d_mask_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc 2D mask F1/IoU from a saved eval bundle.")
    parser.add_argument("--bundle", type=str, required=True, help="Path to eval_bundle.pt")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for rendered masks")
    parser.add_argument("--save_json", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.bundle):
        raise FileNotFoundError(
            f"Eval bundle not found: {args.bundle}\n"
            "Run pipeline.py successfully first to generate logs/<name>/eval_bundle.pt."
        )

    device = torch.device(args.device)
    phi_grid, seg_result, cams, bundle_args = load_eval_bundle(args.bundle, device)

    metrics = compute_2d_mask_metrics(phi_grid, seg_result, cams, bundle_args, device, threshold=args.threshold)
    out_json = args.save_json or args.bundle.replace(".pt", f"_metrics_t{args.threshold:.2f}.json")
    save_2d_mask_metrics(metrics, out_json)


if __name__ == "__main__":
    main()
