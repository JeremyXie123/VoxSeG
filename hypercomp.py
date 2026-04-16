import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import gc

from core.splat_io import load_ply, print_gpu_memory
from core.camera import CameraState, get_batch_viewmats, get_batch_Ks, compute_orbit_radius
from stages.rendering import render_splat_views
from stages.segmentation import generate_sam_masks
from stages.optimize import BasicGrid, optimize_voxel_grid
from stages.evaluation import visualize_batch_grid
from core.ui import PromptBoxUI

def plot_sweep_comparison(all_histories, input_filename, sweep_arg_name):
    """
    Generalized plotting function for any parameter sweep.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    sweep_values = list(all_histories.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sweep_values)))

    for i, val in enumerate(sweep_values):
        history = all_histories[val]
        iters = range(len(history['mask_loss']))
        label_str = f"{sweep_arg_name}={val}"
        
        ax1.plot(iters, history['mask_loss'], color=colors[i], label=label_str, linewidth=2)
        ax2.plot(iters, history['smooth_loss'], color=colors[i], label=label_str, linestyle='--')
        ax3.plot(iters, history['time'], color=colors[i], label=label_str, linestyle='-.')

    ax1.set_ylabel('Mask Loss')
    ax1.set_title(f'Sweep Comparison: {input_filename} ({sweep_arg_name})')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Smoothness Loss')
    ax3.set_ylabel('Cumulative Time (s)')
    ax3.set_xlabel('Iteration')
    
    for ax in [ax2, ax3]: ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"sweep/{sweep_arg_name}/comparison.png"
    plt.savefig(out_path)
    print(f"Sweep visualization saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized Hyperparameter Sweep")
    
    # Input/output and Sweep Config
    parser.add_argument("--input", type=str, default="splats/truck.ply")
    parser.add_argument("--sweep_arg", type=str, default="beta", help="Argument to sweep over")
    parser.add_argument("--sweep_vals", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    
    # Box/Camera Params (matching pipeline.py)
    parser.add_argument("--box_center", type=float, nargs=3, required=True)
    parser.add_argument("--box_size", type=float, nargs=3, required=True)
    parser.add_argument("--box_angles", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--label", type=str, default="truck")
    
    parser.add_argument("--num_rings", type=int, default=5, help="Number of elevation rings")
    parser.add_argument("--cameras_per_ring", type=int, default=20, help="Cameras per ring")
    parser.add_argument("--focal_length", type=float, default=550.0, help="Default focal length")
    parser.add_argument("--padding", type=float, default=1.0, help="Default padding factor")
    parser.add_argument("--elevation_min", type=float, default=5.0, help="Min elevation angle")
    parser.add_argument("--elevation_max", type=float, default=15.0, help="Max elevation angle")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")

    # Optimization defaults
    parser.add_argument("--grid_resolution", type=int, default=128)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric", type=str, default="bce")
    parser.add_argument("--sharpness", type=float, default=5.0)
    parser.add_argument("--iso_level", type=float, default=0.5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_name = os.path.splitext(os.path.basename(args.input))[0]

    # --- Phase 1: Geometry & Rendering ---
    splats = load_ply(args.input, device)
    print(f"Loaded {splats.means.shape[0]:,} Gaussians from {args.input}")

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("neg_y_up")
    ps.set_navigation_style("first_person")

    ps.register_gaussian_particles(
        "gaussians",
        subsample_factor=1,
        means=splats.means.unsqueeze(0),
        colors=splats.colors.unsqueeze(0).unsqueeze(2),  # [1, N, 1, 3]
        opacities=splats.opacities.unsqueeze(0),
        scales=splats.scales.exp().unsqueeze(0),
        quats=splats.quats.unsqueeze(0),
        sh_degree=0,
    )
    
    box_ui = PromptBoxUI(args, args.box_center)
    box_ui.register()

    ps.set_user_callback(box_ui.make_ui_callback())
    print("\n[Polyscope] Adjust the box, then close the window to continue...")
    ps.show()

    # Mirroring pipeline.py logic for radius and cameras
    grid_radius = float(np.max(args.box_size) / 2.0)
    cam_radius = compute_orbit_radius(args.box_size, args.focal_length, args.resolution, args.padding)
    
    viewmats, cameras_per_ring = get_batch_viewmats(
        center=box_ui.get_center(),
        radius=cam_radius,
        num_rings=int(box_ui.num_rings),
        cameras_per_ring=int(box_ui.cameras_per_ring),
        elevation_min=box_ui.elevation_min,
        elevation_max=box_ui.elevation_max,
        up_axis=box_ui.get_up_axis(),
        device=device
    )
    num_views = len(viewmats)
    Ks = get_batch_Ks(args.focal_length, args.resolution, args.resolution, num_views, device)
    
    args.num_views = num_views
    args.cameras_per_ring = cameras_per_ring
    args.width = args.resolution
    args.height = args.resolution

    # Temporary CameraState for rendering
    cams = CameraState(
        target_center=torch.tensor(args.box_center, device=device),
        target_radius=float(np.max(args.box_size)),
        cam_radius=cam_radius, grid_radius=grid_radius,
        grid_rotation=torch.eye(3, device=device), # Simplified for script
        viewmats=viewmats, Ks=Ks
    )
    
    args.width = args.height = args.resolution
    rendered_images = render_splat_views(splats, cams, args)
    
    # segmentation needs 3D center as (1,3) tensor
    box_center_3d = torch.tensor(args.box_center, device=device).unsqueeze(0)
    seg_result = generate_sam_masks(rendered_images, box_center_3d, cams, args, device, label=args.label)

    # --- Phase 2: Sweep Execution ---
    all_histories = {}
    original_cams = cams # Keep original for reference

    for val in args.sweep_vals:
        print(f"\n>>> Running Sweep: {args.sweep_arg} = {val}")
        setattr(args, args.sweep_arg, val) # Dynamically update the arg
        
        # Re-filter cams based on valid indices from SAM
        valid_idx = seg_result.valid_indices
        current_cams = CameraState(
            target_center=original_cams.target_center,
            target_radius=original_cams.target_radius,
            cam_radius=original_cams.cam_radius,
            grid_radius=original_cams.grid_radius,
            grid_rotation=original_cams.grid_rotation,
            viewmats=original_cams.viewmats[valid_idx],
            Ks=original_cams.Ks[valid_idx]
        )
        args.num_views = len(valid_idx)

        grid = BasicGrid(args, device)
        os.makedirs(f"sweep/{args.sweep_arg}/{val}", exist_ok=True)
        history = optimize_voxel_grid(grid, seg_result, current_cams, args, device, path=f"sweep/{args.sweep_arg}/{val}")
        
        all_histories[val] = {k: [float(v) for v in l] for k, l in history.items() if k != 'time' or True}
        all_histories[val]['time'] = history['time']

        del grid
        gc.collect()
        torch.cuda.empty_cache()

    # --- Phase 3: Plotting ---
    plot_sweep_comparison(all_histories, input_name, args.sweep_arg)