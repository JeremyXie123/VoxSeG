import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.splat_io import load_ply, print_gpu_memory
from core.camera import CameraState, setup_camera_geometry
from stages.rendering import render_splat_views
from stages.segmentation import generate_sam_masks
from stages.optimize import optimize_voxel_grid

def plot_beta_comparison(all_histories, input_filename):
    """
    Plots Mask Loss, Smoothness Loss, and Cumulative Time curves 
    for multiple beta values in a 3-high vertical stack.
    """
    # Create a 3x1 grid of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    beta_keys = list(all_histories.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(beta_keys)))

    for i, beta in enumerate(beta_keys):
        history = all_histories[beta]
        iters = range(len(history['mask_loss']))
        
        # 1. Top Plot: Mask BCE Loss (Data Fidelity)
        ax1.plot(iters, history['mask_loss'], color=colors[i], 
                 label=f'beta={beta}', linewidth=2)
        
        # 2. Middle Plot: Smoothness Loss (Regularization)
        ax2.plot(iters, history['smooth_loss'], color=colors[i], 
                 label=f'beta={beta}', linestyle='--')
        
        # 3. Bottom Plot: Cumulative Time (Performance)
        ax3.plot(iters, history['time'], color=colors[i], 
                 label=f'beta={beta}', linestyle='-.')

    # Formatting Top Plot (Mask Loss)
    ax1.set_ylabel('Mask BCE Loss')
    ax1.set_title(f'Convergence Comparison: {input_filename}')
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Formatting Middle Plot (Smoothness)
    ax2.set_ylabel('Smoothness Loss')
    ax2.set_title('Regularization Magnitude')
    ax2.grid(True, alpha=0.3)

    # Formatting Bottom Plot (Time)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xlabel('Iteration')
    ax3.set_title('Computation Time')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    out_path = f"graphs/{input_filename}_beta_sweep_full.png"
    plt.savefig(out_path)
    print(f"Full sweep visualization saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="splats/truck.ply", help="Path to input PLY file")
    parser.add_argument("--output_render", type=str, default="rendered", help="Directory to save rendered images")
    parser.add_argument("--sam_checkpoint", type=str, default="sam2.1_hiera_l.pt", help="Path to SAM checkpoint")
    parser.add_argument("--sam_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to SAM config")
    parser.add_argument("--cam_radius_mul", type=float, default=1, help="Multiplier for camera radius based on target object size")
    parser.add_argument("--grid_radius_mul", type=float, default=1, help="Multiplier for voxel grid radius based on target object size")
    parser.add_argument("--grid_resolution", type=int, default=64, help="Resolution of voxelization grid")
    parser.add_argument("--width", type=int, default=512, help="Width of rendered images")
    parser.add_argument("--height", type=int, default=512, help="Height of rendered images")
    parser.add_argument("--focal", type=float, default=1100.0, help="Focal length for rendering")
    parser.add_argument("--num_views", type=int, default=10, help="Number of views to render")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations for optimization")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per ray")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for optimization")
    parser.add_argument("--sharpness", type=float, default=1.0, help="Sharpness parameter for converting phi to opacity")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for smoothness regularization")
    parser.add_argument("--iso_level", type=float, default=0.0, help="Isosurface level for visualization")
    parser.add_argument("--num_test_views", type=int, default=7, help="Number of unseen views to render for evaluation")
    parser.add_argument("--num_test_samples", type=int, default=100, help="Number of samples per ray for test view rendering")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of views to sample per optimization step")
    parser.add_argument("--metric", type=str, default="bce", choices=["bce", "mse", "kl"], help="Loss metric for optimization")
    args = parser.parse_args()

    input_filename = os.path.splitext(os.path.basename(args.input))[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interior_3d = torch.tensor([
        [ 2.293, -0.090, 0.407], [ 2.347, -0.697, 0.407], 
        [ 0.574, -0.697, 0.407], [ 0.628,  0.264, 0.407],
        [-2.866, -0.658, 0.407], [-0.899, -0.580, 0.407], 
        [-0.429, -0.138, 0.407]
    ], dtype=torch.float32, device=device)

    # --- 1. SETUP (Run once per script) ---
    splats = load_ply(args.input, device)
    cams = setup_camera_geometry(interior_3d, splats.means, args, device)
    
    # We only need to render and segment once; the masks remain the same for all betas
    rendered_images = render_splat_views(splats, cams, args)
    seg_result = generate_sam_masks(rendered_images, interior_3d, cams, args, device)
    
    # --- 2. BETA SWEEP ---
    metric_values = ["bce", "mse", "kl"]  # Example metrics to sweep over; replace with actual beta values if needed
    all_histories = {}

    for x in metric_values:
        print(f"\n" + "="*40)
        print(f"STARTING OPTIMIZATION: Metric = {x}")
        print("="*40)
        
        # Manually override beta in args for the optimizer
        args.metric = x
        
        # optimize_voxel_grid initializes a fresh phi grid internally each time it is called
        phi_grid, history = optimize_voxel_grid(seg_result, cams, args, device)
        all_histories[x] = {k: [float(v) for v in l] for k, l in history.items()}

        del phi_grid
        del history
        torch.cuda.empty_cache() # Frees the "Reserved" memory back to the OS
        import gc
        gc.collect() # Forces Python to clear orphaned objects
        print_gpu_memory()

    # --- 3. PLOTTING ---
    plot_beta_comparison(all_histories, input_filename)