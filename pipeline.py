import argparse
import torch
import numpy as np

from core.splat_io import load_ply, print_gpu_memory
from core.camera import CameraState, setup_camera_geometry, get_batch_Ks, get_batch_viewmats
from stages.rendering import render_splat_views
from stages.segmentation import generate_sam_masks
from stages.optimize import optimize_voxel_grid, render_phi_to_image
from stages.evaluation import visualize_with_polyscope, plot_training_metrics, visualize_batch_grid

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
    parser.add_argument("--num_test_views", type=int, default=5, help="Number of unseen views to render for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of views to sample per optimization step")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_memory()

    interior_3d = torch.tensor([
        [ 2.293, -0.090, 0.407], [ 2.347, -0.697, 0.407], 
        [ 0.574, -0.697, 0.407], [ 0.628,  0.264, 0.407],
        [-2.866, -0.658, 0.407], [-0.899, -0.580, 0.407], 
        [-0.429, -0.138, 0.407]
    ], dtype=torch.float32, device=device)

    # --- 1. CORE PIPELINE ---
    splats = load_ply(args.input, device)
    cams = setup_camera_geometry(interior_3d, splats.means, args, device)
    
    rendered_images = render_splat_views(splats, cams, args)
    seg_result = generate_sam_masks(rendered_images, interior_3d, cams, args, device)
    
    phi_grid, history = optimize_voxel_grid(seg_result, cams, args, device)

    # --- 2. EVALUATION & VISUALIZATION ---
    print("Optimization complete. Visualizing training history...")
    plot_training_metrics(history, filename="graphs/truck_optimization_log.png")

    print("Visualizing vertices in polyscope...")
    visualize_with_polyscope(seg_result.blended_images, cams, phi_grid, args)

    print("Generating unseen views for evaluation...")
    test_Ks = get_batch_Ks(args.focal, args.width, args.height, args.num_test_views, device)
    test_viewmats = get_batch_viewmats(splats.means, cams.target_center, cams.cam_radius, args.num_test_views)
    
    # Bundle the new views into our structured CameraState
    test_cams = CameraState(
        target_center=cams.target_center,
        target_radius=cams.target_radius,
        cam_radius=cams.cam_radius,
        grid_radius=cams.grid_radius,
        viewmats=test_viewmats,
        Ks=test_Ks
    )

    test_renders = render_splat_views(splats, test_cams, args, chunk_size=args.num_test_views)
    phi_renders = render_phi_to_image(phi_grid, test_cams, args, device)

    # Concat and visualize
    original_imgs = test_renders.detach().float().clamp(0, 1).cpu().numpy()
    phi_masks = phi_renders.detach().float().cpu().numpy()
    phi_masks_rgb = np.repeat(phi_masks[:, :, :, None], 3, axis=-1)

    combined = np.concatenate([original_imgs, phi_masks_rgb], axis=0)
    visualize_batch_grid(combined, num_cols=args.num_test_views)