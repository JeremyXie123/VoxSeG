"""
Voxelization Pipeline with Interactive Polyscope Setup

Usage:
    python pipeline.py --input splats/truck.ply --num_views 100

The pipeline opens a Polyscope window where you can:
1. Position/scale/rotate the orange prompt box around the target object
2. Adjust camera parameters (focal length, padding, elevation)
3. Preview camera positions with "Preview Cameras" button
4. Close the window to start rendering and optimization
"""

import argparse
import os
import torch
import numpy as np
import polyscope as ps

from core.splat_io import load_ply, print_gpu_memory, sh_to_rgb
from core.camera import CameraState, get_batch_viewmats, get_batch_Ks, compute_orbit_radius, compute_widget_focal_length
from core.ui import PromptBoxUI
from stages.rendering import render_splat_views
from stages.segmentation import generate_sam_masks
from stages.optimize import BasicGrid, optimize_voxel_grid
from stages.evaluation import plot_training_metrics, visualize_batch_grid

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelization pipeline with interactive setup")
    
    # Input/output
    parser.add_argument("--input", type=str, default="splats/truck.ply", help="Path to input PLY file")
    
    # Box presets (optional, for scripted runs)
    parser.add_argument("--box_center", type=float, nargs=3, default=None, help="Initial box center (X Y Z)")
    parser.add_argument("--box_size", type=float, nargs=3, default=None, help="Initial box size (X Y Z)")
    parser.add_argument("--box_angles", type=float, nargs=3, default=None, help="Initial box rotation (X Y Z) degrees")
    parser.add_argument("--label", type=str, default="", help="Object label for SAM3 text prompt (e.g., 'truck', 'car')")
    
    # Camera defaults (can be adjusted in UI)
    parser.add_argument("--num_rings", type=int, default=5, help="Number of elevation rings")
    parser.add_argument("--cameras_per_ring", type=int, default=20, help="Cameras per ring")
    parser.add_argument("--focal_length", type=float, default=550.0, help="Default focal length")
    parser.add_argument("--padding", type=float, default=1.0, help="Default padding factor")
    parser.add_argument("--elevation_min", type=float, default=5.0, help="Min elevation angle")
    parser.add_argument("--elevation_max", type=float, default=45.0, help="Max elevation angle")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution")
    
    # Optimization
    parser.add_argument("--grid_resolution", type=int, default=128, help="Voxel grid resolution")
    parser.add_argument("--num_iters", type=int, default=300, help="Optimization iterations")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per ray")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--sharpness", type=float, default=5.0, help="Sharpness for phi→opacity")
    parser.add_argument("--beta", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--iso_level", type=float, default=0.5, help="Isosurface level")
    parser.add_argument("--batch_size", type=int, default=32, help="Views per optimization step")
    parser.add_argument("--metric", type=str, default="bce", choices=["bce", "mse", "kl"])
    
    # Evaluation
    parser.add_argument("--num_test_views", type=int, default=7, help="Unseen views for evaluation")
    parser.add_argument("--num_test_samples", type=int, default=100, help="Samples per ray for test")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_memory()
    
    # -------------------------------------------------------------------------
    # Phase 1: Load splats and setup interactive Polyscope
    # -------------------------------------------------------------------------
    
    splats = load_ply(args.input, device)
    print(f"Loaded {splats.means.shape[0]:,} Gaussians from {args.input}")
    
    # Initialize Polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("neg_y_up")
    ps.set_navigation_style("first_person")
    
    # Register Gaussians (Polyscope expects raw SH colors with shape [1, N, 1, 3])
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
    
    # Create and register prompt box UI
    default_center = splats.means.mean(dim=0).cpu().numpy()
    box_ui = PromptBoxUI(args, default_center)
    box_ui.register()
    
    # Set UI callback and show
    ps.set_user_callback(box_ui.make_ui_callback())
    print("\n[Polyscope] Adjust the box, then close the window to continue...")
    ps.show()
    
    # -------------------------------------------------------------------------
    # Phase 2: Extract parameters and build CameraState
    # -------------------------------------------------------------------------
    
    # Print final box properties for reproducibility
    box_ui.print_properties()
    
    center, size, _ = box_ui.get_properties()
    
    # Compute radii from box size
    grid_radius = float(np.max(size) / 2.0)
    cam_radius = compute_orbit_radius(size, focal_length=box_ui.focal_length, image_size=args.resolution, padding=box_ui.padding)
    
    # Generate camera matrices
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

    Ks = get_batch_Ks(box_ui.focal_length, args.resolution, args.resolution, num_views, device)
    
    # Update args with derived values
    args.num_views = num_views
    args.cameras_per_ring = cameras_per_ring
    args.width = args.resolution
    args.height = args.resolution
    args.focal = box_ui.focal_length
    
    target_center = torch.tensor(box_ui.get_center(), dtype=torch.float32, device=device)
    
    print(f"\n[Box] Center: {box_ui.get_center()}")
    print(f"[Box] Size: {size}")
    print(f"[Box] Grid radius: {grid_radius:.4f}")
    print(f"[Box] Camera radius: {cam_radius:.4f}")
    print(f"[Box] Num views: {num_views} ({int(box_ui.num_rings)} rings × {cameras_per_ring}/ring)")
    
    cams = CameraState(
        target_center=target_center,
        target_radius=float(np.max(size)),
        cam_radius=cam_radius,
        grid_radius=grid_radius,
        viewmats=viewmats,
        Ks=Ks
    )
    
    # -------------------------------------------------------------------------
    # Phase 3: Render views and generate SAM masks
    # -------------------------------------------------------------------------
    
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # For SAM prompts, use box center projected to each view
    box_center_3d = torch.tensor(box_ui.get_center(), dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"\n[Render] Rendering {args.num_views} views...")
    rendered_images = render_splat_views(splats, cams, args)
    
    print(f"[SAM] Generating masks with label '{box_ui.label}'...")
    seg_result = generate_sam_masks(rendered_images, box_center_3d, cams, args, device, label=box_ui.label)
    
    # -------------------------------------------------------------------------
    # Phase 4: Optimize voxel grid
    # -------------------------------------------------------------------------
    
    print(f"\n[Optimize] Grid resolution: {args.grid_resolution}³")
    phi_grid = BasicGrid(args, device)
    history = optimize_voxel_grid(phi_grid, seg_result, cams, args, device)
    
    # -------------------------------------------------------------------------
    # Phase 5: Visualization and evaluation
    # -------------------------------------------------------------------------
    
    print("\n[Eval] Saving optimization metrics...")
    os.makedirs("graphs", exist_ok=True)
    plot_training_metrics(history, filename=f"graphs/{input_name}_optimization.png")
    
    print("[Eval] Generating test views...")
    test_viewmats, _ = get_batch_viewmats(
        center=box_ui.get_center(),
        radius=cams.cam_radius,
        num_rings=1,
        cameras_per_ring=args.num_test_views,
        elevation_min=box_ui.elevation_min,
        elevation_max=box_ui.elevation_max,
        up_axis=box_ui.get_up_axis(),
        device=device
    )
    test_Ks = get_batch_Ks(box_ui.focal_length, args.resolution, args.resolution, args.num_test_views, device)
    
    test_cams = CameraState(
        target_center=cams.target_center,
        target_radius=cams.target_radius,
        cam_radius=cams.cam_radius,
        grid_radius=cams.grid_radius,
        viewmats=test_viewmats,
        Ks=test_Ks
    )
    
    test_renders = render_splat_views(splats, test_cams, args, chunk_size=args.num_test_views)
    
    phi_masks = torch.stack([phi_grid.render_mask(i, test_cams, args.num_test_samples) 
                             for i in range(args.num_test_views)])
    phi_renders = phi_masks.view(args.num_test_views, args.height, args.width)
    
    original_imgs = test_renders.detach().float().clamp(0, 1).cpu().numpy()
    phi_masks_np = phi_renders.detach().float().cpu().numpy()
    phi_masks_rgb = np.repeat(phi_masks_np[:, :, :, None], 3, axis=-1)
    
    combined = np.concatenate([original_imgs, phi_masks_rgb], axis=0)
    visualize_batch_grid(combined, num_cols=args.num_test_views)
    
    print("\n[Polyscope] Final visualization...")
    phi_grid.visualize(cams)
    widget_size = compute_widget_focal_length(cams.cam_radius, args.cameras_per_ring)
    box_ui.register_cameras(cams.viewmats, cams.Ks, masked_rgbs=seg_result.blended_images,
                            widget_focal_length=widget_size, color=(0.5, 0.5, 0.5))
    ps.show()
    
    print("\n[Done]")
    print_gpu_memory()