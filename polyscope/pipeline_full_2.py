# Example command: python pipeline_full_2.py --num_views=300 --cam_radius_mul=0.8 --grid_radius_mul=0.5 --focal=550.0 --sharpness=5 --num_iters=300 --iso_level=0.5 --batch_size=16 --grid_resolution=128

import math
from fvdb_optimizer import fVDBOptimizerGrid
import torch
import os
import numpy as np
from plyfile import PlyData
from gsplat import rasterization
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import torch
import torch.nn.functional as F
import gc
import tempfile
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# --- NVIDIA fVDB Import ---
try:
    import fvdb
except ImportError:
    print("CRITICAL ERROR: NVIDIA fVDB is not installed in this environment.")
    print("Please ensure you have built fVDB from the OpenVDB GitHub repository.")
    exit(1)

def init_phi_grid(resolution, device):
    """Initializes a voxel grid of the given resolution centered at the given point and with the given radius"""
    phi = torch.nn.Parameter(torch.randn((resolution, resolution, resolution), device=device) * 0.1 + 0.5)
    return phi

def query_phi_trilinear(phi, points, grid_center, grid_radius):
    """Queries the voxel grid at 3D points using trilinear interpolation."""
    # Normalize points to [-1, 1] range for grid_sample
    center = grid_center.to(points.device)
    points_norm = (points - center) / grid_radius

    # grid_sample expects [N, C, D, H, W] and coordinates in [W, H, D] order
    # phi is [D, H, W], we add Batch and Channel dims
    grid = phi[None, None, ...]

    # Reshape points for grid_sample: [1, N_rays, N_samples, 1, 3]
    N_rays, N_samples, _ = points_norm.shape
    sampling_coords = points_norm.reshape(1, N_rays * N_samples, 1, 1, 3)

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    # When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear. However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    vals = torch.nn.functional.grid_sample(grid, sampling_coords, mode='bilinear', padding_mode='border', align_corners=True)

    return vals.reshape(N_rays, N_samples)

def dirichlet_energy(phi):
    """Computes the Dirichlet energy (L2 norm of gradients) as a smoothness prior."""
    # https://en.wikipedia.org/wiki/Dirichlet_energy
    dx = phi[1:, :, :] - phi[:-1, :, :]
    dy = phi[:, 1:, :] - phi[:, :-1, :]
    dz = phi[:, :, 1:] - phi[:, :, :-1]

    return (dx**2).mean() + (dy**2).mean() + (dz**2).mean()

def construct_rays(viewmat, K, height, width, device):
    """Generates rays (origin and direction) for each pixel in the image."""
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")

    # K maps [X_cam, Y_cam, Z_cam] to [u, v, 1], apply inverse to get camera directions from pixels
    inv_K = torch.linalg.inv(K)
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # [H, W, 3]
    cam_dirs = pixels @ inv_K.T

    # Rotate ray directions based on viewmat rotation
    cam_to_world = torch.linalg.inv(viewmat)
    ray_dirs = cam_dirs @ cam_to_world[:3, :3].T
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

    # Move ray origins based on viewmat translation
    ray_origins = cam_to_world[:3, 3].expand(height, width, 3)

    # Return as [N, 3] for use with sampling function
    return ray_origins.reshape(-1, 3), ray_dirs.reshape(-1, 3)

def sample_points_along_rays(ray_origins, ray_dirs, num_samples, near=0.1, far=10.0):
    """Samples points along rays between near and far planes."""
    t_vals = torch.linspace(near, far, num_samples, device=ray_origins.device) # [num_samples]
    points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None] # [H*W, 3]
    return points

def render_phi_to_image(grid, viewmats, Ks, height, width, sharpness, device):
    """Renders the fVDB grid as an image (masks) using volume rendering."""
    renderings = []
    for i in range(len(viewmats)):
        ray_origins, ray_dirs = construct_rays(viewmats[i], Ks[i], height, width, device)
        points = sample_points_along_rays(ray_origins, ray_dirs, num_samples=100)
        
        # Query the fVDB grid object directly
        phi_vals = grid.query(points)
        
        alpha = torch.sigmoid(-sharpness * phi_vals)
        mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)
        renderings.append(mask.reshape(height, width))
    return torch.stack(renderings)

# def render_phi_to_image(phi, viewmats, Ks, height, width, center, radius, sharpness, device):
#     """Renders the phi grid as an image (masks) using volume rendering."""
#     renderings = []
#     for i in range(len(viewmats)):
#         ray_origins, ray_dirs = construct_rays(viewmats[i], Ks[i], height, width, device)
#         # Use a higher sample count for cleaner evaluation
#         points = sample_points_along_rays(ray_origins, ray_dirs, num_samples=100) 
#         phi_vals = query_phi_trilinear(phi, points, center, radius)
#         alpha = torch.sigmoid(-sharpness * phi_vals)
#         mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)
#         renderings.append(mask.reshape(height, width))
#     return torch.stack(renderings)

# -- Utility Functions --

def print_gpu_memory():
    """Prints the current GPU memory usage in GB."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")

def get_size(obj):
    """Computes the memory size of a tensor or a list/tuple."""
    if isinstance(obj, (list, tuple)):
        return sum(get_size(item) for item in obj)
    
    if torch.is_tensor(obj):
        return obj.element_size() * obj.nelement()
    
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    
    return 0

def print_tensor_memory(data, name):
    """Reports memory usage in MiB."""
    size_mib = get_size(data) / (1024**2)
    print(f"{name} memory: {size_mib:.2f} MiB")

def load_ply(path, device):
    """Loads a PLY file and various information about it onto the given device"""
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    means = torch.stack([torch.tensor(v['x']), torch.tensor(v['y']), torch.tensor(v['z'])], dim=-1).to(device)
    scales = torch.stack([torch.tensor(v['scale_0']), torch.tensor(v['scale_1']), torch.tensor(v['scale_2'])], dim=-1).to(device)
    quats = torch.stack([torch.tensor(v['rot_0']), torch.tensor(v['rot_1']), torch.tensor(v['rot_2']), torch.tensor(v['rot_3'])], dim=-1).to(device)
    colors = torch.stack([torch.tensor(v['f_dc_0']), torch.tensor(v['f_dc_1']), torch.tensor(v['f_dc_2'])], dim=-1).to(device)
    colors = (colors * 0.28209) + 0.5 # Normalize spherical harmonic colors to [0, 1]
        
    opacities = torch.sigmoid(torch.tensor(v['opacity'])).to(device)
    return means, scales, quats, colors, opacities

def get_batch_viewmats(means, center, distance, num_views, num_rotations=4, max_pitch=np.pi/9):
    viewmats = []
    
    for i in range(num_views):
        yaw = (2 * np.pi * num_rotations / num_views) * i
        pitch = (max_pitch / (num_views - 1)) * i #- max_pitch/2
        
        # Remove the negative sign on pitch so cameras climb ABOVE the object
        x = distance * np.cos(pitch) * np.cos(yaw)
        y = distance * np.sin(-pitch)
        z = distance * np.cos(pitch) * np.sin(yaw)
        
        cam_pos = center + torch.tensor([x, y, z], device=means.device)

        # Standard OpenCV LookAt (Right-handed, Y-Down, Z-Forward)
        z_axis = (center - cam_pos)
        z_axis /= torch.norm(z_axis)  # Forward (+Z)

        up = torch.tensor([0, 1, 0], dtype=torch.float32, device=means.device)
        
        # Right (+X) = Cross(World Up, Forward)
        x_axis = torch.linalg.cross(up, z_axis)
        x_axis /= torch.norm(x_axis)

        # Down (+Y) = Cross(Forward, Right)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        y_axis /= torch.norm(y_axis)
        
        R = torch.stack([x_axis, y_axis, z_axis], dim=0)
        T = -R @ cam_pos

        mat = torch.eye(4, device=means.device)
        mat[:3, :3] = R
        mat[:3, 3] = T
        viewmats.append(mat)

    return torch.stack(viewmats)

def get_batch_Ks(focal, width, height, num_views, device):
    """Generates a batch of identical camera intrinsics"""
    Ks = torch.tensor([
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ], device=device).repeat(num_views, 1, 1)
    return Ks

def visualize_batch_grid(images_input, num_cols=5, axes_pad=0.1):
    """Renders a batch of images in an ImageGrid"""
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

    # plt.tight_layout()
    plt.show()

def project_points(points_3d, viewmat, K):
    """Projects 3D points to 2D pixel coordinates using camera extrinsics and intrinsics."""
    # World to Camera Space
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)
    cam_points = (viewmat @ points_homo.T).T[:, :3]

    # Camera to Image Plane
    pixel_points = (K @ cam_points.T).T
    pixel_points = pixel_points[:, :2] / pixel_points[:, 2:3]
    return pixel_points.detach().cpu().float().numpy()

def run_sam_video_on_batch(rendered_images, checkpoint_path, model_cfg, interior_3d, viewmats, Ks, device):
    """
    Runs SAM 2.1 Video Predictor by pointing it to a high-speed temp directory.
    """
    print("Initializing SAM 2.1 Video Predictor...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

    # SAM's video model requires a directory of images as input. Helps with automatic cleanup and running in RAM?
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {len(rendered_images)} frames...")

        for i in range(len(rendered_images)):
            img_np = (rendered_images[i].detach().clamp(0, 1) * 255).byte().cpu().numpy()
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(temp_dir, f"{i:05d}.jpg"), quality=85) # Reduced quality saves I/O time

        # Initialize the video inference state
        print("Initializing SAM state from folder...")
        inference_state = predictor.init_state(video_path=temp_dir)

        # Add interior points as an initial prompt
        input_points = project_points(interior_3d, viewmats[0], Ks[0])
        input_labels = np.ones(len(input_points), dtype=np.int32)

        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=1,
            points=input_points, labels=input_labels
        )

        # Propagate
        print("Propagating masks...")
        total_frames = len(rendered_images)
        final_masks = [None] * total_frames

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
            final_masks[out_frame_idx] = mask

        return np.stack(final_masks)
    


def plot_training_metrics(history, filename="training_metrics.png"):
    """
    Graphs the Mask Loss and Smoothness Loss over the iterations.
    'history' should be a dictionary containing lists of values.
    """
    iters = range(len(history['total_loss']))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Mask Loss on the left Y-axis
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mask BCE Loss', color=color)
    ax1.plot(iters, history['mask_loss'], color=color, label='Mask Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second Y-axis for the Smoothness Loss
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Smoothness Loss', color=color)
    ax2.plot(iters, history['smooth_loss'], color=color, label='Smoothness', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Voxel Optimization Metrics')
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Metrics graph saved to {filename}")
    plt.show()

import polyscope as ps

def visualize_with_polyscope(masked_rgbs, viewmats, Ks, phi, center, radius, iso_level=0.0):
    """Visualizes the optimized phi grid and camera frustums using masked RGB views."""
    ps.init()
    ps.set_up_dir("neg_y_up")

    # Phi Grid Registration
    bound_low = (center - radius).detach().cpu().numpy()
    bound_high = (center + radius).detach().cpu().numpy()
    phi_data = phi.detach().cpu().numpy().transpose(2, 1, 0)

    mask = phi_data < iso_level
    idx = np.argwhere(mask) # (N,3) voxel indices
    res = phi_data.shape[0] # grid resolution
    points_local = (idx / (res - 1)) * 2 - 1 # map [0,res-1] → [-1,1]
    points_world = center.detach().cpu().numpy() + points_local * radius # Map from local [-1,1] to world

    ps_pts = ps.register_point_cloud("Phi Voxel Nodes", points_world, radius=0.0025, color=(1.0, 0.9, 0.1))
    ps_pts.add_scalar_quantity("phi_val", phi_data[mask], cmap='coolwarm')

    ps_grid = ps.register_volume_grid("Phi Grid", phi_data.shape, bound_low, bound_high)
    ps_grid.add_scalar_quantity(
        "phi",
        phi_data,
        defined_on='nodes',
        cmap='coolwarm',
        enabled=True,
        enable_isosurface_viz=True, # Surface extraction
        isosurface_level=iso_level, # Level set
        isosurface_color=(0.2, 0.5, 0.8), # RGB color for the isosurface
        enable_gridcube_viz=False # This tends to obscure the mesh, ignore.
    )

    # Camera Registration with Billboard Images
    centers, rights, ups, forwards = [], [], [], []
    for i in range(len(viewmats)):
        c2w = torch.linalg.inv(viewmats[i]).detach().cpu().numpy()
        # 1. Extract raw coordinates
        root = c2w[:3, 3]

        # 2. OpenCV to OpenGL conversion
        look_dir = c2w[:3, 2]   # Z is forward
        up_dir = -c2w[:3, 1]    # Invert OpenCV Y (Down) to OpenGL Y (Up)
        right_dir = c2w[:3, 0]  # X is right

        centers.append(root)
        rights.append(right_dir)
        ups.append(up_dir)
        forwards.append(look_dir)

        # Calculate FovY from focal length
        focal_px = Ks[i, 0, 0].item()
        fov_y = 2 * math.atan(args.height / (2 * focal_px)) * (180 / np.pi)
        
        params = ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=fov_y, aspect=args.width/args.height),
            ps.CameraExtrinsics(root=root, look_dir=look_dir, up_dir=up_dir)
        )
        
        cam = ps.register_camera_view(f"Cam_{i}", params)
        cam.set_widget_focal_length(0.05)
        cam.set_widget_color((0.5, 0.5, 0.5))

        cam.add_color_image_quantity(f"MaskedView_{i}", masked_rgbs[i], enabled=True, show_in_camera_billboard=True)

    ps_cloud = ps.register_point_cloud("Cam-ctr", np.array(centers), enabled=False)
    ps_cloud.add_vector_quantity("Cam-forward", np.array(forwards), color=(0.8, 0.2, 0.2))
    ps_cloud.add_vector_quantity("Cam-right", np.array(rights), color=(0.2, 0.8, 0.2))
    ps_cloud.add_vector_quantity("Cam-up", np.array(ups), color=(0.2, 0.2, 0.8))
    ps.show()

def render_in_chunks(means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree, backgrounds, chunk_size=50):
    """Renders the given views in batches to manage GPU memory usage. Huge batches tend to allocate memory for all views at once"""
    all_renders = []
    num_views = len(viewmats)
    
    for i in range(0, num_views, chunk_size):
        end = min(i + chunk_size, num_views)
        print(f"Rendering batch {i} to {end}...")
        
        # Rasterize just this chunk
        renders, alphas, meta = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors[i:end], # Slice the colors for this batch
            viewmats=viewmats[i:end], 
            Ks=Ks[i:end],
            width=width, height=height,
            sh_degree=sh_degree,
        )
        
        all_renders.append(renders)
        del renders, alphas, meta # Explicitly free as memory saving measure
        gc.collect()
        torch.cuda.empty_cache()
        
    return torch.cat(all_renders, dim=0)

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

    os.makedirs(args.output_render, exist_ok=True)

    print_gpu_memory()

    means, scales, quats, colors, opacities = load_ply(args.input, device)

    print_gpu_memory()

    # Known 3D points on the interior of the object to segment (FOR THE TRUCK SPLAT)
    interior_3d = torch.tensor([
        [ 2.293, -0.090, 0.407], [ 2.347, -0.697, 0.407],
        [ 0.574, -0.697, 0.407], [ 0.628,  0.264, 0.407],
        [-2.866, -0.658, 0.407], [-0.899, -0.580, 0.407],
        [-0.429, -0.138, 0.407]
    ], dtype=torch.float32, device=device)

    # Derive grid properties from known interior points
    target_points_min = interior_3d.min(dim=0).values
    target_points_max = interior_3d.max(dim=0).values
    target_center = (target_points_min + target_points_max) / 2.0
    target_radius = (target_points_max - target_points_min).max().item() * 1.2 # Add some padding since internal points aren't all at the boundaries

    print(f"Target object information:")
    print(f"    Center: {target_center.tolist()}")
    print(f"    Radius: {target_radius:.4f}")

    cam_radius = target_radius * args.cam_radius_mul
    print(f"Setting camera radius to {cam_radius:.4f}")

    grid_radius = target_radius * args.grid_radius_mul
    print(f"Setting voxel grid radius to {grid_radius:.4f}")

    # Compute camera intrinsics
    Ks = get_batch_Ks(args.focal, args.width, args.height, num_views=args.num_views, device=device)

    # Compute camera extrinsics
    viewmats = get_batch_viewmats(means, target_center, torch.tensor(cam_radius, device=device, dtype=torch.float32), num_views=args.num_views)
    print_gpu_memory()

    # Perform rasterization on the gaussian splat
    print(f"Rendering {len(viewmats)} views...")
    rendered_images = render_in_chunks(
        means=means,
        quats=quats,
        scales=torch.exp(scales), # Scales stored logarithmically
        opacities=opacities,
        colors=colors[None, :, :].expand(args.num_views, -1, -1), # Expand colors for batch
        viewmats=viewmats, # Extrinsics
        Ks=Ks, # Intrinsics
        width=args.width,
        height=args.height,
        sh_degree=None, # Spherical harmonics degree
        backgrounds=torch.zeros((args.height, args.width, 3), device=device) # Black background
    )

    print_gpu_memory()

    # Run SAM on the batch of rendered images
    print(f"Running SAM on {len(viewmats)} rendered views...")
    target_masks = run_sam_video_on_batch(rendered_images, args.sam_checkpoint, args.sam_config, interior_3d, viewmats, Ks, device)

    # Save sam segmented images
    original = rendered_images.detach().float().clamp(0, 1).cpu().numpy()
    masks = target_masks
    masks_expanded = target_masks[..., None]
    masked_rgb_images = (rendered_images.detach().cpu().numpy() * masks_expanded).astype(np.float32)

    original_renders = rendered_images.detach().cpu().numpy()
    masks = target_masks

    blended_images = []
    for i in range(args.num_views):
        tinted = np.zeros((*original_renders.shape[1:3], 3), dtype=np.float32)
        tinted[..., 0] = 1.0
        alpha_map = (masks[i] * 0.5)[..., None].astype(np.float32)
        blended = original_renders[i] * (1.0 - alpha_map) + tinted * alpha_map
        blended_images.append(blended)

    print("Cleaning up rasterized memory")
    del rendered_images
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize fVDB voxel grid and optimizer
    grid = fVDBOptimizerGrid(args.grid_resolution, target_center, grid_radius, device)
    optimizer = torch.optim.Adam(grid.parameters(), lr=args.lr)

    print("Memory Report before optimization:")
    print_gpu_memory()
    print_tensor_memory(grid.phi, "fVDB Phi Features")
    print_tensor_memory(viewmats, "View Matrices")
    print_tensor_memory(Ks, "Camera Intrinsics")
    print_tensor_memory(target_masks, "SAM Target Masks")
    print_tensor_memory(blended_images, "Blended Images")

    print(f"Starting optimization for {args.num_iters} iterations...")

    history = {
        'total_loss': [],
        'mask_loss': [],
        'smooth_loss': []
    }

    # Per epochs
    for iter in range(args.num_iters):
        total_loss = 0.0
        total_mask_loss = 0.0

        # Stochastic batching
        batch_indices = torch.randperm(args.num_views)[:args.batch_size].tolist()
        for view_idx in batch_indices:
            # Perform volumetric rendering of the current phi grid to get predicted mask
            # print_gpu_memory()
            ray_origins, ray_dirs = construct_rays(viewmats[view_idx], Ks[view_idx], args.height, args.width, device)
            points = sample_points_along_rays(ray_origins, ray_dirs, args.num_samples)
            phi_vals = grid.query(points)
            
            alpha = torch.sigmoid(-args.sharpness * phi_vals)
            pred_mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)

            # Compute BCE loss with SAM mask as target
            target = torch.from_numpy(target_masks[view_idx]).float().to(device).view(-1)
            mask_loss = F.binary_cross_entropy(pred_mask, target)

            view_loss = mask_loss / args.num_views
            total_loss += view_loss
            total_mask_loss += view_loss.item()

        # Add smoothness regularization
        smooth_term = args.beta * grid.tv_loss()
        total_loss += smooth_term

        # Backprop and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute information about phi
        phi_min = grid.phi.min().item()
        phi_max = grid.phi.max().item()
        phi_mean = grid.phi.mean().item()
        phi_median = torch.median(grid.phi).item()
        phi_midrange = (phi_min + phi_max) / 2.0
        print(f"Iter {iter+1}/{args.num_iters}: Loss={total_loss.item():.4f}, Mask={total_mask_loss:.6f}, Smooth={smooth_term.item():.6f}")
        print(f"fVDB Phi stats: [{phi_min:.4f}, {phi_max:.4f}, {phi_mean:.4f}, {phi_median:.4f}, {phi_midrange:.4f}]")
        print_gpu_memory()

        history['total_loss'].append(total_loss.item())
        history['mask_loss'].append(total_mask_loss)
        history['smooth_loss'].append(smooth_term.item())

    print("Optimization complete. Visualizing training history...")

    # flattened_masks = [mask for iter_row in all_debug_masks for mask in iter_row]
    # visualize_batch_grid(flattened_masks, num_cols=args.num_views)

    # Graph the results
    plot_training_metrics(history, filename="truck_optimization_log.png")

    print("\nReconstructing dense grid for Polyscope visualization...")
    res = args.grid_resolution
    
    z_lin = torch.linspace(-1, 1, res, device=device)
    y_lin = torch.linspace(-1, 1, res, device=device)
    x_lin = torch.linspace(-1, 1, res, device=device)
    z, y, x = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
    
    dense_pts_local = torch.stack([x, y, z], dim=-1)
    dense_pts_world = target_center + dense_pts_local * grid_radius
    
    with torch.no_grad():
        dense_phi_flat = grid.query(dense_pts_world.reshape(-1, 3))
        dense_phi = dense_phi_flat.reshape(res, res, res)

    print("Visualizing vertices in polyscope")
    visualize_with_polyscope(
        blended_images, 
        viewmats, 
        Ks, 
        dense_phi, 
        target_center, 
        grid_radius, 
        iso_level=args.iso_level
    )

    # -----------------------------------------------------------------------
    # Final Evaluation Renders
    # -----------------------------------------------------------------------
    print("Generating unseen views for evaluation...")
    num_test_views = args.num_test_views
    test_Ks = get_batch_Ks(args.focal, args.width, args.height, num_test_views, device=device)
    test_viewmats = get_batch_viewmats(means, target_center, torch.tensor(cam_radius, device=device), num_test_views)

    test_renders, _, _ = rasterization(
        means=means, quats=quats, scales=torch.exp(scales), opacities=opacities,
        colors=colors[None, :, :].expand(num_test_views, -1, -1),
        viewmats=test_viewmats, Ks=test_Ks,
        width=args.width, height=args.height, sh_degree=None
    )

    phi_renders = render_phi_to_image(grid, test_viewmats, test_Ks, args.height, args.width, args.sharpness, device)

    # Visualize the new views and SAM masks
    original_imgs = test_renders.detach().float().clamp(0, 1).cpu().numpy()
    phi_masks = phi_renders.detach().float().cpu().numpy()
    phi_masks_rgb = np.repeat(phi_masks[:, :, :, None], 3, axis=-1)

    combined = np.concatenate([original_imgs, phi_masks_rgb], axis=0)
    visualize_batch_grid(combined, num_cols=num_test_views)