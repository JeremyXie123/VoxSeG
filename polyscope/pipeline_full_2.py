# Purpose of this file:
# Continuation of other pipeline file, with the voxelization part added

import math
import torch
import os
import numpy as np
from plyfile import PlyData
from gsplat import rasterization
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
from skimage import measure

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

def get_batch_viewmats(means, center, distance, num_views):
    """Generates a batch of camera poses orbitting around the given center at the given distance, looking towards the center"""    
    viewmats = []
    for i in range(num_views):
        angle = (2 * np.pi / num_views) * i
        # Orbit around vertical axis (Y)
        cam_pos = center + torch.tensor([distance * np.cos(angle), 0, distance * np.sin(angle)], device=means.device)
        
        # Compute camera extrinsics
        z = (center - cam_pos) # Forward
        z /= torch.norm(z)
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device=means.device)
        x = torch.linalg.cross(up, z) # Rightward
        x /= torch.norm(x)
        y = torch.linalg.cross(z, x) # Downward
        y /= torch.norm(y)

        R = torch.stack([x, y, z], dim=0) # [3x3] Rotation matrix stacked row by row
        T = -R @ cam_pos # [3x1] Translation
        
        mat = torch.eye(4, device=means.device) # [4x4] Homogenous transform
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
        
    plt.tight_layout()
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

def run_sam_on_batch(rendered_images, checkpoint_path, model_cfg, interior_3d, device):
    """Runs the Segment Anything Model on a batch of rendered images and returns the predicted masks"""
    # Initialize SAM 2.1
    model = build_sam2(os.path.abspath(model_cfg), checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)

    batched_masks = []
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(rendered_images.shape[0]):
            print(f"Processing View {i+1} with 7 projected points (Single Pass)...")
            
            input_points = project_points(interior_3d, viewmats[i], Ks[i])
            input_labels = np.ones(len(input_points), dtype=np.int32)

            # Convert render to format SAM 2.1 expects
            img_np = (rendered_images[i].detach().clamp(0, 1) * 255).byte().cpu().numpy()
            predictor.set_image(img_np)

            # Run inference using the backward projected 3d points
            masks, _, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            
            batched_masks.append(masks[0])
            
    return np.stack(batched_masks)

def init_phi_grid(resolution, device):
    """Initializes a voxel grid of the given resolution centered at the given point and with the given radius"""
    nx, ny, nz = resolution, resolution, resolution
    phi = torch.nn.Parameter(torch.randn((nz, ny, nx), device=device) * 0.1 + 0.5)
    return phi

def construct_rays(viewmat, K, height, width, device):
    """Generates rays (origin and direction) for each pixel in the image."""
    # Create a grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    
    # Invert the intrinsic matrix to go from pixels to camera space
    inv_K = torch.linalg.inv(K)
    pixels_homo = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # [H, W, 3]
    
    # Directions in camera space
    cam_dirs = (inv_K @ pixels_homo.reshape(-1, 3).T).T # [H*W, 3]
    
    # Camera to World transform
    cam_to_world = torch.linalg.inv(viewmat)
    ray_origins = cam_to_world[:3, 3].expand(cam_dirs.shape[0], -1) # Origin is camera center
    
    # Rotate directions to world space
    ray_dirs = (cam_to_world[:3, :3] @ cam_dirs.T).T
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    
    return ray_origins, ray_dirs # [H*W, 3]

def sample_points_along_rays(ray_origins, ray_dirs, num_samples, near=0.1, far=10.0):
    """Samples points along rays between near and far planes."""
    t_vals = torch.linspace(near, far, num_samples, device=ray_origins.device) # [num_samples]
    points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None] # [H*W, 3]
    return points

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

from skimage import measure

def save_reconstruction_to_obj(phi, filename="truck_mesh.obj"):
    """
    Extracts a 3D manifold mesh where phi=0 and saves as an OBJ file.
    """
    grid_np = phi.detach().cpu().numpy()
    
    # Level=0.0 is our surface. Values < 0 are "inside" the truck.
    verts, faces, normals, values = measure.marching_cubes(grid_np, level=0.0)

    # 4. Write the OBJ file
    with open(filename, 'w') as f:
        f.write("# Truck Reconstruction Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ indices are 1-based, so we add 1 to the 0-based faces
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Successfully saved {len(verts)} vertices to {filename}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="splats/truck.ply", help="Path to input PLY file")
    parser.add_argument("--output_render", type=str, default="rendered", help="Directory to save rendered images")
    parser.add_argument("--sam_checkpoint", type=str, default="C:\\Users\\Jeremy\\Desktop\\CSC494\\CSC494\\polyscope\\sam2.1_hiera_l.pt", help="Path to SAM checkpoint")
    parser.add_argument("--sam_config", type=str, default="C:\\Users\\Jeremy\\Desktop\\CSC494\\CSC494\\polyscope\\sam2.1_hiera_l.yaml", help="Path to SAM config")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_render, exist_ok=True)

    means, scales, quats, colors, opacities = load_ply(args.input, device)

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
    print(f"  Center: {target_center.tolist()}")
    print(f"  Radius: {target_radius:.4f}")

    cam_radius = target_radius * args.cam_radius_mul
    print(f"Setting camera radius to {cam_radius:.4f}")

    grid_radius = target_radius * args.grid_radius_mul
    print(f"Setting voxel grid radius to {grid_radius:.4f}")

    # Compute camera intrinsics
    Ks = get_batch_Ks(args.focal, args.width, args.height, num_views=args.num_views, device=device)

    # Compute camera extrinsics
    viewmats = get_batch_viewmats(means, target_center, torch.tensor(cam_radius, device=device, dtype=torch.float32), num_views=args.num_views)

    # Perform rasterization on the gaussian splat
    print(f"Rendering {len(viewmats)} views...")
    rendered_images, alphas, meta = rasterization(
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
        backgrounds=torch.zeros((args.num_views, 3), device=device) # Black background
    )

    # Run SAM on the batch of rendered images
    print(f"Running SAM on {len(viewmats)} rendered views...")
    target_masks = run_sam_on_batch(rendered_images, args.sam_checkpoint, args.sam_config, interior_3d, device)

    # Visualize the views and what SAM is predicting
    masked_images = rendered_images.cpu().numpy() * target_masks[:, :, :, None]
    original = rendered_images.detach().float().clamp(0, 1).cpu().numpy()
    masks_rgb = np.repeat(target_masks[:, :, :, None], 3, axis=-1).astype(float)
    segmented = original * masks_rgb
    stacked_images = np.concatenate([original, masks_rgb, segmented], axis=0)
    visualize_batch_grid(stacked_images, num_cols=args.num_views, axes_pad=0.05)

    # Initialize voxel grid and optimizer
    phi = init_phi_grid(args.grid_resolution, device)
    optimizer = torch.optim.Adam([phi], lr=args.lr)

    print(f"Starting optimization for {args.num_iters} iterations...")
    
    history = {
        'total_loss': [],
        'mask_loss': [],
        'smooth_loss': []
    }

    all_debug_masks = []
    for iter in range(args.num_iters):
        print(f"Iteration {iter+1}/{args.num_iters}...")

        total_loss = 0.0
        total_mask_loss = 0.0

        current_iter_masks = []
        for view_idx in range(args.num_views):
            ray_origins, ray_dirs = construct_rays(viewmats[view_idx], Ks[view_idx], args.height, args.width, device)
            points = sample_points_along_rays(ray_origins, ray_dirs, args.num_samples)
            phi_vals = query_phi_trilinear(phi, points, target_center, grid_radius)
            alpha = torch.sigmoid(-args.sharpness * phi_vals)
            pred_mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)

            if iter % 10 == 0:
                mask_2d = pred_mask.reshape(args.height, args.width).detach().cpu().numpy()
                current_iter_masks.append(mask_2d)

            # Compute BCE loss with SAM mask as target
            target = torch.from_numpy(target_masks[view_idx]).float().to(device).view(-1)
            mask_loss = F.binary_cross_entropy(pred_mask, target)

            view_loss = mask_loss / args.num_views
            total_loss += view_loss
            total_mask_loss += view_loss.item()

        all_debug_masks.append(current_iter_masks)

        # Add smoothness regularization
        smooth_term = args.beta * dirichlet_energy(phi)
        total_loss += smooth_term

        # Backprop and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Iter {iter}: Loss={total_loss.item():.4f}, Mask={total_mask_loss:.6f}, Smooth={smooth_term.item():.6f}")
        print(f"Phi Range: [{phi.min().item():.4f}, {phi.max().item():.4f}]")

        history['total_loss'].append(total_loss.item())
        history['mask_loss'].append(total_mask_loss)
        history['smooth_loss'].append(smooth_term.item())

    print("Optimization complete. Visualizing training history...")
    
    # flattened_masks = [mask for iter_row in all_debug_masks for mask in iter_row]
    # visualize_batch_grid(flattened_masks, num_cols=args.num_views)

    # Final step: Graph the results
    plot_training_metrics(history, filename="truck_optimization_log.png")

    print("Extracting mesh and saving to OBJ...")
    save_reconstruction_to_obj(phi, filename="truck_mesh.obj")