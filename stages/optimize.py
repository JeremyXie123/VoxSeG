import torch
import torch.nn.functional as F

# Import our custom types and camera math
from core.camera import CameraState, construct_rays
from core.splat_io import print_gpu_memory
from stages.segmentation import SegmentationResult

def init_phi_grid(resolution: int, device: torch.device) -> torch.nn.Parameter:
    """Initializes a voxel grid of the given resolution as a trainable PyTorch parameter."""
    phi = torch.nn.Parameter(torch.randn((resolution, resolution, resolution), device=device) * 0.1 + 0.5)
    return phi

def sample_points_along_rays(ray_origins: torch.Tensor, ray_dirs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Samples points along rays between near and far planes."""
    t_vals = torch.linspace(0.1, 10.0, num_samples, device=ray_origins.device) 
    points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None] 
    return points

def query_phi_trilinear(phi: torch.Tensor, points: torch.Tensor, grid_center: torch.Tensor, grid_radius: float) -> torch.Tensor:
    """Queries the voxel grid at 3D points using trilinear interpolation."""
    center = grid_center.to(points.device)
    points_norm = (points - center) / grid_radius
    
    # grid_sample expects [N, C, D, H, W]
    grid = phi[None, None, ...] 
    N_rays, N_samples, _ = points_norm.shape
    sampling_coords = points_norm.reshape(1, N_rays * N_samples, 1, 1, 3)
    
    vals = F.grid_sample(grid, sampling_coords, mode='bilinear', padding_mode='border', align_corners=True)
    return vals.reshape(N_rays, N_samples)

def dirichlet_energy(phi: torch.Tensor) -> torch.Tensor:
    """Computes the Dirichlet energy (L2 norm of gradients) as a smoothness prior."""
    dx = phi[1:, :, :] - phi[:-1, :, :]
    dy = phi[:, 1:, :] - phi[:, :-1, :]
    dz = phi[:, :, 1:] - phi[:, :, :-1]
    
    return (dx**2).mean() + (dy**2).mean() + (dz**2).mean()

def render_phi_to_image(phi: torch.Tensor, cams: CameraState, args, device: torch.device) -> torch.Tensor:
    """Renders the phi grid as an image (masks) using volume rendering."""
    renderings = []
    for i in range(len(cams.viewmats)):
        ray_origins, ray_dirs = construct_rays(cams.viewmats[i], cams.Ks[i], args.height, args.width, device)
        points = sample_points_along_rays(ray_origins, ray_dirs, num_samples=args.num_test_samples) 
        phi_vals = query_phi_trilinear(phi, points, cams.target_center, cams.grid_radius)
        alpha = torch.sigmoid(-args.sharpness * phi_vals)
        mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)
        renderings.append(mask.reshape(args.height, args.width))
    return torch.stack(renderings)

def optimize_voxel_grid(seg_result: SegmentationResult, cams: CameraState, args, device: torch.device) -> torch.Tensor:
    """
    The main training loop. Casts rays through the grid, compares the rendered
    opacities against the SAM 2 masks, and updates the voxel weights.
    """
    print("Initializing voxel grid and optimizer...")
    phi = init_phi_grid(args.grid_resolution, device)
    optimizer = torch.optim.Adam([phi], lr=args.lr)

    print(f"Starting optimization for {args.num_iters} iterations...")

    history = {'total_loss': [], 'mask_loss': [], 'smooth_loss': []}
    
    for iter in range(args.num_iters):
        total_loss = 0.0
        total_mask_loss = 0.0

        # Stochastic batching
        batch_indices = torch.randperm(args.num_views)[:args.batch_size].tolist()
        
        for view_idx in batch_indices:
            # Perform volumetric rendering of the current phi grid to get predicted mask
            ray_origins, ray_dirs = construct_rays(cams.viewmats[view_idx], cams.Ks[view_idx], args.height, args.width, device)
            points = sample_points_along_rays(ray_origins, ray_dirs, args.num_samples)
            phi_vals = query_phi_trilinear(phi, points, cams.target_center, cams.grid_radius)
            alpha = torch.sigmoid(-args.sharpness * phi_vals)
            pred_mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)

            # Compute BCE loss with SAM mask as target
            target = torch.from_numpy(seg_result.masks[view_idx]).float().to(device).view(-1)
            mask_loss = F.binary_cross_entropy(pred_mask, target)

            view_loss = mask_loss / args.num_views
            total_loss += view_loss
            total_mask_loss += view_loss.item()

        # 4. Apply Regularization
        smooth_term = args.beta * dirichlet_energy(phi)
        total_loss += smooth_term

        # 5. Backpropagate & Step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute information about phi
        phi_min = phi.min().item()
        phi_max = phi.max().item()
        phi_mean = phi.mean().item()
        phi_median = torch.median(phi).item()
        phi_midrange = (phi_min + phi_max) / 2.0
        print(f"Iter {iter+1}/{args.num_iters}: Loss={total_loss.item():.4f}, Mask={total_mask_loss:.6f}, Smooth={smooth_term.item():.6f}")
        print(f"Phi min/max/avg/med/mid: [{phi_min:.4f}, {phi_max:.4f}, {phi_mean:.4f}, {phi_median:.4f}, {phi_midrange:.4f}]")
        print_gpu_memory()
        
        history['total_loss'].append(total_loss.item())
        history['mask_loss'].append(total_mask_loss)
        history['smooth_loss'].append(smooth_term.item())

    print("Optimization complete.")
    return phi, history
