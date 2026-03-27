import math
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraState:
    """A clean container for all camera and grid geometry."""
    target_center: torch.Tensor
    target_radius: float
    cam_radius: float
    grid_radius: float
    viewmats: torch.Tensor
    Ks: torch.Tensor

def get_batch_viewmats(means: torch.Tensor, center: torch.Tensor, distance: float, num_views: int, num_rotations: int = 4, max_pitch: float = np.pi/9) -> torch.Tensor:
    """Generates a batch of OpenCV LookAt view matrices."""
    viewmats = []
    
    for i in range(num_views):
        yaw = (2 * np.pi * num_rotations / num_views) * i
        pitch = (max_pitch / (num_views - 1)) * i 
        
        # Remove the negative sign on pitch so cameras climb ABOVE the object
        x = distance * np.cos(pitch) * np.cos(yaw)
        y = distance * np.sin(-pitch)
        z = distance * np.cos(pitch) * np.sin(yaw)
        
        cam_pos = center + torch.tensor([x, y, z], dtype=torch.float32, device=means.device)
        
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

def get_batch_Ks(focal: float, width: int, height: int, num_views: int, device: torch.device) -> torch.Tensor:
    """Generates a batch of identical camera intrinsics."""
    Ks = torch.tensor([
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ], device=device).repeat(num_views, 1, 1)
    return Ks

def project_points(points_3d: torch.Tensor, viewmat: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    """Projects 3D points to 2D pixel coordinates using camera extrinsics and intrinsics."""
    # World to Camera Space
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)
    cam_points = (viewmat @ points_homo.T).T[:, :3]
    
    # Camera to Image Plane
    pixel_points = (K @ cam_points.T).T
    pixel_points = pixel_points[:, :2] / pixel_points[:, 2:3]
    return pixel_points.detach().cpu().float().numpy()

def construct_rays(viewmat: torch.Tensor, K: torch.Tensor, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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

def setup_camera_geometry(interior_3d: torch.Tensor, means: torch.Tensor, args, device: torch.device) -> CameraState:
    """Wrapper function to derive all target bounds and initialize camera matrices."""
    # Derive grid properties from known interior points
    target_points_min = interior_3d.min(dim=0).values
    target_points_max = interior_3d.max(dim=0).values
    target_center = (target_points_min + target_points_max) / 2.0
    target_radius = (target_points_max - target_points_min).max().item() * 1.2 

    cam_radius = target_radius * args.cam_radius_mul
    grid_radius = target_radius * args.grid_radius_mul

    print(f"Target object information:")
    print(f"    Center: {target_center.tolist()}")
    print(f"    Radius: {target_radius:.4f}")
    print(f"Setting camera radius to {cam_radius:.4f}")
    print(f"Setting voxel grid radius to {grid_radius:.4f}")

    # Compute matrices
    Ks = get_batch_Ks(args.focal, args.width, args.height, args.num_views, device)
    viewmats = get_batch_viewmats(means, target_center, cam_radius, args.num_views)

    # Return everything neatly bundled
    return CameraState(
        target_center=target_center,
        target_radius=target_radius,
        cam_radius=cam_radius,
        grid_radius=grid_radius,
        viewmats=viewmats,
        Ks=Ks
    )