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
    grid_rotation: torch.Tensor  # (3, 3) rotation matrix for oriented bounding box
    viewmats: torch.Tensor
    Ks: torch.Tensor


def get_batch_Ks(focal: float, width: int, height: int, num_views: int, device: torch.device) -> torch.Tensor:
    """Generates a batch of identical camera intrinsics."""
    K = torch.tensor([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    return K.unsqueeze(0).expand(num_views, -1, -1).contiguous()


def get_batch_viewmats(
    center: np.ndarray | torch.Tensor,
    radius: float,
    num_rings: int = 1,
    cameras_per_ring: int = 10,
    elevation_min: float = 15.0,
    elevation_max: float = 15.0,
    up_axis: np.ndarray = None,
    device: torch.device = None,
) -> tuple[torch.Tensor, int]:
    """
    Generates a batch of view matrices for cameras orbiting around a center point.
    
    Cameras are arranged in horizontal rings at different elevations.
    
    Args:
        center: (3,) orbit center point (numpy array or torch tensor)
        radius: distance from center
        num_rings: number of elevation rings
        cameras_per_ring: number of cameras in each ring
        elevation_min: minimum camera elevation angle in degrees
        elevation_max: maximum camera elevation angle in degrees  
        up_axis: (3,) custom up axis for orbit (default: [0, -1, 0] for neg_y_up)
        device: torch device for output tensors
    
    Returns:
        viewmats: (num_views, 4, 4) world-to-camera transformation matrices
        cameras_per_ring: number of cameras per ring (for widget sizing)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert center to numpy if needed
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    center = np.asarray(center, dtype=np.float32)
    
    # Default up axis (neg_y_up convention)
    if up_axis is None:
        up_axis = np.array([0., -1., 0.], dtype=np.float32)
    up_axis = up_axis / np.linalg.norm(up_axis)
    
    # Build a local coordinate frame around the up axis
    if abs(np.dot(up_axis, np.array([1., 0., 0.]))) < 0.9:
        right = np.cross(up_axis, np.array([1., 0., 0.]))
    else:
        right = np.cross(up_axis, np.array([0., 0., 1.]))
    right = right / np.linalg.norm(right)
    forward = np.cross(right, up_axis)
    forward = forward / np.linalg.norm(forward)

    # Compute elevations for each ring
    if num_rings == 1:
        elevations_deg = [(elevation_min + elevation_max) / 2]
    else:
        elevations_deg = np.linspace(elevation_min, elevation_max, num_rings)
    
    viewmats = []
    for ring_idx, el_deg in enumerate(elevations_deg):
        el = np.radians(el_deg)
        
        # Azimuths for this ring - offset alternate rings by half spacing for better coverage
        azimuth_offset = (np.pi / cameras_per_ring) if (ring_idx % 2 == 1) else 0
        azimuths = np.linspace(0, 2 * np.pi, cameras_per_ring, endpoint=False) + azimuth_offset
        
        for az in azimuths:
            # Camera position in local frame, then transform to world
            local_offset = (
                np.cos(el) * np.sin(az) * right -
                np.cos(el) * np.cos(az) * forward +
                np.sin(el) * up_axis
            ) * radius
            
            cam_pos = center + local_offset

            # Forward: camera looks toward center
            z = center - cam_pos
            z = z / np.linalg.norm(z)

            # Handle gimbal lock when z is nearly parallel to up_axis
            world_up = up_axis
            if abs(np.dot(z, world_up)) > 0.99:
                world_up = forward

            x = np.cross(z, world_up)
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            # Build world-to-camera matrix
            R = np.stack([x, y, z], axis=0)  # rows = cam X, Y, Z axes
            t = -R @ cam_pos

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            viewmats.append(w2c)

    return torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device), cameras_per_ring


def compute_widget_focal_length(radius: float, cameras_per_ring: int, scale: float = 0.3) -> float:
    """
    Compute widget focal length to prevent camera overlap in Polyscope.
    
    Args:
        radius: Camera orbit radius
        cameras_per_ring: Number of cameras in each elevation ring
        scale: Fraction of max size to use (0.3 = 30% of space between cameras)
    
    Returns:
        Recommended widget focal length
    """
    # Arc distance between adjacent cameras in a ring
    arc_distance = radius * (2 * np.pi / cameras_per_ring)
    
    # Widget should be a fraction of half the arc distance
    return arc_distance * scale * 0.5


def compute_orbit_radius(box_size: np.ndarray | list, focal_length: float = 550.0, 
                         image_size: int = 512, padding: float = 1.0) -> float:
    """
    Compute the minimum orbit radius needed to see the entire bounding box.
    
    Args:
        box_size: (3,) array of box dimensions [x, y, z]
        focal_length: camera focal length in pixels
        image_size: image height/width in pixels
        padding: multiplier for extra margin (1.0 = tight fit)
    
    Returns:
        radius: orbit distance from box center
    """
    box_size = np.asarray(box_size)
    diagonal = np.linalg.norm(box_size)
    min_distance = (diagonal / 2) * focal_length / (image_size / 2)
    return min_distance * padding


def project_points(points_3d: torch.Tensor, viewmat: torch.Tensor, K: torch.Tensor) -> np.ndarray:
    """Projects 3D points to 2D pixel coordinates using camera extrinsics and intrinsics."""
    # World to Camera Space
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)
    cam_points = (viewmat @ points_homo.T).T[:, :3]
    
    # Camera to Image Plane
    pixel_points = (K @ cam_points.T).T
    pixel_points = pixel_points[:, :2] / pixel_points[:, 2:3]
    return pixel_points.detach().cpu().float().numpy()


def construct_rays(viewmat: torch.Tensor, K: torch.Tensor, height: int, width: int, 
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rays (origin and direction) for each pixel in the image."""
    y, x = torch.meshgrid(torch.arange(height, device=device), 
                          torch.arange(width, device=device), indexing="ij")
    
    # K maps [X_cam, Y_cam, Z_cam] to [u, v, 1], apply inverse to get camera directions from pixels
    inv_K = torch.linalg.inv(K)
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    cam_dirs = pixels @ inv_K.T 

    # Rotate ray directions based on viewmat rotation
    cam_to_world = torch.linalg.inv(viewmat)
    ray_dirs = cam_dirs @ cam_to_world[:3, :3].T
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    
    # Move ray origins based on viewmat translation
    ray_origins = cam_to_world[:3, 3].expand(height, width, 3)
    
    # Return as [N, 3] for use with sampling function
    return ray_origins.reshape(-1, 3), ray_dirs.reshape(-1, 3)