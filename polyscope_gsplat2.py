import torch
import os
import sys
import os.path as path

# Path to where the bindings live
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
if os.name == 'nt': # if Windows
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "Debug")))
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))

import polyscope as ps
import polyscope.imgui as psim
from polyscope import imgui as psim

import sys
import argparse
import numpy as np
from plyfile import PlyData
from PIL import Image, ImageDraw
from gsplat import rasterization
from core.splat_io import load_ply, print_gpu_memory


def load_gaussians_from_ply(path_ply, device='cuda'):

    plydata = PlyData.read(path_ply)

    centers = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])

    features_dc = np.zeros((centers.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((centers.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((centers.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((centers.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    with torch.no_grad():
        centers  = torch.tensor(centers,   dtype=torch.float, device=device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous()
        opacity  = torch.tensor(opacities, dtype=torch.float, device=device)
        scaling  = torch.tensor(scales,    dtype=torch.float, device=device)
        rotation = torch.tensor(rots,      dtype=torch.float, device=device)

    opacity = torch.sigmoid(opacity)
    scaling = torch.exp(scaling)

    return centers, features_dc, opacity, scaling, rotation


# ---------------------------------------------------------------------------
# SAM3 prompt box
# ---------------------------------------------------------------------------
sam_box = {
    "label": [""],
    "transform": [np.eye(4, dtype=np.float32)],  # 4x4 transform matrix
    "focal_length": [550.0],  # default, will be set from args
    "padding": [1.0],  # default, will be set from args
    "elevation_min": [15.0],  # default, will be set from args
    "elevation_max": [15.0],  # default, will be set from args
}

_BOX_NAME = "Prompt Box"


def create_unit_box():
    """Create a unit cube centered at origin with vertices at ±0.5, plus axis indicators."""
    # Box corners
    corners = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
        # Center point
        [ 0.0,  0.0,  0.0],  # 8 (center)
        # Face centers for axis indicators (right-handed basis)
        [ 0.5,  0.0,  0.0],  # 9  (+X) - red
        [ 0.0, -0.5,  0.0],  # 10 (-Y = UP in neg_y_up) - green
        [ 0.0,  0.0, -0.5],  # 11 (-Z) - blue (right-handed: X cross Y = -Z)
    ], dtype=np.float32)
    
    edges = np.array([
        # Box edges - bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Box edges - top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Box edges - vertical
        [0, 4], [1, 5], [2, 6], [3, 7],
        # Axis indicators from center to face centers
        [8, 9],   # X axis (red)
        [8, 10],  # Y axis (green) - points UP in neg_y_up convention
        [8, 11],  # Z axis (blue)
    ])
    
    return corners, edges


def register_prompt_box():
    """Register the prompt box as a curve network with gizmo enabled."""
    corners, edges = create_unit_box()
    
    box_net = ps.register_curve_network(_BOX_NAME, corners, edges, radius=0.002)
    box_net.set_color((1.0, 0.5, 0.0))  # orange
    
    # Color the axis indicator edges
    edge_colors = np.zeros((len(edges), 3), dtype=np.float32)
    edge_colors[:12] = [1.0, 0.5, 0.0]  # orange for box edges
    edge_colors[12] = [1.0, 0.0, 0.0]   # red for X axis
    edge_colors[13] = [0.0, 1.0, 0.0]   # green for Y axis
    edge_colors[14] = [0.0, 0.0, 1.0]   # blue for Z axis
    box_net.add_color_quantity("axis_colors", edge_colors, defined_on='edges', enabled=True)
    
    box_net.set_transform(sam_box["transform"][0])
    box_net.set_transform_gizmo_enabled(True)
    
    return box_net


def get_box_center():
    """Get the world-space center of the prompt box."""
    if ps.has_curve_network(_BOX_NAME):
        transform = ps.get_curve_network(_BOX_NAME).get_transform()
        return transform[:3, 3].copy()
    return sam_box["transform"][0][:3, 3].copy()


def get_box_up_axis():
    """Get the world-space up axis (direction the green edge points)."""
    if ps.has_curve_network(_BOX_NAME):
        transform = ps.get_curve_network(_BOX_NAME).get_transform()
    else:
        transform = sam_box["transform"][0]
    
    # Green edge points in -Y direction in local coords (UP in neg_y_up)
    # Column 1 is the Y axis, negate it to get the green edge direction
    y_axis = transform[:3, 1].copy()
    up = -y_axis / np.linalg.norm(y_axis)
    return up


def get_box_transform():
    """Get the full transform of the prompt box."""
    if ps.has_curve_network(_BOX_NAME):
        return ps.get_curve_network(_BOX_NAME).get_transform().copy()
    return sam_box["transform"][0].copy()


def get_box_properties():
    """Extract center, size, and rotation angles from the box transform."""
    transform = get_box_transform()
    center = transform[:3, 3]
    
    # Extract scale from transform matrix (length of each column vector)
    scale_x = np.linalg.norm(transform[:3, 0])
    scale_y = np.linalg.norm(transform[:3, 1])
    scale_z = np.linalg.norm(transform[:3, 2])
    size = np.array([scale_x, scale_y, scale_z])
    
    # Extract rotation matrix (normalize columns)
    R = transform[:3, :3].copy()
    R[:, 0] /= scale_x
    R[:, 1] /= scale_y
    R[:, 2] /= scale_z
    
    # Extract Euler angles (XYZ extrinsic = ZYX intrinsic)
    # This matches the construction order: R = Rz @ Ry @ Rx
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        angle_x = np.arctan2(R[2, 1], R[2, 2])
        angle_y = np.arctan2(-R[2, 0], sy)
        angle_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        angle_x = np.arctan2(-R[1, 2], R[1, 1])
        angle_y = np.arctan2(-R[2, 0], sy)
        angle_z = 0
    
    angles_deg = np.degrees([angle_x, angle_y, angle_z])
    
    return center, size, angles_deg


def get_sam_label():
    return sam_box["label"][0]


def clear_camera_previews():
    """Remove all camera preview objects from Polyscope."""
    for i in range(100):  # Clear up to 100 cameras
        name = f"Cam_{i}"
        if ps.has_camera_view(name):
            ps.remove_camera_view(name)


def preview_cameras(args):
    """Compute orbit cameras and display frustums in Polyscope."""
    import math
    
    # Clear any existing camera previews
    clear_camera_previews()
    
    # Get current box properties
    box_center = get_box_center()
    box_up_axis = get_box_up_axis()
    center, size, angles_deg = get_box_properties()
    
    # Use parameters from sam_box (updated by sliders)
    focal_length = sam_box["focal_length"][0]
    padding = sam_box["padding"][0]
    elevation_min = sam_box["elevation_min"][0]
    elevation_max = sam_box["elevation_max"][0]
    resolution = args.resolution
    
    # Set Polyscope editor FOV to match the focal length
    fov_y_deg = 2 * math.atan(resolution / (2 * focal_length)) * (180 / math.pi)
    ps.set_view_camera_parameters(ps.CameraParameters(
        ps.CameraIntrinsics(fov_vertical_deg=fov_y_deg, aspect=1.0),
        ps.get_view_camera_parameters().get_extrinsics()
    ))
    
    # Compute orbit radius
    if args.orbit_radius is not None:
        orbit_radius = args.orbit_radius
    else:
        orbit_radius = compute_orbit_radius_from_box(size, focal_length=focal_length, image_size=resolution, padding_factor=padding)
    
    # Generate camera views
    viewmats, Ks, W, H = orbit_cameras(
        center=box_center,
        n_views=args.n_views,
        radius=orbit_radius,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        up_axis=box_up_axis,
        focal_length=focal_length,
        resolution=resolution,
    )
    
    # Register camera frustums
    for i in range(len(viewmats)):
        c2w = torch.linalg.inv(viewmats[i]).cpu().numpy()
        root = c2w[:3, 3]
        look_dir = c2w[:3, 2]
        up_dir = -c2w[:3, 1]
        
        params = ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=fov_y_deg, aspect=W/H),
            ps.CameraExtrinsics(root=root, look_dir=look_dir, up_dir=up_dir)
        )
        
        cam = ps.register_camera_view(f"Cam_{i}", params)
        cam.set_widget_focal_length(0.02)
        cam.set_widget_color((0.6, 0.2, 0.8))
    
    print(f"[Preview] {args.n_views} cameras at radius {orbit_radius:.2f}, focal_length {focal_length:.1f}, fov {fov_y_deg:.1f}°, elevation {elevation_min:.1f}°-{elevation_max:.1f}°")


def make_ui_callback(prompt_box, args):
    """Create the UI callback for the prompt box interface."""
    
    def ui_callback():
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if not psim.TreeNode("SAM3 Prompt Box"):
            return

        psim.Text("Object to segment:")
        _, sam_box["label"][0] = psim.InputText("##label", sam_box["label"][0])

        psim.Separator()
        psim.Text("Adjust the orange box to fit the object.")
        psim.Text("Green edge should point UP (orbit axis).")
        psim.Spacing()
        psim.Text("Right-click 'Prompt Box' > Options > Transform:")
        psim.Text("  - Show Transform Window")
        psim.Text("  - Enable Scaling")
        psim.Text("  - Enable Non-Uniform Scaling")
        psim.Text("  - Select 'Local' mode")
        
        psim.Separator()
        
        # Get transform info
        center, size, angles_deg = get_box_properties()
        
        psim.Text(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        psim.Text(f"Size:   ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
        psim.Text(f"Angle:  ({angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f}) deg")
        
        psim.Separator()
        
        # Focal length drag/input
        _, sam_box["focal_length"][0] = psim.DragFloat("Focal Length", sam_box["focal_length"][0], 1.0, 100.0, 1500.0)
        
        # Padding drag/input
        _, sam_box["padding"][0] = psim.DragFloat("Padding", sam_box["padding"][0], 0.05, 0.5, 3.0)
        
        # Elevation min/max drag/input
        _, sam_box["elevation_min"][0] = psim.DragFloat("Elevation Min", sam_box["elevation_min"][0], 1.0, -89.0, 89.0)
        _, sam_box["elevation_max"][0] = psim.DragFloat("Elevation Max", sam_box["elevation_max"][0], 1.0, -89.0, 89.0)
        
        psim.Separator()
        
        # Preview cameras button
        if psim.Button("Preview Cameras"):
            preview_cameras(args)
        
        psim.SameLine()
        if psim.Button("Clear Cameras"):
            clear_camera_previews()
        
        psim.Separator()
        psim.TextColored((0.7, 0.7, 0.7, 1.0), "Axes: Red=X, Green=Y (up), Blue=Z")

        psim.TreePop()

    return ui_callback


# ---------------------------------------------------------------------------
# Camera view generation
# ---------------------------------------------------------------------------

def compute_orbit_radius_from_box(box_size, focal_length=550.0, image_size=512, padding_factor=1.0):
    """
    Compute the minimum orbit radius needed to see the entire bounding box.
    
    Args:
        box_size: (3,) array of box dimensions [x, y, z]
        focal_length: camera focal length in pixels
        image_size: image height in pixels
        padding_factor: multiplier for extra margin (1.0 = no padding)
    
    Returns:
        radius: minimum orbit distance from box center
    """
    # Full diagonal is the longest possible extent from any viewing angle
    diagonal = np.linalg.norm(box_size)
    
    # Calculate distance using focal length
    # (diagonal/2) / distance = (image_size/2) / focal_length
    # distance = (diagonal/2) * focal_length / (image_size/2)
    min_distance = (diagonal / 2) * focal_length / (image_size / 2)
    
    # Apply padding
    radius = min_distance * padding_factor
    
    return radius


def orbit_cameras(center, n_views=6, radius=3.0, elevation_min=15.0, elevation_max=15.0, up_axis=None, focal_length=550.0, resolution=512, device='cuda'):
    """
    Generate n_views cameras evenly spaced around `center` with elevations
    distributed between elevation_min and elevation_max, all looking toward `center`.

    Args:
        center: (3,) orbit center point
        n_views: number of camera views
        radius: distance from center
        elevation_min: minimum camera elevation angle in degrees
        elevation_max: maximum camera elevation angle in degrees
        up_axis: (3,) custom up axis for orbit (default: neg Y)
        focal_length: camera focal length in pixels
        resolution: image resolution (square)
        device: torch device

    Returns viewmats (N,4,4), Ks (N,3,3), W, H.
    """
    azimuths = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    
    # Distribute elevations evenly between min and max
    if n_views == 1:
        elevations_deg = np.array([(elevation_min + elevation_max) / 2])
    else:
        elevations_deg = np.linspace(elevation_min, elevation_max, n_views)

    W, H = resolution, resolution
    fx = fy = focal_length
    cx, cy = W / 2.0, H / 2.0

    # Default up axis (neg_y_up convention)
    if up_axis is None:
        up_axis = np.array([0., -1., 0.])
    up_axis = up_axis / np.linalg.norm(up_axis)
    
    # Build a local coordinate frame around the up axis
    # Find a perpendicular vector for the "right" direction
    if abs(np.dot(up_axis, np.array([1., 0., 0.]))) < 0.9:
        right = np.cross(up_axis, np.array([1., 0., 0.]))
    else:
        right = np.cross(up_axis, np.array([0., 0., 1.]))
    right = right / np.linalg.norm(right)
    forward = np.cross(right, up_axis)
    forward = forward / np.linalg.norm(forward)

    viewmats = []
    for i, az in enumerate(azimuths):
        el = np.radians(elevations_deg[i])
        
        # Camera position in local frame, then transform to world
        # Elevation lifts camera along up_axis
        # Start from forward direction (red X axis) at az=0
        # Camera is positioned along +forward, looking back at center
        local_offset = (
            np.cos(el) * np.sin(az) * right -
            np.cos(el) * np.cos(az) * forward +
            np.sin(el) * up_axis
        ) * radius
        
        cam_pos = center + local_offset

        # Forward: camera looks toward center
        z = center - cam_pos
        z /= np.linalg.norm(z)

        # Use the box's up axis as world_up
        world_up = up_axis

        # Avoid gimbal lock when z is nearly parallel to world_up
        if abs(np.dot(z, world_up)) > 0.99:
            # Fall back to a perpendicular axis
            world_up = forward

        x = np.cross(z, world_up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=0)  # rows = cam X, Y, Z axes
        t = -R @ cam_pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3,  3] = t
        viewmats.append(w2c)

    viewmats = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=device)
    Ks = K.unsqueeze(0).expand(n_views, -1, -1).contiguous()

    return viewmats, Ks, W, H


# ---------------------------------------------------------------------------
# Gsplat rendering
# ---------------------------------------------------------------------------

def render_views(centers, features_dc, opacity, scaling, rotation,
                 viewmats, Ks, W, H, background_color=(1.0, 1.0, 1.0)):
    """Render N views. Returns list of (H, W, 3) uint8 numpy arrays."""
    device = centers.device
    bg = torch.tensor(background_color, dtype=torch.float32, device=device)
    
    images = []
    with torch.no_grad():
        for i in range(viewmats.shape[0]):
            render_out, _, _ = rasterization(
                means=centers,
                quats=rotation,
                scales=scaling,
                opacities=opacity,
                colors=features_dc,
                viewmats=viewmats[i].unsqueeze(0),
                Ks=Ks[i].unsqueeze(0),
                width=W,
                height=H,
                sh_degree=0,
                backgrounds=bg,
            )
            img_np = (render_out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            images.append(img_np)
    return images


# ---------------------------------------------------------------------------
# 3-D → 2-D projection
# ---------------------------------------------------------------------------

def project_points_to_image(points_3d, viewmat, K, W, H):
    """
    Project (N,3) world-space points into pixel coords for one camera.
    Returns (coords (M,2) int [u,v], valid_mask (N,) bool).
    """
    if len(points_3d) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=bool)

    pts_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1).T
    vm  = viewmat.cpu().numpy()
    K_np = K.cpu().numpy()

    cam   = vm @ pts_h
    z     = cam[2]
    valid = z > 0.01

    cam_xy = cam[:2, valid] / z[valid]
    pix_h  = K_np @ np.vstack([cam_xy, np.ones((1, valid.sum()))])
    u = pix_h[0].astype(int)
    v = pix_h[1].astype(int)

    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    coords   = np.stack([u, v], axis=1)[in_frame]

    final_valid = np.zeros(len(points_3d), dtype=bool)
    final_valid[np.where(valid)[0][in_frame]] = True

    return coords, final_valid


def get_box_corners_world():
    """Get the 8 box corners in world space from the current transform."""
    # Unit cube corners (same as in create_unit_box, but just the 8 corners)
    unit_corners = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    
    transform = get_box_transform()
    
    # Transform corners to world space
    corners_h = np.hstack([unit_corners, np.ones((8, 1), dtype=np.float32)])  # (8, 4)
    world_corners = (transform @ corners_h.T).T[:, :3]  # (8, 3)
    
    return world_corners


def create_convex_hull_mask(corners_2d, W, H):
    """Create a binary mask from the convex hull of projected 2D points.
    
    Args:
        corners_2d: (N, 2) array of 2D points (u, v)
        W, H: image dimensions
        
    Returns:
        mask: (H, W) boolean array
    """
    import cv2
    from scipy.spatial import ConvexHull
    
    if len(corners_2d) < 3:
        # Not enough points for a hull, return empty mask
        return np.zeros((H, W), dtype=bool)
    
    try:
        hull = ConvexHull(corners_2d)
        hull_points = corners_2d[hull.vertices].astype(np.int32)
    except:
        # Convex hull failed (e.g., collinear points), return empty mask
        return np.zeros((H, W), dtype=bool)
    
    # Create mask using cv2.fillPoly
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [hull_points], 1)
    
    return mask.astype(bool)


def clip_mask_to_box(mask, box_corners_3d, viewmat, K, W, H):
    """Clip a mask to the convex hull of the projected 3D box.
    
    Args:
        mask: (H, W) boolean mask from SAM3
        box_corners_3d: (8, 3) world-space box corners
        viewmat: camera view matrix
        K: camera intrinsic matrix
        W, H: image dimensions
        
    Returns:
        clipped_mask: (H, W) boolean mask
    """
    # Project box corners to 2D
    corners_2d, valid = project_points_to_image(box_corners_3d, viewmat, K, W, H)
    
    if len(corners_2d) < 3:
        # Not enough visible corners, return original mask
        print("  [warn] <3 box corners visible, skipping hull clipping")
        return mask
    
    # Create convex hull mask
    hull_mask = create_convex_hull_mask(corners_2d, W, H)
    
    # Intersect with SAM3 mask
    clipped = mask & hull_mask
    
    # Report clipping stats
    original_pixels = mask.sum()
    clipped_pixels = clipped.sum()
    if original_pixels > 0:
        removed_pct = 100 * (1 - clipped_pixels / original_pixels)
        if removed_pct > 1:  # Only report if significant
            print(f"  [clip] removed {removed_pct:.1f}% of mask pixels outside box hull")
    
    return clipped


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------

def run_sam3(images, add_pts_3d, sub_pts_3d, label, viewmats, Ks, W, H,
             gaussian_centers, output_dir="sam3_output", filtered_output_dir="sam3_output_filtered",
             box_corners_3d=None):
    """
    For each rendered view:
      1. Text prompt (PCS)  → best mask
      2. Project 3D prompts → refine with point prompt (PVS)
      3. Clip mask to projected box convex hull
      4. Save render / mask / overlay PNG
      5. Collect masks for 3D bounding box computation
    
    Returns:
        masks_list: List of (H, W) boolean masks (or None for failed views)
    """
    import warnings
    import shutil
    
    # Clear output directory for a clean run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Clear filtered output directory for a clean run
    if os.path.exists(filtered_output_dir):
        shutil.rmtree(filtered_output_dir)
    os.makedirs(filtered_output_dir)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    masks_list = []  # Collect masks for bbox computation
    filtered_count = 0  # Track number of filtered views

    print("[SAM3] Loading model …")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model     = build_sam3_image_model(device="cuda")
        processor = Sam3Processor(model)

        for view_idx, img_np in enumerate(images):
            print(f"[SAM3] View {view_idx + 1}/{len(images)} …")
            pil_img = Image.fromarray(img_np)

            # Project 3D prompt points to this view first (needed for mask selection)
            vm = viewmats[view_idx]
            K  = Ks[view_idx]
            add_coords, _ = project_points_to_image(add_pts_3d, vm, K, W, H)
            sub_coords, _ = project_points_to_image(sub_pts_3d, vm, K, W, H)

            # --- Stage 1: text prompt --------------------------------------
            state = processor.set_image(pil_img)

            if label.strip():
                output = processor.set_text_prompt(state=state, prompt=label)
                masks  = output["masks"]    # (K, 1, H, W)
                scores = output["scores"]   # (K,)

                if scores.numel() == 0:
                    print(f"  [warn] no detections for '{label}' in view {view_idx} — saving render only")
                    masks_list.append(None)
                    # Save render anyway for debugging
                    pil_img.save(os.path.join(output_dir, f"view_{view_idx:02d}_render.png"))
                    # Save empty mask
                    Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(
                        os.path.join(output_dir, f"view_{view_idx:02d}_mask.png"))
                    # Save overlay with just the projected points (no mask)
                    overlay = pil_img.convert("RGB").copy()
                    draw = ImageDraw.Draw(overlay)
                    r = 6
                    for u, v in add_coords:
                        draw.ellipse([u-r, v-r, u+r, v+r], fill=(50, 230, 80), outline=(255,255,255), width=2)
                    for u, v in sub_coords:
                        draw.ellipse([u-r, v-r, u+r, v+r], fill=(230, 50, 50), outline=(255,255,255), width=2)
                    overlay.save(os.path.join(output_dir, f"view_{view_idx:02d}_overlay.png"))
                    # Also save the "all masks" debug image showing no detections
                    _save_all_masks_debug(pil_img, torch.zeros(0, 1, H, W), torch.zeros(0), add_coords, view_idx, output_dir)
                    print(f"  saved view_{view_idx:02d}_{{render,mask,overlay}}.png (no detections)")
                    continue

                n_detections = masks.shape[0]
                print(f"  [info] detected {n_detections} '{label}' instances")

                # Save debug image showing ALL detected masks
                _save_all_masks_debug(pil_img, masks, scores, add_coords, view_idx, output_dir)

                # Select mask containing the most projected add-points
                best = int(scores.argmax())  # default: highest score
                if len(add_coords) > 0:
                    masks_np = masks[:, 0].cpu().numpy()  # (K, H, W)
                    best_count = 0
                    best_score = -1
                    
                    for k in range(masks_np.shape[0]):
                        # Count how many add-points fall inside this mask
                        points_inside = 0
                        for (u, v) in add_coords:
                            if 0 <= v < H and 0 <= u < W and masks_np[k, v, u]:
                                points_inside += 1
                        
                        # Prefer mask with more points; break ties by score
                        if points_inside > best_count or (points_inside == best_count and scores[k].item() > best_score):
                            best_count = points_inside
                            best_score = scores[k].item()
                            best = k
                    
                    if best_count > 0:
                        print(f"  [info] selected mask {best} (contains {best_count}/{len(add_coords)} pts)")
                    else:
                        print(f"  [warn] no mask contains projected points — using highest score (mask {best})")

                mask = masks[best, 0].cpu().numpy().astype(bool)
            else:
                mask = np.ones((H, W), dtype=bool)

            # Clip mask to projected box convex hull
            if box_corners_3d is not None:
                mask = clip_mask_to_box(mask, box_corners_3d, vm, K, W, H)

            masks_list.append(mask)

            # --- Stage 2: save ---------------------------------------------
            _save_view_outputs(pil_img, mask, add_coords, sub_coords,
                               view_idx, output_dir, filtered_output_dir, filtered_count)
            filtered_count += 1

    print(f"\n[SAM3] Done. Results in '{output_dir}/'")
    print(f"[SAM3] Filtered views ({filtered_count} with detections) in '{filtered_output_dir}/'")
    
    return masks_list


def _save_all_masks_debug(pil_img, masks, scores, add_coords, view_idx, output_dir):
    """Save a debug image showing ALL detected masks with different colors."""
    W, H = pil_img.size
    overlay = pil_img.convert("RGBA").copy()
    
    # Handle case with no masks
    if masks.numel() == 0:
        overlay = overlay.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        
        # Draw projected points
        r = 8
        for u, v in add_coords:
            draw.ellipse([u-r, v-r, u+r, v+r], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        
        # Add "No detections" text
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((10, 10), "No detections", fill=(255, 0, 0), font=font)
        
        overlay.save(os.path.join(output_dir, f"view_{view_idx:02d}_all_masks.png"))
        return
    
    masks_np = masks[:, 0].cpu().numpy()  # (K, H, W)
    
    # Convert to numpy for blending
    img_np = np.array(overlay.convert("RGB")).astype(np.float32) / 255.0
    
    # Color values for each mask (normalized)
    color_values = [
        (1.0, 0.0, 0.0),    # red
        (0.0, 1.0, 0.0),    # green
        (0.0, 0.0, 1.0),    # blue
        (1.0, 1.0, 0.0),    # yellow
        (1.0, 0.0, 1.0),    # magenta
        (0.0, 1.0, 1.0),    # cyan
        (1.0, 0.5, 0.0),    # orange
        (0.5, 0.0, 1.0),    # purple
    ]
    
    alpha = 0.5
    for k in range(masks_np.shape[0]):
        color = color_values[k % len(color_values)]
        tint = np.zeros_like(img_np)
        tint[..., 0] = color[0]
        tint[..., 1] = color[1]
        tint[..., 2] = color[2]
        alpha_map = (masks_np[k] * alpha)[..., None].astype(np.float32)
        img_np = img_np * (1.0 - alpha_map) + tint * alpha_map
    
    overlay = Image.fromarray((img_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    
    # Draw projected points
    r = 8
    for u, v in add_coords:
        draw.ellipse([u-r, v-r, u+r, v+r], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    
    # Add legend with scores
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    y_offset = 10
    for k in range(masks_np.shape[0]):
        color_rgb = tuple(int(c * 255) for c in color_values[k % len(color_values)])
        score = scores[k].item()
        draw.rectangle([10, y_offset, 30, y_offset + 20], fill=color_rgb, outline=(255, 255, 255))
        draw.text((35, y_offset + 3), f"Mask {k}: {score:.3f}", fill=(255, 255, 255), font=font)
        y_offset += 25
    
    overlay.save(os.path.join(output_dir, f"view_{view_idx:02d}_all_masks.png"))


def _save_view_outputs(pil_img, mask, add_coords, sub_coords, view_idx, output_dir, filtered_output_dir=None, filtered_idx=None):
    W, H = pil_img.size

    # raw render
    pil_img.save(os.path.join(output_dir, f"view_{view_idx:02d}_render.png"))

    # binary mask
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"view_{view_idx:02d}_mask.png"))

    # colour overlay with proper alpha blending
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    tint = np.zeros_like(img_np)
    tint[..., 0] = 80 / 255.0   # R
    tint[..., 1] = 200 / 255.0  # G
    tint[..., 2] = 120 / 255.0  # B
    alpha = 0.5
    alpha_map = (mask * alpha)[..., None].astype(np.float32)
    blended = img_np * (1.0 - alpha_map) + tint * alpha_map
    overlay = Image.fromarray((blended * 255).astype(np.uint8))

    draw = ImageDraw.Draw(overlay)
    r = 6
    for u, v in add_coords:
        draw.ellipse([u-r, v-r, u+r, v+r], fill=(50, 230, 80),  outline=(255,255,255), width=2)
    for u, v in sub_coords:
        draw.ellipse([u-r, v-r, u+r, v+r], fill=(230, 50,  50), outline=(255,255,255), width=2)

    overlay.save(os.path.join(output_dir, f"view_{view_idx:02d}_overlay.png"))
    print(f"  saved view_{view_idx:02d}_{{render,mask,overlay}}.png")
    
    # Save to filtered output directory (render and overlay only, no debug points)
    if filtered_output_dir is not None and filtered_idx is not None:
        # Save clean render (no points)
        pil_img.save(os.path.join(filtered_output_dir, f"view_{filtered_idx:02d}_render.png"))
        
        # Save clean overlay (no points)
        clean_overlay = Image.fromarray((blended * 255).astype(np.uint8))
        clean_overlay.save(os.path.join(filtered_output_dir, f"view_{filtered_idx:02d}_overlay.png"))
        
        # Also save the mask for downstream tasks
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(filtered_output_dir, f"view_{filtered_idx:02d}_mask.png"))
        
        print(f"  saved filtered view_{filtered_idx:02d}_{{render,overlay,mask}}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian_ply',  type=str,   required=True)
    parser.add_argument('--n_views',       type=int,   default=6)
    parser.add_argument('--resolution',    type=int,   default=512, help='render resolution (default: 512)')
    parser.add_argument('--orbit_radius',  type=float, default=None, help='orbit radius around the target (default: auto from box size)')
    parser.add_argument('--elevation_min', type=float, default=15.0, help='minimum camera elevation in degrees (default: 15)')
    parser.add_argument('--elevation_max', type=float, default=15.0, help='maximum camera elevation in degrees (default: 15)')
    parser.add_argument('--focal_length',  type=float, default=550.0, help='camera focal length in pixels (default: 550)')
    parser.add_argument('--padding',       type=float, default=1.0, help='padding factor for auto radius (default: 1.0)')
    parser.add_argument('--output_dir',    type=str,   default='sam3_output')
    parser.add_argument('--filtered_output_dir', type=str, default='sam3_output_filtered', help='directory for filtered outputs (views with detections only)')
    
    # Box preset arguments for automated testing
    parser.add_argument('--box_center',    type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='Initial box center position')
    parser.add_argument('--box_size',      type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='Initial box size (scale)')
    parser.add_argument('--box_angles',    type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='Initial box rotation angles in degrees (Euler XYZ)')
    parser.add_argument('--label',         type=str,   default=None, help='Object label for segmentation')
    
    args = parser.parse_args()

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("neg_y_up")
    ps.set_navigation_style("first_person")

    centers, features_dc, opacity, scaling, rotation = load_gaussians_from_ply(args.gaussian_ply)
    print(f"Loaded {centers.shape[0]} gaussian particles")

    ps.register_gaussian_particles("gaussians",
                                   subsample_factor=1,
                                   means=centers.unsqueeze(0),
                                   colors=features_dc.unsqueeze(0),
                                   opacities=opacity.unsqueeze(0),
                                   scales=scaling.unsqueeze(0),
                                   quats=rotation.unsqueeze(0),
                                   sh_degree=0,
                                   )

    # Build initial transform from command line arguments
    initial_transform = np.eye(4, dtype=np.float32)
    
    if args.box_center is not None:
        initial_transform[:3, 3] = np.array(args.box_center, dtype=np.float32)
    
    if args.box_angles is not None:
        # Build rotation matrix from Euler angles (XYZ extrinsic = Rz @ Ry @ Rx)
        angles_rad = np.radians(args.box_angles)
        ax, ay, az = angles_rad
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        
        R = Rz @ Ry @ Rx
        initial_transform[:3, :3] = R
    
    if args.box_size is not None:
        # Apply scale to rotation matrix columns
        scale = np.array(args.box_size, dtype=np.float32)
        initial_transform[:3, 0] *= scale[0]
        initial_transform[:3, 1] *= scale[1]
        initial_transform[:3, 2] *= scale[2]
    
    if args.label is not None:
        sam_box["label"][0] = args.label
    
    # Initialize parameters from args
    sam_box["focal_length"][0] = args.focal_length
    sam_box["padding"][0] = args.padding
    sam_box["elevation_min"][0] = args.elevation_min
    sam_box["elevation_max"][0] = args.elevation_max
    
    # Register the prompt box with initial transform
    sam_box["transform"][0] = initial_transform
    prompt_box = register_prompt_box()

    ps.set_user_callback(make_ui_callback(prompt_box, args))
    ps.show()

    # --- After window closes -----------------------------------------------
    box_center = get_box_center()
    box_up_axis = get_box_up_axis()
    center, size, angles_deg = get_box_properties()
    label = get_sam_label()
    focal_length = sam_box["focal_length"][0]
    padding = sam_box["padding"][0]
    elevation_min = sam_box["elevation_min"][0]
    elevation_max = sam_box["elevation_max"][0]
    
    # Debug: print the full transform
    transform = get_box_transform()
    print(f"\n[Debug] Box transform matrix:")
    print(f"  [{transform[0,0]:.4f}, {transform[0,1]:.4f}, {transform[0,2]:.4f}, {transform[0,3]:.4f}]")
    print(f"  [{transform[1,0]:.4f}, {transform[1,1]:.4f}, {transform[1,2]:.4f}, {transform[1,3]:.4f}]")
    print(f"  [{transform[2,0]:.4f}, {transform[2,1]:.4f}, {transform[2,2]:.4f}, {transform[2,3]:.4f}]")
    print(f"  [{transform[3,0]:.4f}, {transform[3,1]:.4f}, {transform[3,2]:.4f}, {transform[3,3]:.4f}]")
    
    print(f"[Orbit] Box -Y up axis: [{box_up_axis[0]:.3f}, {box_up_axis[1]:.3f}, {box_up_axis[2]:.3f}]")

    # Print box properties for reuse
    print(f"\n[Box Properties]")
    print(f"  --box_center {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}")
    print(f"  --box_size {size[0]:.4f} {size[1]:.4f} {size[2]:.4f}")
    print(f"  --box_angles {angles_deg[0]:.2f} {angles_deg[1]:.2f} {angles_deg[2]:.2f}")
    print(f"  --focal_length {focal_length:.1f}")
    print(f"  --padding {padding:.2f}")
    print(f"  --elevation_min {elevation_min:.1f}")
    print(f"  --elevation_max {elevation_max:.1f}")
    if label.strip():
        print(f"  --label \"{label}\"")

    # Create add_pts array from box center (single point)
    add_pts = np.array([box_center], dtype=np.float32)
    sub_pts = np.zeros((0, 3), dtype=np.float32)

    print(f"\n[SAM3] Prompts ready:")
    print(f"  label     = '{label}'")
    print(f"  box center = {box_center}")

    if not label.strip():
        print("[SAM3] No label set — skipping segmentation.")
        return

    # Orbit around box center
    orbit_center = box_center
    
    # Auto-compute orbit radius from box size if not specified
    if args.orbit_radius is not None:
        orbit_radius = args.orbit_radius
    else:
        orbit_radius = compute_orbit_radius_from_box(size, focal_length=focal_length, image_size=args.resolution, padding_factor=padding)
        print(f"[Render] Auto orbit radius: {orbit_radius:.2f} (from box size {size}, focal_length {focal_length:.1f}, padding {padding:.2f}x)")

    print(f"\n[Render] {args.n_views} orbit views around {orbit_center} (radius={orbit_radius:.2f}, elevation={elevation_min:.1f}°-{elevation_max:.1f}°, focal_length={focal_length:.1f}, resolution={args.resolution}) …")
    viewmats, Ks, W, H = orbit_cameras(
        center=orbit_center,
        n_views=args.n_views,
        radius=orbit_radius,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        up_axis=box_up_axis,
        focal_length=focal_length,
        resolution=args.resolution,
    )

    images = render_views(centers, features_dc, opacity, scaling, rotation,
                          viewmats, Ks, W, H)
    print(f"[Render] {len(images)} views at {W}x{H}")

    # Get box corners in world space for convex hull clipping
    box_corners_3d = get_box_corners_world()

    masks_list = run_sam3(images, add_pts, sub_pts, label, viewmats, Ks, W, H,
                          gaussian_centers=centers, output_dir=args.output_dir,
                          filtered_output_dir=args.filtered_output_dir,
                          box_corners_3d=box_corners_3d)

    print("\n[SAM3] Segmentation complete.")
    print(f"  All results: '{args.output_dir}/'")
    print(f"  Filtered (detections only): '{args.filtered_output_dir}/'")


if __name__ == '__main__':
    main()
    print_gpu_memory()

# Example (auto radius from box):
# python3 polyscope_gsplat2.py --gaussian_ply splats/m60.ply --n_views 6 --output_dir sam3_output
#
# Example (manual radius):
# python3 polyscope_gsplat2.py --gaussian_ply splats/m60.ply --n_views 6 --orbit_radius 4 --output_dir sam3_output
#
# With preset box (for automated testing):
# python3 polyscope_gsplat2.py --gaussian_ply splats/m60.ply --n_views 6 \
#     --box_center -0.5 0.2 0.0 --box_size 2.0 1.5 3.0 --box_angles 0 0 0 --label "Tank"