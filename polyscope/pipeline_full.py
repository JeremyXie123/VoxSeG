# Purpose of this file:
# Load a gaussian splat
# Render a batch of images from multiple arbitrary view points
# Run Segment Anything Model to segment out the object in each image
# (TODO?) Use segmented images to produce a new gaussian splat

# Questions
# - PLY file standards and SAM prompting (is object always at 0,0,0, etc.), x,y,z conventions?
#   - Maybe use a bounding box?
# - How to go from segmented images to new splat (and how do the metrics play in)
#   - Differentiable rendering?

# Example usage:
# python pipeline_full.py --distance 5 --num_views 7
# May need to update the SAM checkpoint and config paths (absolute paths)

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

def get_batch_Ks(focal, width, height, num_views):
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

def run_sam_on_batch(render_colors, checkpoint_path, model_cfg, device):
    """Runs the Segment Anything Model on a batch of rendered images and returns the predicted masks"""
    # Initialize SAM 2.1
    model = build_sam2(os.path.abspath(model_cfg), os.path.abspath(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(model)
    
    input_point = np.array([[512, 512], [600, 600], [500, 500], [600, 500], [500, 600]])
    input_label = np.array([1, 1, 1, 1, 1])

    batched_masks = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i in range(render_colors.shape[0]):
            print(f"Running SAM 2.1 on image {i+1}/{render_colors.shape[0]}...")
            
            # Convert render to format SAM 2.1 expects
            img_np = (render_colors[i].detach().clamp(0, 1) * 255).byte().cpu().numpy()
            predictor.set_image(img_np)
            
            # Run inference using only the fixed points
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            batched_masks.append(masks[0])
            
    return np.stack(batched_masks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="splats/truck.ply", help="Path to input PLY file")
    parser.add_argument("--output_render", type=str, default="rendered", help="Directory to save rendered images")
    parser.add_argument("--sam_checkpoint", type=str, default="C:\\Users\\Jeremy\\Desktop\\CSC494\\CSC494\\polyscope\\sam2.1_hiera_l.pt", help="Path to SAM checkpoint")
    parser.add_argument("--sam_config", type=str, default="C:\\Users\\Jeremy\\Desktop\\CSC494\\CSC494\\polyscope\\sam2.1_hiera_l.yaml", help="Path to SAM config")
    parser.add_argument("--orbit", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="X Y Z coordinates of orbit center")
    parser.add_argument("--distance", type=float, default=2, help="Distance of camera from orbit center")
    parser.add_argument("--width", type=int, default=1024, help="Width of rendered images")
    parser.add_argument("--height", type=int, default=1024, help="Height of rendered images")
    parser.add_argument("--focal", type=float, default=1100.0, help="Focal length for rendering")
    parser.add_argument("--num_views", type=int, default=10, help="Number of views to render")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_render, exist_ok=True)
    means, scales, quats, colors, opacities = load_ply(args.input, device)

    # Compute camera intrinsics
    Ks = get_batch_Ks(args.focal, args.width, args.height, num_views=args.num_views)

    # Compute camera extrinsics
    viewmats = get_batch_viewmats(means, torch.tensor(args.orbit, device=means.device).float(), torch.tensor(args.distance, device=means.device).float(), num_views=args.num_views)

    # Perform rasterization on the gaussian splat
    print(f"Rendering {len(viewmats)} views...")
    render_colors, alphas, meta = rasterization(
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

    # Export rendered images before SAM
    # for i in range(render_colors.shape[0]):
    #     img = render_colors[i].detach().float().clamp(0, 1).cpu().numpy()
    #     plt.imsave(os.path.join(args.output_render, f"view_{i:03d}.jpg"), img)

    # visualize_batch_grid(render_colors)

    # Run SAM on the batch of rendered images
    print(f"Running SAM on {len(viewmats)} rendered views...")
    masks = run_sam_on_batch(render_colors, args.sam_checkpoint, args.sam_config, device)

    # Overlay masks onto rendered images
    masked_images = render_colors.cpu().numpy() * masks[:, :, :, None]
    # visualize_batch_grid(torch.from_numpy(masked_images))

    original = render_colors.detach().float().clamp(0, 1).cpu().numpy()
    masks_rgb = np.repeat(masks[:, :, :, None], 3, axis=-1).astype(float)
    segmented = original * masks_rgb
    stacked_images = np.concatenate([original, masks_rgb, segmented], axis=0)
    visualize_batch_grid(stacked_images, num_cols=args.num_views, axes_pad=0.05)