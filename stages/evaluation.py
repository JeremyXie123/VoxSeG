import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import polyscope as ps

from core.camera import CameraState

def visualize_batch_grid(images_input, num_cols=5, axes_pad=0.1):
    """Renders a batch of images in a Matplotlib ImageGrid."""
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

def plot_training_metrics(history: dict, filename="training_metrics.png"):
    """Graphs the Mask Loss and Smoothness Loss over the iterations."""
    iters = range(len(history['total_loss']))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mask BCE Loss', color=color)
    ax1.plot(iters, history['mask_loss'], color=color, label='Mask Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Smoothness Loss', color=color)
    ax2.plot(iters, history['smooth_loss'], color=color, label='Smoothness', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Voxel Optimization Metrics')
    fig.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(filename)
    print(f"Metrics graph saved to {filename}")
    plt.show()

def visualize_with_polyscope(masked_rgbs: list[np.ndarray], cams: CameraState, phi: torch.Tensor, args):
    """Visualizes the optimized phi grid and camera frustums using masked RGB views."""
    ps.set_window_size(1920, 1080*0.75)

    ps.init()
    ps.set_up_dir("neg_y_up")

    # Phi Grid Registration
    bound_low = (cams.target_center - cams.grid_radius).detach().cpu().numpy()
    bound_high = (cams.target_center + cams.grid_radius).detach().cpu().numpy()
    phi_data = phi.detach().cpu().numpy().transpose(2, 1, 0)
    
    mask = phi_data < args.iso_level
    idx = np.argwhere(mask)
    res = phi_data.shape[0] 
    points_local = (idx / (res - 1)) * 2 - 1 
    points_world = cams.target_center.detach().cpu().numpy() + points_local * cams.grid_radius 

    ps_pts = ps.register_point_cloud("Phi Voxel Nodes", points_world, radius=0.0025, color=(1.0, 0.9, 0.1))
    ps_pts.add_scalar_quantity("phi_val", phi_data[mask], cmap='coolwarm')
    
    ps_grid = ps.register_volume_grid("Phi Grid", phi_data.shape, bound_low, bound_high)
    ps_grid.add_scalar_quantity(
        "phi", phi_data, defined_on='nodes', cmap='coolwarm', enabled=True,
        enable_isosurface_viz=True, isosurface_level=args.iso_level, 
        isosurface_color=(0.2, 0.5, 0.8), enable_gridcube_viz=False
    )

    # Camera Registration
    centers, rights, ups, forwards = [], [], [], []
    for i in range(len(cams.viewmats)):
        c2w = torch.linalg.inv(cams.viewmats[i]).detach().cpu().numpy()
        root = c2w[:3, 3] 
        look_dir = c2w[:3, 2]   
        up_dir = -c2w[:3, 1]    
        right_dir = c2w[:3, 0]  

        centers.append(root)
        rights.append(right_dir)
        ups.append(up_dir)
        forwards.append(look_dir)
        
        focal_px = cams.Ks[i, 0, 0].item()
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