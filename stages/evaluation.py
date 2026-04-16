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

    # Convert tensor values to numpy, moving to CPU if needed
    mask_loss_vals = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in history['mask_loss']]
    smooth_loss_vals = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in history['smooth_loss']]

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mask BCE Loss', color=color)
    ax1.plot(iters, mask_loss_vals, color=color, label='Mask Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Smoothness Loss', color=color)
    ax2.plot(iters, smooth_loss_vals, color=color, label='Smoothness', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Voxel Optimization Metrics')
    fig.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(filename)
    print(f"Metrics graph saved to {filename}")
    plt.show()

def visualize_with_polyscope(masked_rgbs: list[np.ndarray], cams: CameraState, phi_grid, args):
    """Visualizes the optimized phi grid and camera frustums using masked RGB views.
    
    Args:
        masked_rgbs: List of masked RGB images from camera views
        cams: Camera state object with view and grid information
        phi_grid: PhiGrid object (DenseGrid, SparseAdaptiveGrid, etc.) with visualize() method
        args: Configuration arguments
    """
    ps.set_window_size(1920, 1080)

    ps.init()
    ps.set_up_dir("neg_y_up")

    # Delegate grid visualization to the grid's own method
    phi_grid.visualize(cams)

    # Camera Registration
    for i in range(len(cams.viewmats)):
        c2w = torch.linalg.inv(cams.viewmats[i]).detach().cpu().numpy()
        root = c2w[:3, 3] 
        look_dir = c2w[:3, 2]   
        up_dir = -c2w[:3, 1]
        
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
    
    ps.show()