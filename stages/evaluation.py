import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def visualize_batch_grid(images_input, num_cols=5, axes_pad=0.1, filename="batch_grid.png", show=True):
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
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metrics(history: dict, filename="training_metrics.png", show=True):
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
    if show:
        plt.show()
    else:
        plt.close(fig)