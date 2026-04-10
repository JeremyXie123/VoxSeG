import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from core.camera import CameraState, construct_rays
from stages.segmentation import SegmentationResult
from core.splat_io import print_gpu_memory

class PhiGrid(nn.Module, ABC):
    """Abstract base class where the grid owns its optimization logic."""
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.optimizer = None

    @abstractmethod
    def summarize(self):
        """Prints summary statistics about the current grid state for debugging."""
        pass

    @abstractmethod
    def step(self, batch_views: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        """Performs one optimization step, including potential subdivision or refinement."""
        pass

    @abstractmethod
    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        pass

    def render_mask(self, view_idx: int, cams: CameraState, num_samples: int) -> torch.Tensor:
        """Shared volumetric rendering logic used by all child classes."""
        ray_origins, ray_dirs = construct_rays(
            cams.viewmats[view_idx], cams.Ks[view_idx], 
            self.args.height, self.args.width, self.device
        )
        t_vals = torch.linspace(0.1, 10.0, num_samples, device=self.device) 
        points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None] 
        
        phi_vals = self.query(points, cams)
        alpha = torch.sigmoid(-self.args.sharpness * phi_vals)
        return 1.0 - torch.prod(1.0 - alpha, dim=-1)

class DenseGrid(PhiGrid):
    """Standard fixed-resolution grid."""
    def __init__(self, args, device):
        super().__init__(args, device)
        res = args.grid_resolution
        self.phi = nn.Parameter(torch.randn((res, res, res), device=device) * 0.1 + 0.5)
        self.optimizer = torch.optim.Adam([self.phi], lr=args.lr)

    def summarize(self):
        phi_min = self.phi.min().item()
        phi_max = self.phi.max().item()
        phi_mean = self.phi.mean().item()
        phi_median = torch.median(self.phi).item()
        phi_midrange = (phi_min + phi_max) / 2.0
        print(f"Phi min/max/avg/med/mid: [{phi_min:.4f}, {phi_max:.4f}, {phi_mean:.4f}, {phi_median:.4f}, {phi_midrange:.4f}]")

    def query(self, points, cams):
        points_norm = (points - cams.target_center) / cams.grid_radius
        grid = self.phi[None, None, ...] 
        N_rays, N_samples = points.shape[0], points.shape[1]
        sampling_coords = points_norm.reshape(1, N_rays * N_samples, 1, 1, 3)
        vals = F.grid_sample(grid, sampling_coords, mode='bilinear', padding_mode='border', align_corners=True)
        return vals.reshape(N_rays, N_samples)

    def step(self, batch_indices, seg_result, cams):
        self.optimizer.zero_grad()
        total_mask_loss = 0.0
        
        for idx in batch_indices:
            pred_mask = self.render_mask(idx, cams, self.args.num_samples)
            target = torch.from_numpy(seg_result.masks[idx]).float().to(self.device).view(-1)
            
            # Loss calculation
            loss = F.binary_cross_entropy(pred_mask, target) / len(batch_indices)
            loss.backward()
            total_mask_loss += loss.item()

        # Regularization
        dx = self.phi[1:, :, :] - self.phi[:-1, :, :]
        dy = self.phi[:, 1:, :] - self.phi[:, :-1, :]
        dz = self.phi[:, :, 1:] - self.phi[:, :, :-1]
        smooth_loss = self.args.beta * ((dx**2).mean() + (dy**2).mean() + (dz**2).mean())
        smooth_loss.backward()

        self.optimizer.step()
        return {"mask_loss": total_mask_loss, "smooth_loss": smooth_loss}

class SparseAdaptiveGrid(PhiGrid):
    """Adaptive grid that handles subdivision logic."""
    def __init__(self, args, device):
        super().__init__(args, device)
        # Initialize fvdb structure here...
        self.current_iteration = 0

    def query(self, points, cams):
        # fvdb-specific lookup
        return torch.zeros((points.shape[0], points.shape[1]), device=self.device)

    def step(self, batch_indices, seg_result, cams):
        self.current_iteration += 1
        
        # 1. Standard Gradient Step (similar to DenseGrid)
        # ... logic to compute loss and update active voxels ...

        # 2. Subdivision Logic
        if self.current_iteration % 50 == 0:
            print("Checking for subdivision triggers...")
            # self.subdivide_high_gradient_regions()
            # self.optimizer = update_optimizer_for_new_params()

        return {"mask_loss": 0.0, "smooth_loss": 0.0}

# --- UNIFIED PIPELINE ENTRY ---

def optimize_voxel_grid(grid: PhiGrid, seg_result: SegmentationResult, cams: CameraState, args, device: torch.device):    
    history = {'total_loss': [], 'mask_loss': [], 'smooth_loss': [], 'time': []}
    start_time = time.time()

    for iter in range(args.num_iters):
        batch_indices = torch.randperm(args.num_views)[:args.batch_size].tolist()
        
        # The grid handles EVERYTHING internally
        metrics = grid.step(batch_indices, seg_result, cams)

        # Logging logic
        total_loss = metrics['mask_loss'] + metrics['smooth_loss']
        time_elapsed = time.time() - start_time

        history['total_loss'].append(total_loss)
        history['mask_loss'].append(metrics['mask_loss'])
        history['smooth_loss'].append(metrics['smooth_loss'])
        history['time'].append(time_elapsed)

        print(f"Iter {iter+1}/{args.num_iters}: Time={time_elapsed:.2f}s, Loss={total_loss:.4f}, Mask={metrics['mask_loss']:.6f}, Smooth={metrics['smooth_loss']:.6f}")        
        grid.summarize()
        print_gpu_memory()
    return history