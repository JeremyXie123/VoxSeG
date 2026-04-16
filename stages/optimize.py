import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import polyscope as ps
from core.camera import CameraState, construct_rays
from stages.segmentation import SegmentationResult
from core.splat_io import print_gpu_memory


@dataclass
class PolyscopeData:
    """Grid data in a form ready for Polyscope registration."""
    all_points: np.ndarray       # [N, 3] world-space voxel centres
    all_phi: np.ndarray          # [N]    phi value at each voxel
    all_voxel_sizes: np.ndarray  # [N, 3] world-space size of each voxel
    surface_points: np.ndarray   # [M, 3] subset near the iso-surface
    surface_phi: np.ndarray      # [M]    phi values for surface voxels


class PhiGrid(nn.Module, ABC):
    """Abstract base class where the grid owns its optimization logic."""
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.optimizer = None

    @abstractmethod
    def summarize(self):
        pass

    @abstractmethod
    def step(self, batch_views: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        pass

    @abstractmethod
    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        pass

    @abstractmethod
    def get_polyscope_data(self) -> PolyscopeData:
        pass

    @abstractmethod
    def visualize(self, cams: CameraState):
        """Register the phi grid in Polyscope. Called by visualize_with_polyscope."""
        pass

    def _compute_mask_loss(self, pred_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.args.metric == "bce":
            return F.binary_cross_entropy(pred_mask, target)
        elif self.args.metric == "mse":
            return F.mse_loss(pred_mask, target)
        elif self.args.metric == "kl":
            eps = 1e-7
            p = torch.stack([pred_mask, 1 - pred_mask], dim=-1).clamp(eps, 1 - eps)
            q = torch.stack([target,    1 - target   ], dim=-1).clamp(eps, 1 - eps)
            return F.kl_div(p.log(), q, reduction='batchmean')
        elif self.args.metric == "mi":
            return self._mutual_information_loss(pred_mask, target)
    
    def _mutual_information_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                  num_bins: int = 32) -> torch.Tensor:
        """
        Compute negative mutual information as a loss (minimize to maximize MI).
        
        Uses soft histogram binning for differentiability.
        MI(P, T) = H(P) + H(T) - H(P, T)
        
        Args:
            pred: Predicted mask values in [0, 1], shape (N,)
            target: Target mask values in [0, 1], shape (N,)
            num_bins: Number of histogram bins
        
        Returns:
            Negative mutual information (lower = better alignment)
        """
        eps = 1e-7
        n = pred.shape[0]
        
        # Bin centers from 0 to 1
        bin_centers = torch.linspace(0, 1, num_bins, device=pred.device)
        
        # Soft binning using Gaussian kernel
        # sigma controls bin width - smaller = sharper bins
        sigma = 1.0 / num_bins
        
        # Compute soft bin assignments: (N, num_bins)
        pred_bins = torch.exp(-0.5 * ((pred.unsqueeze(1) - bin_centers) / sigma) ** 2)
        target_bins = torch.exp(-0.5 * ((target.unsqueeze(1) - bin_centers) / sigma) ** 2)
        
        # Normalize to get soft histograms
        pred_bins = pred_bins / (pred_bins.sum(dim=1, keepdim=True) + eps)
        target_bins = target_bins / (target_bins.sum(dim=1, keepdim=True) + eps)
        
        # Marginal distributions: P(pred_bin), P(target_bin)
        p_pred = pred_bins.mean(dim=0) + eps      # (num_bins,)
        p_target = target_bins.mean(dim=0) + eps  # (num_bins,)
        
        # Joint distribution: P(pred_bin, target_bin)
        # Outer product averaged over samples
        p_joint = torch.einsum('ni,nj->ij', pred_bins, target_bins) / n + eps  # (num_bins, num_bins)
        
        # Entropies
        h_pred = -torch.sum(p_pred * torch.log(p_pred))
        h_target = -torch.sum(p_target * torch.log(p_target))
        h_joint = -torch.sum(p_joint * torch.log(p_joint))
        
        # Mutual information: I(P; T) = H(P) + H(T) - H(P, T)
        mi = h_pred + h_target - h_joint
        
        # Return negative MI (we want to maximize MI, so minimize -MI)
        # Normalize by max possible MI for stability
        max_mi = torch.log(torch.tensor(num_bins, device=pred.device, dtype=pred.dtype))
        return -mi / max_mi

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


class BasicGrid(PhiGrid):
    """Standard fixed-resolution grid."""
    def __init__(self, args, device):
        super().__init__(args, device)
        res = args.grid_resolution
        self.phi = nn.Parameter(torch.randn((res, res, res), device=device) * 0.1 + 0.5)
        self.optimizer = torch.optim.Adam([self.phi], lr=args.lr)

    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        center = cams.target_center.to(points.device)
        R = cams.grid_rotation.to(points.device)  # (3, 3) maps local->world
        
        # Transform world points to box-local coordinates:
        # R's columns are the box's local axes in world coords
        # To go world->local: local = R^T @ (world - center)
        # For row vectors: local_row = (world_row - center) @ R
        points_centered = points - center
        points_local = points_centered @ R
        points_norm = points_local / cams.grid_radius
        
        grid = self.phi[None, None, ...]  # (1, 1, res, res, res)
        N_rays, N_samples, _ = points_norm.shape
        sampling_coords = points_norm.reshape(1, N_rays * N_samples, 1, 1, 3)
        vals = F.grid_sample(grid, sampling_coords, mode='bilinear', padding_mode='border', align_corners=True)
        return vals.reshape(N_rays, N_samples)

    def step(self, batch_indices: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        total_loss = 0.0
        total_mask_loss = 0.0

        for view_idx in batch_indices:
            pred_mask = self.render_mask(view_idx, cams, self.args.num_samples)
            target = torch.from_numpy(seg_result.masks[view_idx]).float().to(self.device).view(-1)
            mask_loss = self._compute_mask_loss(pred_mask, target)
            view_loss = mask_loss / self.args.num_views
            total_loss += view_loss
            total_mask_loss += view_loss.item()

        dx = self.phi[1:, :, :] - self.phi[:-1, :, :]
        dy = self.phi[:, 1:, :] - self.phi[:, :-1, :]
        dz = self.phi[:, :, 1:] - self.phi[:, :, :-1]
        smooth_term = self.args.beta * ((dx**2).mean() + (dy**2).mean() + (dz**2).mean())
        total_loss += smooth_term

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"mask_loss": total_mask_loss, "smooth_loss": smooth_term.item()}

    def summarize(self):
        phi_min = self.phi.min().item()
        phi_max = self.phi.max().item()
        phi_mean = self.phi.mean().item()
        phi_median = torch.median(self.phi).item()
        phi_midrange = (phi_min + phi_max) / 2.0
        print(f"Phi min/max/avg/med/mid: [{phi_min:.4f}, {phi_max:.4f}, {phi_mean:.4f}, {phi_median:.4f}, {phi_midrange:.4f}]")

    def get_polyscope_data(self) -> PolyscopeData:
        phi_np = self.phi.detach().cpu().numpy()
        res = phi_np.shape[0]
        idx = np.argwhere(np.ones_like(phi_np, dtype=bool))
        points_local = (idx / (res - 1)) * 2 - 1
        all_phi = phi_np.ravel()
        surface_mask = np.abs(all_phi - self.args.iso_level) < 0.5
        cell_size = 2.0 / (res - 1)  # local-space voxel size
        all_voxel_sizes = np.full_like(points_local, cell_size)
        return PolyscopeData(
            all_points=points_local.astype(np.float32),
            all_phi=all_phi.astype(np.float32),
            all_voxel_sizes=all_voxel_sizes.astype(np.float32),
            surface_points=points_local[surface_mask].astype(np.float32),
            surface_phi=all_phi[surface_mask].astype(np.float32),
        )

    def visualize(self, cams: CameraState):
        """Register the dense phi grid in Polyscope as a volume grid with isosurface."""
        center = cams.target_center.detach().cpu().numpy()
        R = cams.grid_rotation.detach().cpu().numpy()  # (3, 3) maps local->world
        radius = cams.grid_radius
        
        phi_data = self.phi.detach().cpu().numpy().transpose(2, 1, 0)

        # Register voxel nodes near the isosurface as a point cloud
        mask = phi_data < self.args.iso_level
        idx = np.argwhere(mask)
        res = phi_data.shape[0]
        
        # Local coordinates in [-1, 1]
        points_local = (idx / (res - 1)) * 2 - 1
        # Scale by radius then rotate local->world: world = local @ R^T + center
        # Since R maps local->world as R @ local_col, for row vectors: local_row @ R^T
        points_scaled = points_local * radius
        points_world = center + points_scaled @ R.T

        ps_pts = ps.register_point_cloud("Phi Voxel Nodes", points_world, radius=0.0025, color=(1.0, 0.9, 0.1))
        ps_pts.add_scalar_quantity("phi_val", phi_data[mask], cmap='coolwarm')


# --------------------------------------------------------------------------- #
# UNIFIED PIPELINE ENTRY
# --------------------------------------------------------------------------- #

def optimize_voxel_grid(grid: PhiGrid, seg_result: SegmentationResult, cams: CameraState, args, device: torch.device):
    history = {'total_loss': [], 'mask_loss': [], 'smooth_loss': [], 'time': []}
    start_time = time.time()

    for iter in range(args.num_iters):
        batch_indices = torch.randperm(args.num_views)[:args.batch_size].tolist()
        metrics = grid.step(batch_indices, seg_result, cams)

        total_loss   = metrics['mask_loss'] + metrics['smooth_loss']
        time_elapsed = time.time() - start_time

        history['total_loss'].append(total_loss)
        history['mask_loss'].append(metrics['mask_loss'])
        history['smooth_loss'].append(metrics['smooth_loss'])
        history['time'].append(time_elapsed)

        print(f"Iter {iter+1}/{args.num_iters}: Time={time_elapsed:.2f}s, Loss={total_loss:.4f}, "
              f"Mask={metrics['mask_loss']:.6f}, Smooth={metrics['smooth_loss']:.6f}")
        grid.summarize()
        print_gpu_memory()

    return history