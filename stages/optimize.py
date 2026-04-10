import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import fvdb
from core.camera import CameraState, construct_rays, project_points
from stages.segmentation import SegmentationResult
from core.splat_io import print_gpu_memory


@dataclass
class PolyscopeData:
    """Grid data in a form ready for Polyscope registration."""
    all_points: np.ndarray       # [N, 3] world-space voxel centres
    all_phi: np.ndarray          # [N]    phi value at each voxel
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
    """Standard fixed-resolution grid. Logic is identical to the original optimize.py."""
    def __init__(self, args, device):
        super().__init__(args, device)
        res = args.grid_resolution
        self.phi = nn.Parameter(torch.randn((res, res, res), device=device) * 0.1 + 0.5)
        self.optimizer = torch.optim.Adam([self.phi], lr=args.lr)

    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        center = cams.target_center.to(points.device)
        points_norm = (points - center) / cams.grid_radius
        grid = self.phi[None, None, ...]
        N_rays, N_samples, _ = points_norm.shape
        sampling_coords = points_norm.reshape(1, N_rays * N_samples, 1, 1, 3)
        vals = F.grid_sample(grid, sampling_coords, mode='bilinear', padding_mode='border', align_corners=True)
        return vals.reshape(N_rays, N_samples)

    def step(self, batch_indices: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        total_loss = 0.0
        total_mask_loss = 0.0

        for view_idx in batch_indices:
            ray_origins, ray_dirs = construct_rays(cams.viewmats[view_idx], cams.Ks[view_idx], self.args.height, self.args.width, self.device)
            t_vals = torch.linspace(0.1, 10.0, self.args.num_samples, device=self.device)
            points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None]
            phi_vals = self.query(points, cams)
            alpha = torch.sigmoid(-self.args.sharpness * phi_vals)
            pred_mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)

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
        return PolyscopeData(
            all_points=points_local.astype(np.float32),
            all_phi=all_phi.astype(np.float32),
            surface_points=points_local[surface_mask].astype(np.float32),
            surface_phi=all_phi[surface_mask].astype(np.float32),
        )


class SparseAdaptiveGrid(PhiGrid):
    """
    Adaptive sparse grid backed by fVDB.

    phi is a flat [num_voxels, 1] nn.Parameter. The fVDB Grid holds the
    topology. Every refine_every iterations, voxels with high mean
    reprojection error across views are subdivided by 2.
    """
    def __init__(self, args, device, cams: CameraState):
        super().__init__(args, device)
        self.iso_level = args.iso_level
        self.band_width = 0.4   # used only for get_polyscope_data surface visualisation
        self.refine_every = max(args.num_iters // 4, 1)
        self.current_iter = 0

        coarse_res = max(args.grid_resolution // 4, 8)
        voxel_size = (2.0 * cams.grid_radius) / coarse_res
        origin = cams.target_center - cams.grid_radius

        self.grid = fvdb.Grid.from_dense(
            dense_dims=[coarse_res, coarse_res, coarse_res],
            voxel_size=voxel_size,
            origin=origin,
            device=device,
        )
        self.phi = nn.Parameter(torch.randn(self.grid.num_voxels, 1, device=device) * 0.1 + 0.5)
        self.optimizer = torch.optim.Adam([self.phi], lr=args.lr)

        print(f"[SparseAdaptiveGrid] Initial grid: {self.grid.num_voxels} voxels at coarse resolution {coarse_res}³ (voxel_size={voxel_size:.4f})")
        print(f"[SparseAdaptiveGrid] Refining every {self.refine_every} iters")

    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        N_rays, N_samples, _ = points.shape
        vals = self.grid.sample_trilinear(points.reshape(-1, 3), self.phi)  # [N_rays*N_samples, 1]
        return vals.reshape(N_rays, N_samples)

    def _dirichlet_energy(self) -> torch.Tensor:
        """Sparse Dirichlet energy: penalises phi differences between neighbouring active voxels only."""
        ijk = self.grid.ijk
        phi_flat = self.phi.squeeze(-1)
        total = torch.zeros(1, device=self.device)
        count = 0
        offsets = torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], device=self.device, dtype=torch.int32)
        for off in offsets:
            nb_idx = self.grid.ijk_to_index(ijk + off)
            mask = nb_idx >= 0
            if mask.sum() == 0:
                continue
            diff = phi_flat[mask] - phi_flat[nb_idx[mask]]
            total = total + (diff ** 2).sum()
            count += mask.sum().item()
        return total / max(count, 1)

    def _compute_voxel_reprojection_error(self, seg_result: SegmentationResult, cams: CameraState) -> torch.Tensor:
        """
        For each active voxel, compute its mean reprojection error across all views.
        1. Render the current predicted mask.
        2. Compute per-pixel error image: |pred - target|.
        3. Project every voxel centre into the image.
        4. Bilinearly sample the error image at each projected voxel location.
        """
        N = self.grid.num_voxels
        accumulated = torch.zeros(N, device=self.device)
        num_views = len(cams.viewmats)

        ijk = self.grid.ijk.float()
        voxel_centres = ijk * self.grid.voxel_size.to(self.device) + self.grid.origin.to(self.device)

        with torch.no_grad():
            for view_idx in range(num_views):
                pred_mask = self.render_mask(view_idx, cams, self.args.num_samples)
                target = torch.from_numpy(seg_result.masks[view_idx]).float().to(self.device).view(-1)
                error_flat = (pred_mask - target).abs()
                error_img = error_flat.reshape(1, 1, self.args.height, self.args.width)

                px = project_points(voxel_centres, cams.viewmats[view_idx], cams.Ks[view_idx])  # numpy [N, 2]
                px_t = torch.from_numpy(px).float().to(self.device)
                px_norm_x = (px_t[:, 0] / (self.args.width  - 1)) * 2 - 1
                px_norm_y = (px_t[:, 1] / (self.args.height - 1)) * 2 - 1
                grid_coords = torch.stack([px_norm_x, px_norm_y], dim=-1).reshape(1, N, 1, 2)

                sampled = F.grid_sample(error_img, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
                accumulated += sampled.reshape(N)

        return accumulated / num_views

    def _refine(self, seg_result: SegmentationResult, cams: CameraState):
        """Subdivide voxels with high mean reprojection error across views."""
        voxel_errors = self._compute_voxel_reprojection_error(seg_result, cams)
        threshold = voxel_errors.mean() + 0.5 * voxel_errors.std()
        refine_mask = voxel_errors > threshold
        n_refine = refine_mask.sum().item()
        print(f"  [Refine] error threshold={threshold:.4f}, marking {n_refine}/{self.grid.num_voxels} voxels")
        if n_refine == 0:
            print("  [Refine] Nothing to refine, skipping.")
            return
        refined_data, fine_grid = self.grid.refine(subdiv_factor=2, data=self.phi.data, mask=refine_mask)
        self.grid = fine_grid
        self.phi = nn.Parameter(refined_data.detach())
        self.optimizer = torch.optim.Adam([self.phi], lr=self.args.lr)
        print(f"  [Refine] Grid now has {self.grid.num_voxels} voxels (voxel_size={self.grid.voxel_size.tolist()})")

    def step(self, batch_indices: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        self.current_iter += 1
        total_loss = 0.0
        total_mask_loss = 0.0

        for view_idx in batch_indices:
            ray_origins, ray_dirs = construct_rays(cams.viewmats[view_idx], cams.Ks[view_idx], self.args.height, self.args.width, self.device)
            t_vals = torch.linspace(0.1, 10.0, self.args.num_samples, device=self.device)
            points = ray_origins[:, None, :] + ray_dirs[:, None, :] * t_vals[None, :, None]
            phi_vals = self.query(points, cams)
            alpha = torch.sigmoid(-self.args.sharpness * phi_vals)
            pred_mask = 1.0 - torch.prod(1.0 - alpha, dim=-1)

            target = torch.from_numpy(seg_result.masks[view_idx]).float().to(self.device).view(-1)
            mask_loss = self._compute_mask_loss(pred_mask, target)
            view_loss = mask_loss / self.args.num_views
            total_loss += view_loss
            total_mask_loss += view_loss.item()

        smooth_term = self.args.beta * self._dirichlet_energy()
        total_loss += smooth_term

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.current_iter % self.refine_every == 0:
            print(f"[SparseAdaptiveGrid] Iter {self.current_iter}: running surface refinement...")
            self._refine(seg_result, cams)

        return {"mask_loss": total_mask_loss, "smooth_loss": smooth_term.item()}

    def summarize(self):
        phi_flat = self.phi.data.squeeze(-1)
        n_surface = ((phi_flat - self.iso_level).abs() < self.band_width).sum().item()
        print(f"  voxels={self.grid.num_voxels}, phi min/max/mean/median: [{phi_flat.min():.4f}, {phi_flat.max():.4f}, {phi_flat.mean():.4f}, {phi_flat.median():.4f}], near-surface={n_surface}")

    def get_polyscope_data(self) -> PolyscopeData:
        ijk = self.grid.ijk.float()
        all_points = (ijk * self.grid.voxel_size + self.grid.origin).detach().cpu().numpy()
        all_phi = self.phi.data.squeeze(-1).detach().cpu().numpy()
        surface_mask = np.abs(all_phi - self.iso_level) < self.band_width
        return PolyscopeData(
            all_points=all_points.astype(np.float32),
            all_phi=all_phi.astype(np.float32),
            surface_points=all_points[surface_mask].astype(np.float32),
            surface_phi=all_phi[surface_mask].astype(np.float32),
        )


# --- UNIFIED PIPELINE ENTRY ---
def optimize_voxel_grid(grid: PhiGrid, seg_result: SegmentationResult, cams: CameraState, args, device: torch.device):
    history = {'total_loss': [], 'mask_loss': [], 'smooth_loss': [], 'time': []}
    start_time = time.time()

    for iter in range(args.num_iters):
        batch_indices = torch.randperm(args.num_views)[:args.batch_size].tolist()
        metrics = grid.step(batch_indices, seg_result, cams)

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