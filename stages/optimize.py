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


class SparseAdaptiveGrid(PhiGrid):
    """
    Sparse voxel grid backed by fvdb.Grid, with error-driven adaptive refinement.

    Strategy (per supervisor instructions):
      1. Train for `refine_after` iterations at the coarse resolution.
      2. For every voxel, project its world-space centre into every training view
         and bilinearly sample the per-pixel error image |pred_mask - target_mask|.
      3. Average the sampled error across all views to get a scalar per voxel.
      4. Subdivide voxels whose mean error exceeds  mean + threshold_sigma * std.
      5. Detach phi, wrap in a new nn.Parameter, rebuild the Adam optimizer,
         and continue training on the finer grid.

    Smoothness is computed via the sparse 6-connected finite-difference trick
    (ijk_to_index with neighbour offsets) rather than dense slicing.
    """

    def __init__(self, args, device, cams: CameraState):
        super().__init__(args, device)

        # Hyper-parameters (can be promoted to argparse later)
        self.subdiv_factor     = getattr(args, 'subdiv_factor',      2)
        # Refine every N iterations.  phi needs time to converge at each
        # resolution before the error signal is meaningful enough to split on.
        # Too frequent = splits before convergence, runaway refinement.
        # Too infrequent = wastes iterations at coarse resolution.
        self.refine_every      = getattr(args, 'refine_every',       args.num_iters // 5)
        # Top-percentile criterion: subdivide only the highest-error voxels.
        self.refine_percentile = getattr(args, 'refine_percentile',  0.20)
        # Absolute floor: never split a voxel whose error is below this,
        # even if it falls in the top percentile.
        self.error_floor       = getattr(args, 'error_floor',        0.02)
        # Hard cap on total voxels — refinement stops once this is reached.
        # grid_resolution^3 matches the dense grid budget as a sensible default.
        self.max_voxels        = getattr(args, 'max_voxels', args.grid_resolution ** 3)
        # Minimum voxel size (metres) — hard stop on subdivision.
        self.min_voxel_size    = getattr(args, 'min_voxel_size',     0.01)

        # ------------------------------------------------------------------ #
        # Build the initial grid: a single voxel covering the whole bounding
        # box.  All resolution comes purely from error-driven refinement —
        # no voxel exists until the reprojection error criterion creates it.
        # ------------------------------------------------------------------ #
        center = cams.target_center.to(device)          # [3]

        # One voxel whose size equals the full bounding-box diameter
        voxel_size = 2.0 * cams.grid_radius

        # origin = world position of voxel [0,0,0] = centre of the bbox
        # (with voxel_size = diameter, voxel [0,0,0] spans center ± radius)
        origin = center - torch.tensor(cams.grid_radius, device=device)

        ijk_zero = torch.zeros(1, 3, dtype=torch.int32, device=device)
        self.grid = fvdb.Grid.from_ijk(
            ijk_zero,
            voxel_size=voxel_size,
            origin=origin,
        )

        # phi is a flat [N, 1] tensor — one value per active voxel
        n = self.grid.num_voxels
        phi_init = torch.randn(n, 1, device=device) * 0.1 + 0.5
        self.phi = nn.Parameter(phi_init)
        self.optimizer = torch.optim.Adam([self.phi], lr=args.lr)

        print(f"[SparseAdaptiveGrid] Initialized: {n} voxel (single root), voxel_size={voxel_size:.4f}")

    # ---------------------------------------------------------------------- #
    # query
    # ---------------------------------------------------------------------- #

    def query(self, points: torch.Tensor, cams: CameraState) -> torch.Tensor:
        """
        Trilinearly interpolate phi at world-space sample points.

        points: [N_rays, N_samples, 3]  (world space)
        returns: [N_rays, N_samples]
        """
        N_rays, N_samples, _ = points.shape
        flat_pts = points.reshape(-1, 3)                        # [N_rays*N_samples, 3]

        # sample_trilinear: points [M,3] world-space, data [V,1] -> [M,1]
        # Points outside active voxels return 0.0 (treated as "empty" / high phi)
        sampled = self.grid.sample_trilinear(flat_pts, self.phi)  # [M, 1]

        return sampled.reshape(N_rays, N_samples)

    # ---------------------------------------------------------------------- #
    # step
    # ---------------------------------------------------------------------- #

    def step(self, batch_indices: list[int], seg_result: SegmentationResult, cams: CameraState) -> dict:
        total_loss   = torch.zeros((), device=self.device)
        total_mask_loss = 0.0

        for view_idx in batch_indices:
            pred_mask = self.render_mask(view_idx, cams, self.args.num_samples)
            target = torch.from_numpy(seg_result.masks[view_idx]).float().to(self.device).view(-1)
            mask_loss = self._compute_mask_loss(pred_mask, target)
            view_loss = mask_loss / self.args.num_views
            total_loss      += view_loss
            total_mask_loss += view_loss.item()

        smooth_term = self._sparse_smoothness()
        total_loss += smooth_term

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"mask_loss": total_mask_loss, "smooth_loss": smooth_term.item()}

    # ---------------------------------------------------------------------- #
    # Sparse smoothness — 6-connected finite differences via ijk_to_index
    # ---------------------------------------------------------------------- #

    def _sparse_smoothness(self) -> torch.Tensor:
        """
        Dirichlet energy over active voxels using ijk neighbour lookups.
        Only counts edges where both voxels are active (index >= 0).
        """
        ijk = self.grid.ijk                              # [N, 3]  int32, on device
        phi_flat = self.phi.squeeze(-1)                  # [N]

        offsets = torch.tensor(
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
            device=self.device, dtype=torch.int32
        )

        total = torch.zeros((), device=self.device)
        for off in offsets:
            nb_idx = self.grid.ijk_to_index(ijk + off)   # [N]  (-1 = inactive)
            mask   = nb_idx >= 0
            if mask.any():
                diff  = phi_flat[mask] - phi_flat[nb_idx[mask]]
                total = total + (diff ** 2).sum()

        # Normalise by number of active voxels to keep scale comparable to DenseGrid
        return self.args.beta * total / max(self.grid.num_voxels, 1)

    # ---------------------------------------------------------------------- #
    # Adaptive refinement
    # ---------------------------------------------------------------------- #

    def maybe_refine(self, seg_result: SegmentationResult, cams: CameraState):
        """
        Called by the training loop at the right iteration.
        Projects each voxel centre into every training view, samples the error
        image, and subdivides high-error voxels.
        """
        print("[SparseAdaptiveGrid] Computing per-voxel reprojection error...")

        num_views  = len(cams.viewmats)
        num_voxels = self.grid.num_voxels

        # World-space voxel centres: ijk * voxel_size + origin
        ijk_float   = self.grid.ijk.float()                           # [N, 3]
        voxel_world = ijk_float * self.grid.voxel_size + self.grid.origin  # [N, 3]

        # Accumulate error across views
        error_sum = torch.zeros(num_voxels, device=self.device)

        with torch.no_grad():
            for view_idx in range(num_views):
                # --- render the current predicted mask for this view ---
                pred_flat = self.render_mask(view_idx, cams, self.args.num_samples)  # [H*W]
                pred_img  = pred_flat.reshape(self.args.height, self.args.width)     # [H, W]

                target_img = torch.from_numpy(
                    seg_result.masks[view_idx]
                ).float().to(self.device)                                            # [H, W]

                # --- per-pixel error image ---
                error_img = (pred_img - target_img).abs()                            # [H, W]

                # --- project voxel centres into this view (stay on-device) ---
                # World -> camera space
                ones      = torch.ones(num_voxels, 1, device=self.device)
                pts_homo  = torch.cat([voxel_world, ones], dim=1)                    # [N, 4]
                cam_pts   = (cams.viewmats[view_idx] @ pts_homo.T).T[:, :3]         # [N, 3]

                # Camera -> pixel space
                px = (cams.Ks[view_idx] @ cam_pts.T).T                              # [N, 3]
                px = px[:, :2] / px[:, 2:3]                                          # [N, 2]  (u, v)

                # Cull voxels that project outside the image or behind the camera
                in_front = cam_pts[:, 2] > 0
                in_u     = (px[:, 0] >= 0) & (px[:, 0] < self.args.width)
                in_v     = (px[:, 1] >= 0) & (px[:, 1] < self.args.height)
                visible  = in_front & in_u & in_v                                    # [N]  bool

                if not visible.any():
                    continue

                # Bilinearly sample error_img at the projected pixel positions.
                # grid_sample expects coords normalised to [-1, 1].
                u_norm = (px[visible, 0] / (self.args.width  - 1)) * 2 - 1          # [M]
                v_norm = (px[visible, 1] / (self.args.height - 1)) * 2 - 1          # [M]

                # grid_sample input: [1, 1, H, W], grid: [1, 1, M, 2]
                sample_grid = torch.stack([u_norm, v_norm], dim=-1).reshape(1, 1, -1, 2)
                sampled = F.grid_sample(
                    error_img[None, None],          # [1, 1, H, W]
                    sample_grid,                    # [1, 1, M, 2]
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )                                                                     # [1, 1, 1, M]
                error_sum[visible] += sampled.reshape(-1)

        mean_error = error_sum / num_views                                            # [N]

        # --- decide which voxels to subdivide ---
        # With very few voxels the std() is undefined (N=1) or meaningless,
        # so we subdivide everything until there are enough voxels for the
        # error distribution to have real variance.
        # Don't subdivide voxels that are already below the minimum size
        current_voxel_size = self.grid.voxel_size.min().item()
        if current_voxel_size / self.subdiv_factor < self.min_voxel_size:
            print(f"[SparseAdaptiveGrid] Voxel size {current_voxel_size:.4f} would go below "
                  f"min_voxel_size={self.min_voxel_size} — skipping refinement.")
            return

        if num_voxels >= self.max_voxels:
            print(f"[SparseAdaptiveGrid] Voxel budget exhausted "
                  f"({num_voxels} >= max_voxels={self.max_voxels}) — skipping refinement.")
            return

        if num_voxels < 8:
            refine_mask = torch.ones(num_voxels, dtype=torch.bool, device=self.device)
            print(f"[SparseAdaptiveGrid] Grid too coarse ({num_voxels} voxels) — subdividing all")
        else:
            # Threshold = the (1 - refine_percentile) quantile of error values,
            # but never below error_floor so converged voxels are left alone.
            quantile    = torch.quantile(mean_error, 1.0 - self.refine_percentile)
            threshold   = max(quantile.item(), self.error_floor)
            refine_mask = mean_error >= threshold                                    # [N] bool

            # Clamp so the post-refinement count stays within budget.
            # Each refined voxel produces subdiv_factor^3 children, and we
            # keep the unrefined ones too, so the new total is approximately:
            #   n_keep + n_refine * S^3  where S = subdiv_factor
            # Solve for the max n_refine that keeps us under max_voxels:
            S = self.subdiv_factor ** 3
            n_keep = num_voxels - refine_mask.sum().item()
            budget = max(0, self.max_voxels - n_keep)
            max_refine = budget // S
            if refine_mask.sum().item() > max_refine:
                # Keep only the highest-error voxels up to the budget
                topk = torch.topk(mean_error, k=int(max_refine))
                refine_mask = torch.zeros(num_voxels, dtype=torch.bool, device=self.device)
                refine_mask[topk.indices] = True

            n_refine = refine_mask.sum().item()
            print(f"[SparseAdaptiveGrid] Refining {n_refine}/{num_voxels} voxels "
                  f"(top {self.refine_percentile*100:.0f}% threshold={threshold:.4f}, "
                  f"mean={mean_error.mean().item():.4f}, max={mean_error.max().item():.4f})")
            if n_refine == 0:
                print("[SparseAdaptiveGrid] Nothing to refine, skipping.")
                return

        # --- fvdb refinement ---
        # grid.refine() only returns children of the masked voxels — unmasked
        # coarse voxels are dropped.  We need to union them back in manually.
        #
        # Strategy:
        #   A) Refine the masked voxels -> get fine children + their phi values
        #   B) Keep the unmasked coarse voxels as-is in the new grid
        #   C) Build a combined grid from the union of both ijk sets
        #   D) Assign phi by looking up each voxel's origin

        S = self.subdiv_factor
        old_ijk  = self.grid.ijk                          # [N, 3] int32
        old_phi  = self.phi.detach()                      # [N, 1]

        # A) Children of refined voxels
        child_phi, child_grid = self.grid.refine(
            subdiv_factor=S,
            data=old_phi,
            mask=refine_mask,
        )
        child_ijk = child_grid.ijk                        # [M*S^3, 3]

        # B) Unmasked coarse voxels — convert to world space first, then back
        #    to ijk in the fine coordinate system.  This is robust across
        #    multiple refinement passes because it never compounds a scale factor.
        #    world = ijk * old_voxel_size + old_origin  (exact, no accumulation)
        #    fine_ijk = round((world - old_origin) / fine_voxel_size)
        keep_mask     = ~refine_mask
        keep_ijk_old  = old_ijk[keep_mask]                # [K, 3] in old coords
        keep_phi      = old_phi[keep_mask]                # [K, 1]

        fine_voxel_size = child_grid.voxel_size           # [3]
        old_origin      = self.grid.origin                # [3]
        # World-space centre of each kept voxel
        keep_world = keep_ijk_old.float() * self.grid.voxel_size + old_origin  # [K, 3]
        # Convert to fine ijk (round to nearest integer)
        keep_ijk   = torch.round(
            (keep_world - old_origin) / fine_voxel_size
        ).to(torch.int32)                                 # [K, 3] in fine coords

        # C) Union: both sets are now in the fine coordinate frame.
        union_ijk = torch.cat([child_ijk, keep_ijk], dim=0)  # [M*S^3 + K, 3]
        new_grid  = fvdb.Grid.from_ijk(
            union_ijk,
            voxel_size=fine_voxel_size,
            origin=old_origin,
        )

        # D) Assign phi to every voxel in the union grid.
        #    Look up each group by ijk index in the new grid.
        new_n       = new_grid.num_voxels
        new_phi_buf = torch.zeros(new_n, 1, device=self.device)

        child_idx = new_grid.ijk_to_index(child_ijk)     # [M*S^3]
        valid_c   = child_idx >= 0
        new_phi_buf[child_idx[valid_c]] = child_phi[valid_c]

        keep_idx  = new_grid.ijk_to_index(keep_ijk)      # [K]
        valid_k   = keep_idx >= 0
        new_phi_buf[keep_idx[valid_k]] = keep_phi[valid_k]

        # Cut the old compute graph and register as a new learnable parameter
        self.grid = new_grid
        self.phi  = nn.Parameter(new_phi_buf.detach())
        self.optimizer = torch.optim.Adam([self.phi], lr=self.args.lr)

        n_children = child_ijk.shape[0]
        n_kept     = keep_ijk.shape[0]
        print(f"[SparseAdaptiveGrid] After refinement: {self.grid.num_voxels} voxels "
              f"({n_children} children + {n_kept} kept coarse)")

    # ---------------------------------------------------------------------- #
    # summarize / get_polyscope_data
    # ---------------------------------------------------------------------- #

    def summarize(self):
        phi_flat = self.phi.detach().squeeze(-1)
        phi_min    = phi_flat.min().item()
        phi_max    = phi_flat.max().item()
        phi_mean   = phi_flat.mean().item()
        phi_median = phi_flat.median().item()
        print(f"[Sparse] voxels={self.grid.num_voxels} | "
              f"phi min/max/avg/med: [{phi_min:.4f}, {phi_max:.4f}, {phi_mean:.4f}, {phi_median:.4f}]")

    def get_polyscope_data(self) -> PolyscopeData:
        ijk_float   = self.grid.ijk.float().cpu().numpy()
        voxel_size  = self.grid.voxel_size.cpu().numpy()        # [3]
        origin      = self.grid.origin.cpu().numpy()
        all_points  = ijk_float * voxel_size + origin           # world-space centres [N,3]
        # Every voxel in a fvdb.Grid shares the same voxel_size
        all_voxel_sizes = np.broadcast_to(voxel_size, all_points.shape).copy()

        phi_np      = self.phi.detach().squeeze(-1).cpu().numpy()
        surface_mask = np.abs(phi_np - self.args.iso_level) < 0.5

        return PolyscopeData(
            all_points     = all_points.astype(np.float32),
            all_phi        = phi_np.astype(np.float32),
            all_voxel_sizes = all_voxel_sizes.astype(np.float32),
            surface_points = all_points[surface_mask].astype(np.float32),
            surface_phi    = phi_np[surface_mask].astype(np.float32),
        )


# --------------------------------------------------------------------------- #
# UNIFIED PIPELINE ENTRY
# --------------------------------------------------------------------------- #

def optimize_voxel_grid(grid: PhiGrid, seg_result: SegmentationResult, cams: CameraState, args, device: torch.device):
    history = {'total_loss': [], 'mask_loss': [], 'smooth_loss': [], 'time': []}
    start_time = time.time()

    for iter in range(args.num_iters):

        # Refine on a schedule so phi has time to converge at each resolution
        # before the error signal is used to decide where to split.
        if (
            isinstance(grid, SparseAdaptiveGrid)
            and iter > 0
            and iter % grid.refine_every == 0
        ):
            grid.maybe_refine(seg_result, cams)

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