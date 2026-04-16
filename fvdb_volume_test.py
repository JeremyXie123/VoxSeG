"""
fVDB Basic Concepts Tutorial - Visualized with Polyscope
"""

import fvdb
from fvdb.utils.examples import load_car_1_mesh, load_car_2_mesh
import torch
import polyscope as ps


def jet(t: torch.Tensor) -> torch.Tensor:
    """Jet colormap: t in [0,1] -> RGB tensor same shape as t but with extra dim."""
    t = t.clamp(0, 1)
    r = (1.5 - abs(t - 0.75) * 4).clamp(0, 1)
    g = (1.5 - abs(t - 0.50) * 4).clamp(0, 1)
    b = (1.5 - abs(t - 0.25) * 4).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)

def pos_to_color(pts: torch.Tensor) -> torch.Tensor:
    """Jet colormap applied to the Y axis (height), matching the tutorial images."""
    y = pts[:, 1]
    t = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return jet(t)


def center_xz(pts: torch.Tensor) -> torch.Tensor:
    pts = pts.clone()
    pts[:, 0] -= (pts[:, 0].max() + pts[:, 0].min()) / 2
    pts[:, 2] -= (pts[:, 2].max() + pts[:, 2].min()) / 2
    return pts


# ── 1. Load and position point clouds ────────────────────────────────────────
pts1, _ = load_car_1_mesh(mode="vf")
pts2, _ = load_car_2_mesh(mode="vf")

pts1 = center_xz(pts1)
pts2 = center_xz(pts2)

# Print scale info so we can choose a good voxel size
extent1 = pts1.max(0).values - pts1.min(0).values
print(f"Car 1 extent: {extent1}")
print(f"Car 1 range: min={pts1.min(0).values}, max={pts1.max(0).values}")

# Voxel size = ~1/50th of the longest dimension gives ~50 voxels across
longest = extent1.max().item()
voxel_size = longest / 50.0
print(f"Using voxel_size={voxel_size:.4f}")

gap = voxel_size * 5
pts2 += torch.tensor(
    [0.0, 0.0, pts1[:, 2].max().item() - pts2[:, 2].min().item() + gap],
    device=pts2.device,
)

clrs1 = pos_to_color(pts1)
clrs2 = pos_to_color(pts2)

# ── 2. Build GridBatch ────────────────────────────────────────────────────────
points = fvdb.JaggedTensor([pts1, pts2])
colors = fvdb.JaggedTensor([clrs1, clrs2])
grid = fvdb.GridBatch.from_points(points, voxel_sizes=voxel_size)

print(f"Grid 1: {grid.num_voxels_at(0)} voxels")
print(f"Grid 2: {grid.num_voxels_at(1)} voxels")

# ── 3. Splat colors onto voxels ───────────────────────────────────────────────
vox_colors = grid.splat_trilinear(points, colors)

# ── 4. Sample random points ───────────────────────────────────────────────────
def random_points_in_bbox(pts, n):
    lo = pts.min(0).values
    hi = pts.max(0).values
    return lo + torch.rand(n, 3, device=pts.device) * (hi - lo)

rpts1 = random_points_in_bbox(pts1, 10_000)
rpts2 = random_points_in_bbox(pts2, 11_000)
sampled_colors = grid.sample_trilinear(fvdb.JaggedTensor([rpts1, rpts2]), vox_colors)

# ── 5. Polyscope ──────────────────────────────────────────────────────────────
def to_np(t):
    return t.detach().cpu().numpy()

ps.init()
ps.set_up_dir("y_up")

# Stage 1: Input point clouds (visible by default)
for i, (pts, clrs) in enumerate([(pts1, clrs1), (pts2, clrs2)]):
    pc = ps.register_point_cloud(f"input_points_car{i+1}", to_np(pts), radius=voxel_size * 0.15)
    pc.add_color_quantity("color", to_np(clrs), enabled=True)

# Stage 2: Per-voxel splatted colors (hidden by default)
vox_ijk = grid.ijk
for i in range(grid.grid_count):
    world = vox_ijk[i].jdata.float() * grid.voxel_sizes[i] + grid.origins[i]
    c = to_np(vox_colors[i].jdata).clip(0, 1)
    pc = ps.register_point_cloud(f"voxel_colors_car{i+1}", to_np(world), radius=voxel_size * 0.4)
    pc.add_color_quantity("splatted_color", c, enabled=True)
    pc.set_enabled(False)

# Stage 3: Per-voxel splatted colors using Polyscope SparseVolumeGrid
for i in range(grid.grid_count):
    occupied_cells = to_np(grid.ijk[i].jdata)
    origin = to_np(grid.origins[i])
    vs = grid.voxel_sizes[i][0].item() 
    cell_width = (vs, vs, vs)
    
    ps_grid = ps.register_sparse_volume_grid(
        f"voxel_grid_car{i+1}", 
        origin, 
        cell_width, 
        occupied_cells
    )
    
    c = to_np(vox_colors[i].jdata).clip(0, 1)
    ps_grid.add_color_quantity("splatted_color", c, defined_on='cells', enabled=True)
    ps_grid.set_enabled(False)

# Stage 4: Randomly sampled points (hidden by default)
for i, rpts in enumerate([rpts1, rpts2]):
    sc = to_np(sampled_colors[i].jdata).clip(0, 1)
    mask = sc.sum(axis=1) > 0.01
    pc = ps.register_point_cloud(f"sampled_points_car{i+1}", to_np(rpts)[mask], radius=voxel_size * 0.15)
    pc.add_color_quantity("sampled_color", sc[mask], enabled=True)
    pc.set_enabled(False)

ps.show()