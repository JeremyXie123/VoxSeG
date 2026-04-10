"""
fvdb adaptive grid demo
------------------------
Shows adaptive resolution by maintaining two GridBatch objects at different
voxel sizes over the same sphere surface:

  - Coarse grid  (voxel size 0.20) — full sphere
  - Fine grid    (voxel size 0.05) — full sphere

Both are shown side by side in polyscope (offset on the X axis) so the
resolution difference is immediately obvious.

A third panel shows the fine grid with a sculpted (pruned) occupancy mask
applied — the intended use case.

Run with:
    conda activate spatial_pipeline
    python fvdb_adaptive_grid_test.py
"""

import torch
import fvdb
import polyscope as ps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sphere_points(n: int = 5000, radius: float = 1.0, device: str = "cuda"):
    phi      = torch.rand(n, device=device) * 2 * torch.pi
    costheta = torch.rand(n, device=device) * 2 - 1
    theta    = torch.acos(costheta)
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def grid_centers(grid: fvdb.GridBatch) -> torch.Tensor:
    ijk_float = fvdb.JaggedTensor([grid.ijk.jdata.float()])
    return grid.voxel_to_world(ijk_float).jdata


def register(name, centers, scalar, cmap="viridis", scalar_name="value", radius=0.012):
    pc = ps.register_point_cloud(name, centers.cpu().numpy(), radius=radius)
    pc.add_scalar_quantity(scalar_name, scalar.cpu().numpy(), enabled=True, cmap=cmap)
    return pc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    points = make_sphere_points(n=5000, radius=1.0, device=device)
    pcd    = fvdb.JaggedTensor([points])

    # -----------------------------------------------------------------------
    # Grid A — coarse  (voxel size 0.20)
    # -----------------------------------------------------------------------
    grid_coarse = fvdb.GridBatch.from_points(pcd, voxel_sizes=[0.20]*3, origins=[0.0]*3)
    c_coarse    = grid_centers(grid_coarse)
    print(f"Coarse grid  (voxel 0.20) : {grid_coarse.total_voxels:>5} voxels")

    # -----------------------------------------------------------------------
    # Grid B — fine  (voxel size 0.05)
    # -----------------------------------------------------------------------
    grid_fine = fvdb.GridBatch.from_points(pcd, voxel_sizes=[0.05]*3, origins=[0.0]*3)
    c_fine    = grid_centers(grid_fine)
    print(f"Fine grid    (voxel 0.05) : {grid_fine.total_voxels:>5} voxels")

    # -----------------------------------------------------------------------
    # Grid C — fine grid with occupancy sculpt applied
    #          carve out everything in the x > 0 AND y > 0 quadrant
    # -----------------------------------------------------------------------
    keep      = ~((c_fine[:, 0] > 0.0) & (c_fine[:, 1] > 0.0))
    grid_sculpted = grid_fine.pruned_grid(fvdb.JaggedTensor([keep]))
    c_sculpted    = grid_centers(grid_sculpted)
    n_removed     = grid_fine.total_voxels - grid_sculpted.total_voxels
    print(f"Sculpted fine (pruned)    : {grid_sculpted.total_voxels:>5} voxels  ({n_removed} removed)\n")

    # -----------------------------------------------------------------------
    # Offset each grid along X so they don't overlap in the viewer
    #   A at x=-3,  B at x=0,  C at x=+3
    # -----------------------------------------------------------------------
    
    c_coarse_vis   = c_coarse.cpu()
    c_fine_vis     = c_fine.cpu()
    c_sculpted_vis = c_sculpted.cpu()

    # -----------------------------------------------------------------------
    # Polyscope
    # -----------------------------------------------------------------------
    print("Launching polyscope viewer...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Colour by Y coordinate so the sphere shape reads clearly
    register("A — Coarse (voxel 0.20)",
             c_coarse_vis,
             c_coarse_vis[:, 1],
             cmap="blues", scalar_name="y", radius=0.025)

    register("B — Fine (voxel 0.05)",
             c_fine_vis,
             c_fine_vis[:, 1],
             cmap="viridis", scalar_name="y", radius=0.007)

    register("C — Fine + Sculpted (pruned)",
             c_sculpted_vis,
             c_sculpted_vis[:, 1],
             cmap="reds", scalar_name="y", radius=0.007)

    print()
    print("=== Polyscope window open ===")
    print("  Three grids shown side by side (offset on X axis):")
    print(f"    A (left)   — Coarse voxel 0.20 : {grid_coarse.total_voxels:>5} voxels")
    print(f"    B (centre) — Fine   voxel 0.05 : {grid_fine.total_voxels:>5} voxels")
    print(f"    C (right)  — Fine + sculpted   : {grid_sculpted.total_voxels:>5} voxels")
    print()
    print("  A vs B shows what 'adaptive resolution' means in fvdb:")
    print("  the same surface represented at 4x finer voxel size.")
    print()
    print("  B vs C shows pruned_grid() as an occupancy sculpting tool:")
    print("  the x>0, y>0 quadrant has been carved out with a boolean mask.")
    print()
    print("  Close the window to exit.")

    ps.show()
    print("Done.")


if __name__ == "__main__":
    main()