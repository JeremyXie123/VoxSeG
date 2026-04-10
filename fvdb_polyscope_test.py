"""
fvdb + polyscope test program
--------------------------------
Creates a sparse voxel grid from a point cloud using fvdb,
then visualizes both the original points and the voxel grid using polyscope.

Run with:
    conda activate spatial_pipeline
    python fvdb_polyscope_test.py
"""

import torch
import numpy as np
import fvdb
import polyscope as ps


def make_sphere_points(n=2000, radius=1.0, device="cuda"):
    """Generate random points on the surface of a sphere."""
    phi = torch.rand(n, device=device) * 2 * torch.pi
    costheta = torch.rand(n, device=device) * 2 - 1
    theta = torch.acos(costheta)

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    return torch.stack([x, y, z], dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Generate a point cloud (sphere surface) ---
    print("Generating point cloud...")
    points = make_sphere_points(n=3000, radius=1.0, device=device)

    # --- 2. Build a sparse fvdb voxel grid from the points ---
    print("Building fvdb sparse voxel grid...")
    voxel_size = 0.1
    # fvdb 0.4.x API: use GridBatch.from_points classmethod
    pcd_jagged = fvdb.JaggedTensor([points])
    grid = fvdb.GridBatch.from_points(
        pcd_jagged,
        voxel_sizes=[voxel_size] * 3,
        origins=[0.0] * 3,
    )

    n_voxels = grid.total_voxels
    print(f"  Voxel size:   {voxel_size}")
    print(f"  Total voxels: {n_voxels}")

    # --- 3. Get voxel centre positions for visualization ---
    voxel_centers = grid.voxel_to_world(fvdb.JaggedTensor([grid.ijk.jdata.float()])).jdata  # (N, 3)

    # --- 4. Assign a simple scalar: distance from origin ---
    point_dists  = torch.norm(points, dim=1)                           # per-point
    voxel_dists  = torch.norm(voxel_centers, dim=1)                    # per-voxel

    # Move everything to CPU / numpy for polyscope
    points_np       = points.cpu().numpy()
    point_dists_np  = point_dists.cpu().numpy()
    voxel_np        = voxel_centers.cpu().numpy()
    voxel_dists_np  = voxel_dists.cpu().numpy()

    # --- 5. Visualize with polyscope ---
    print("Launching polyscope viewer...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Original point cloud
    pc = ps.register_point_cloud("Sphere Point Cloud", points_np, radius=0.005)
    pc.add_scalar_quantity("distance from origin", point_dists_np,
                           enabled=True, cmap="viridis")

    # fvdb voxel centres
    vc = ps.register_point_cloud("fvdb Voxel Centres", voxel_np, radius=0.008)
    vc.add_scalar_quantity("voxel distance", voxel_dists_np,
                           enabled=True, cmap="coolwarm")

    print()
    print("=== Polyscope window open ===")
    print("  - 'Sphere Point Cloud' : original 3 000 surface points")
    print(f"  - 'fvdb Voxel Centres' : {n_voxels} voxels built by fvdb")
    print("  Toggle quantities in the left panel to switch colour maps.")
    print("  Close the window to exit.")
    print()

    ps.show()
    print("Done.")


if __name__ == "__main__":
    main()