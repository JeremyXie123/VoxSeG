import polyscope as ps
import numpy as np
import gpytoolbox as gpy

if __name__ == "__main__":
    ps.init()

    # Load vertices and faces from obj (we will use .ply later)
    V, F = gpy.read_mesh("models/bunny.obj")
    ps.register_surface_mesh("ground truth", V, F)

    # Compute per-vertex normals
    N = gpy.per_vertex_normals(V, F)

    # Perform poisson reconstruction
    # The 0.3.7 API does things a bit differnetly: https://gpytoolbox.org/latest/point_cloud_to_mesh
    V_rec, F_rec = gpy.point_cloud_to_mesh(V, N, method="PSR", psr_depth=8)
    ps.register_surface_mesh("psr reconstruction", V_rec, F_rec, smooth_shade=True)

    ps.show()
