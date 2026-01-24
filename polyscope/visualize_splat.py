import polyscope as ps
import numpy as np
import gpytoolbox as gpy
import open3d as o3d

if __name__ == "__main__":
    ps.init()
    
    # Load vertices and faces from ply
    V, F = gpy.read_mesh("splats/airplane/train/airplane_0001/point_cloud.ply")
    ps.register_point_cloud("ground truth", V)
    
    print("Vertices shape:", V.shape)
    print("Faces shape: ", F.shape)

    # Open3d normal estimation from point cloud
    # https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    N = np.asarray(pcd.normals, dtype=np.float64)
    
    # Perform poisson reconstruction
    # The 0.3.7 API does things a bit differnetly: https://gpytoolbox.org/latest/point_cloud_to_mesh
    V_rec, F_rec = gpy.point_cloud_to_mesh(V, N, method="PSR", psr_depth=8)
    ps.register_surface_mesh("psr reconstruction", V_rec, F_rec, smooth_shade=True)

    ps.show()