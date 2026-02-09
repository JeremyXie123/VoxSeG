# Purpose of this file:
# Load a gaussian splat into polyscope and visualize the bases representing the covariances of each guassian

import polyscope as ps
import numpy as np
import gpytoolbox as gpy
import open3d as o3d
import igl
from plyfile import PlyData
from scipy.spatial.transform import Rotation

def load_splat_ply(path):
    # https://python-plyfile.readthedocs.io/en/latest/usage.html#reading-a-ply-file
    ply = PlyData.read(path)
    v = ply["vertex"].data

    V = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)                           # Position
    S = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)         # Scale
    Q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)   # Quaternion

    return V, S, Q

def get_ellipsoid_mesh(V, S_linear, Q, R, density=10, scale=0.2):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=density)
    sphere_verts = np.asarray(sphere.vertices)
    sphere_faces = np.asarray(sphere.triangles)
    
    R = Rotation.from_quat(Q).as_matrix()

    all_verts = []
    all_faces = []
    
    for i in range(len(V)):
        # Scale, Rotate, Translate
        T = R[i] * S_linear[i] * scale
        v = (sphere_verts @ T.T) + V[i]
        
        all_faces.append(sphere_faces + len(all_verts) * len(sphere_verts))
        all_verts.append(v)
        
    return np.vstack(all_verts), np.vstack(all_faces)

# https://stackoverflow.com/questions/36920562/python-plyfile-vs-pymesh
if __name__ == "__main__":
    ps.init()
    
    V, S, Q = load_splat_ply("splats/bathtub/train/bathtub_0001/point_cloud.ply")
    R = Rotation.from_quat(Q).as_matrix().astype(np.float32)
    pc = ps.register_point_cloud("splats", V)

    S_linear = np.exp(S) # Some PLY files store logarithmic measurements
    A0 = R[:, :, 0] * S_linear[:, 0, None] # Scale first column of rotation matrix by S0
    A1 = R[:, :, 1] * S_linear[:, 1, None] # Scale second column of rotation matrix by S1
    A2 = R[:, :, 2] * S_linear[:, 2, None] # Scale third column of rotation matrix by S2

    pc.add_vector_quantity("axis_0", A0, enabled=True, color=(1, 0, 0))
    pc.add_vector_quantity("axis_1", A1, enabled=True, color=(0, 1, 0))
    pc.add_vector_quantity("axis_2", A2, enabled=True, color=(0, 0, 1))

    verts, faces = get_ellipsoid_mesh(V, S_linear, Q, R)
    ps.register_surface_mesh("splat_mesh", verts, faces)

    ps.show()