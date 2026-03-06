import torch
import polyscope as ps
import polyscope.imgui as psim
import gsplat

import argparse
import numpy as np
from plyfile import PlyData

print("polyscope file:", ps.__file__)
print("gsplat file:", gsplat.__file__)
print("polyscope version:", ps.__version__)
print("gsplat version:", gsplat.__version__)


def load_gaussians_from_ply(path_ply, device='cuda'):
    plydata = PlyData.read(path_ply)

    centers = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )  # (N, 3)

    opacities = np.asarray(plydata.elements[0]["opacity"])  # (N,)

    # DC SH coefficients as (N, 1, 3), not (N, 3)
    features_dc = np.stack(
        (
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ),
        axis=1,
    )[:, None, :]  # (N, 1, 3)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((centers.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((centers.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    with torch.no_grad():
        centers = torch.tensor(centers, dtype=torch.float32, device=device)        # (N, 3)
        features_dc = torch.tensor(features_dc, dtype=torch.float32, device=device) # (N, 1, 3)
        opacity = torch.tensor(opacities, dtype=torch.float32, device=device)      # (N,)
        scaling = torch.tensor(scales, dtype=torch.float32, device=device)         # (N, 3)
        rotation = torch.tensor(rots, dtype=torch.float32, device=device)          # (N, 4)

    # activate stored parameters
    opacity = torch.sigmoid(opacity)
    scaling = torch.exp(scaling)

    return centers, features_dc, opacity, scaling, rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian_ply', type=str, required=True, help='path to a .ply file')
    parser.add_argument('--device', type=str, default='cuda', help='torch device: cuda or cpu')
    args = parser.parse_args()

    ps.init()
    ps.set_ground_plane_mode("none")

    centers, features_dc, opacity, scaling, rotation = load_gaussians_from_ply(
        args.gaussian_ply, device=args.device
    )

    print(f"Loaded {centers.shape[0]} gaussian particles")
    print(f"centers shape:      {centers.shape}")
    print(f"features_dc shape:  {features_dc.shape}")
    print(f"opacity shape:      {opacity.shape}")
    print(f"scaling shape:      {scaling.shape}")
    print(f"rotation shape:     {rotation.shape}")

    # Add leading batch dimension to match Polyscope's camera tensors [1, C, 4, 4]
    means_b = centers.unsqueeze(0)        # (1, N, 3)
    colors_b = features_dc.unsqueeze(0)   # (1, N, 1, 3)
    opacity_b = opacity.unsqueeze(0)      # (1, N)
    scales_b = scaling.unsqueeze(0)       # (1, N, 3)
    quats_b = rotation.unsqueeze(0)       # (1, N, 4)

    print(f"means_b shape:      {means_b.shape}")
    print(f"colors_b shape:     {colors_b.shape}")
    print(f"opacity_b shape:    {opacity_b.shape}")
    print(f"scales_b shape:     {scales_b.shape}")
    print(f"quats_b shape:      {quats_b.shape}")

    ps.register_gaussian_particles(
        "gaussians",
        subsample_factor=2,
        means=means_b,
        colors=colors_b,
        opacities=opacity_b,
        scales=scales_b,
        quats=quats_b,
        sh_degree=0,
    )

    ps.show()


if __name__ == '__main__':
    main()