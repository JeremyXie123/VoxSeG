# VoxSeG
This is the repository for the VoxSeG paper.

# Installation
The order of installation is important, given many libraries we use require specific versions of pytorch (gsplat, sam3, etc.)

# Running the main script
The script uses argparse to run the program from the command line for reproducability and testing. The meaning of each argument can be found in the code, or through `python3 pipeline.py --help`
- `python3 pipeline.py --input splats/truck.ply --box_center 0.2128 0.2911 0.4706 --box_size 5.7407 1.7651 2.1704 --box_angles 170.42 2.86 -178.06 --label="Truck" --resolution 512 --focal_length=300 --num_iters=50 --padding=1.25`
- `python3 pipeline.py --input splats/train.ply --box_center -0.5892 -0.2067 -0.0802 --box_size 6.6744 1.9212 1.6552  --box_angles -173.45 28.43 -179.19 --label="Train" --resolution 512 --focal_length 300.0 --num_iters=50 --padding 1.50`
- `python3 pipeline.py --input splats/panther.ply --box_center -0.2041 0.2134 0.5350 --box_size 4.4317 2.1076 2.3953 --box_angles 0.00 43.63 0.00 --label="Tank" --resolution 512 --focal_length 300.0 --num_iters=50 --padding 1.50 `
- `python3 pipeline.py --input="splats/ignatius.ply" --box_center 0.3738 0.1920 -0.0366 --box_size 1.9203 2.8647 1.7559 --box_angles 171.64 39.65 174.66 --label="Statue" --resolution 512 --focal_length 300.0 --num_iters=50 --padding 1.50`
- `python3 pipeline.py --input="splats/m60.ply" --box_center -0.4267 0.0921 0.2831 --box_size 4.5081 2.1180 2.4946 --box_angles 96.58 83.57 96.54 --label="Tank" --resolution 512 --focal_length 300.0 --num_iters=50 --padding 1.50`

# Example Gaussians 
Here are some example public gaussian splat repositories you can download `.ply` files from.

## Splats With Ground Truth Geometry
- `TODO`

## Single Objects
- https://huggingface.co/datasets/ShapeSplats/ModelNet_Splats

## Scenes
- https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/tree/main
- https://huggingface.co/datasets/Voxel51/gaussian_splatting/tree/main/FO_dataset

# Troubleshooting
- If the polyscope window opens but the gaussian scene is not rendered, please run the following to force gsplat to recompile on next launch:
  
  `rm -rf ~/.cache/torch_extensions/`

  Please be mindful that this may remove other torch extensions.
- The code was ran with CUDA 12.8, so it may not work for other versions.

# Extra Notes
- The `hypercomp.py` file can be used to generate plots of parameter sweeps, e.g. values of `beta` for comparison.