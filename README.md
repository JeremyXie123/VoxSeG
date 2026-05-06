# VoxSeG
This is the repository for the VoxSeG paper.

# Installation
The order of installation is important, given many libraries we use require specific versions of pytorch (gsplat, sam3, etc.).

Due to the complexity of the requirements, a simple `requirements.txt` like file is not possible. Please follow the instructions below.

### Install system dependencies
```bash
sudo apt install cmake python-is-python3 ninja-build build-essential git
```

### Install CUDA 12.8 Toolkit (Skip if applicable)
```bash
nvcc --version # If already on 12.8, skip the rest of this step.
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --override
rm cuda_12.8.0_570.86.10_linux.run
```

### Add CUDA to PATH (Skip if applicable)
```bash
# If these are already in your PATH, skip this step
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Setup conda environment
```bash
conda create -n spatial_pipeline python=3.12 -y
conda activate spatial_pipeline
```

### Install PyTorch 2.10.0 with CUDA 12.8 ---
```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Install sam3 and its undeclared dependencies
```bash
# Create a hugging face account and request access to the SAM3 model (https://huggingface.co/facebook/sam3)
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..

pip install einops pycocotools psutil
pip install "numpy<2" "setuptools<81" "tifffile<2025
```

### Install gsplat and its dependencies
```bash
pip install ninja jaxtyping rich
pip install gsplat
```

### Install polyscope and other pipeline dependencies
```bash
pip install polyscope
pip install matplotlib plyfile scikit-image
pip install open3d scipy
```

### Install Hugging Face for sam3 model weights
```bash
# Use the hugging face account you created earlier
pip install huggingface_hub
hf auth login
```

### Verify everything was installed propperly
```bash
pip check   # should print "No broken requirements found."

python -c "
import torch, torchvision, sam3, polyscope, gsplat, skimage, plyfile
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('polyscope:', polyscope.__version__)
print('gsplat:', gsplat.__version__)
print('skimage:', skimage.__version__)
print('CUDA available:', torch.cuda.is_available())
from sam3.model_builder import build_sam3_image_model
print('sam3 import OK')
"
```

# Running the main script
The script uses argparse to run the program from the command line for reproducability and testing. The meaning of each argument can be found in the code, or through `python3 pipeline.py --help`.

If polyscope opens but no scene is rendered, run `rm -rf ~/.cache/torch_extensions/` to force gsplat to recompile.

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

# Extra Notes
- The `hypercomp.py` file can be used to generate plots of parameter sweeps, e.g. values of `beta` for comparison.
- The code was ran with CUDA 12.8, so it may not work for other versions.
