import torch
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass

@dataclass
class SplatData:
    """Container for the raw Gaussian Splat attributes."""
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

def load_ply(path: str, device: torch.device) -> SplatData:
    """Loads a PLY file and maps its attributes to GPU tensors."""
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    means = torch.stack([torch.tensor(v['x']), torch.tensor(v['y']), torch.tensor(v['z'])], dim=-1).to(device)
    scales = torch.stack([torch.tensor(v['scale_0']), torch.tensor(v['scale_1']), torch.tensor(v['scale_2'])], dim=-1).to(device)
    quats = torch.stack([torch.tensor(v['rot_0']), torch.tensor(v['rot_1']), torch.tensor(v['rot_2']), torch.tensor(v['rot_3'])], dim=-1).to(device)
    colors = torch.stack([torch.tensor(v['f_dc_0']), torch.tensor(v['f_dc_1']), torch.tensor(v['f_dc_2'])], dim=-1).to(device)
    colors = (colors * 0.28209) + 0.5 # Normalize spherical harmonic colors to [0, 1]
    opacities = torch.sigmoid(torch.tensor(v['opacity'])).to(device)
    
    return SplatData(means, scales, quats, colors, opacities)

def print_gpu_memory():
    """Prints the current GPU memory usage in GB."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB")
    