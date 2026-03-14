import torch
import gc
from gsplat import rasterization

def print_gpu_memory(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"--- {label} ---")
    print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# 1. Baseline
print_gpu_memory("Initial State")

# 2. Mock Data (Matches your truck.ply scale)
num_points = 1_000_000
# 1. Force all input tensors to float32
means = torch.randn(num_points, 3, device='cuda', dtype=torch.float32)
quats = torch.randn(num_points, 4, device='cuda', dtype=torch.float32)
scales = torch.randn(num_points, 3, device='cuda', dtype=torch.float32)
colors = torch.randn(num_points, 3, device='cuda', dtype=torch.float32)
opacities = torch.ones(num_points, device='cuda', dtype=torch.float32) # Must be 1D shape (N,)
viewmats = torch.eye(4, device='cuda', dtype=torch.float32)[None]
Ks = torch.tensor([[550, 0, 256], [0, 550, 256], [0, 0, 1]], device='cuda', dtype=torch.float32)[None]

print_gpu_memory("Before Rasterization")

# 3. Call Rasterizer
# This is the line that likely pins the 8.9GB
rendered_images, alphas, meta = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors, viewmats=viewmats, Ks=Ks, 
    width=512, height=512, sh_degree=None, 
    backgrounds=torch.zeros((1, 3), device='cuda')
)
print_gpu_memory("After Rasterization")

# 4. Attempt Cleanup
del rendered_images, alphas, meta, means, quats, scales, colors, opacities
gc.collect()
torch.cuda.empty_cache()
print_gpu_memory("After Cleanup")