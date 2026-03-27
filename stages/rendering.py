import gc
import torch
from gsplat import rasterization

# Import the strict types from your core module
from core.camera import CameraState
from core.splat_io import SplatData

def render_splat_views(splats: SplatData, cams: CameraState, args, chunk_size: int = 50) -> torch.Tensor:
    """
    Renders the given camera views in batches to manage GPU memory usage.
    """
    print(f"Rendering {len(cams.viewmats)} views...")
    all_renders = []
    num_views = len(cams.viewmats)
    
    for i in range(0, num_views, chunk_size):
        end = min(i + chunk_size, num_views)
        print(f"Rendering batch {i} to {end}...")
        
        # We explicitly expand scales inside the loop so we aren't storing massive
        # expanded tensors in VRAM before they are strictly needed by the rasterizer.
        renders, _, _ = rasterization(
            means=splats.means, 
            quats=splats.quats, 
            scales=torch.exp(splats.scales), 
            opacities=splats.opacities,
            colors=splats.colors[None, :, :].expand(args.num_views, -1, -1)[i:end],
            viewmats=cams.viewmats[i:end], 
            Ks=cams.Ks[i:end],
            width=args.width, 
            height=args.height,
            sh_degree=None
        )
        
        all_renders.append(renders)
        
        # Explicitly free intermediate buffers as a memory saving measure
        del renders
        gc.collect()
        torch.cuda.empty_cache()
        
    return torch.cat(all_renders, dim=0)