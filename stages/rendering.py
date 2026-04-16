import gc
import torch
from gsplat import rasterization

from core.camera import CameraState
from core.splat_io import SplatData, sh_to_rgb

def render_splat_views(splats: SplatData, cams: CameraState, args, chunk_size: int = 50) -> torch.Tensor:
    """
    Renders the given camera views in batches to manage GPU memory usage.
    """
    print(f"Rendering {len(cams.viewmats)} views...")
    all_renders = []
    num_views = len(cams.viewmats)
    
    # Convert SH to RGB for gsplat rendering
    rgb_colors = sh_to_rgb(splats.colors)
    
    for i in range(0, num_views, chunk_size):
        end = min(i + chunk_size, num_views)
        print(f"Rendering batch {i} to {end}...")
        
        renders, _, _ = rasterization(
            means=splats.means, 
            quats=splats.quats, 
            scales=torch.exp(splats.scales), 
            opacities=splats.opacities,
            colors=rgb_colors[None, :, :].expand(args.num_views, -1, -1)[i:end],
            viewmats=cams.viewmats[i:end], 
            Ks=cams.Ks[i:end],
            width=args.width, 
            height=args.height,
            sh_degree=None
        )
        
        all_renders.append(renders)
        
        del renders
        gc.collect()
        torch.cuda.empty_cache()
        
    return torch.cat(all_renders, dim=0)