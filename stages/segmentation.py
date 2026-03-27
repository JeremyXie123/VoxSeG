import os
import gc
import tempfile
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from sam2.build_sam import build_sam2_video_predictor

# Import the camera state from your new core module
from core.camera import CameraState, project_points

@dataclass
class SegmentationResult:
    """Container for the output masks and visualization renders."""
    masks: np.ndarray
    blended_images: list[np.ndarray]

def generate_sam_masks(rendered_images: torch.Tensor, interior_3d: torch.Tensor, cams: CameraState, args, device: torch.device) -> SegmentationResult:
    """
    Runs SAM 2.1 Video Predictor by pointing it to a high-speed temp directory,
    then computes blended visualizations.
    """
    print("Initializing SAM 2.1 Video Predictor...")
    # Note: We pass args.sam_config directly so Hydra can resolve its internal pkg:// path
    predictor = build_sam2_video_predictor(args.sam_config, args.sam_checkpoint, device=device)
    
    # SAM's video model requires a directory of images as input.
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {len(rendered_images)} frames...")
        
        for i in range(len(rendered_images)):
            img_np = (rendered_images[i].detach().clamp(0, 1) * 255).byte().cpu().numpy()
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(temp_dir, f"{i:05d}.jpg"), quality=85)

        # Initialize the video inference state
        print("Initializing SAM state from folder...")
        inference_state = predictor.init_state(video_path=temp_dir)
        
        # Add interior points as an initial prompt to Frame 0
        input_points = project_points(interior_3d, cams.viewmats[0], cams.Ks[0])
        input_labels = np.ones(len(input_points), dtype=np.int32)
        
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=1,
            points=input_points, labels=input_labels
        )

        # Propagate masks across all frames
        print("Propagating masks...")
        total_frames = len(rendered_images)
        final_masks = [None] * total_frames 

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
            final_masks[out_frame_idx] = mask

        target_masks = np.stack(final_masks)

    # Generate blended images (red tint) for Polyscope visualization later
    original_renders = rendered_images.detach().cpu().numpy()
    blended_images = []
    
    for i in range(args.num_views):
        tinted = np.zeros((*original_renders.shape[1:3], 3), dtype=np.float32)
        tinted[..., 0] = 1.0 
        alpha_map = (target_masks[i] * 0.5)[..., None].astype(np.float32)
        blended = original_renders[i] * (1.0 - alpha_map) + tinted * alpha_map
        blended_images.append(blended)

    # Aggressively clean up the rasterized images to free VRAM for the optimization stage
    print("Cleaning up rasterized memory...")
    del rendered_images
    gc.collect()
    torch.cuda.empty_cache()

    return SegmentationResult(
        masks=target_masks,
        blended_images=blended_images
    )