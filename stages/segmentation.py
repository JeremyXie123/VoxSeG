import os
import gc
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

from core.camera import CameraState, project_points


@dataclass
class SegmentationResult:
    """Container for the output masks and visualization renders."""
    masks: np.ndarray
    blended_images: list[np.ndarray]


def project_points_to_image(points_3d: np.ndarray, viewmat: torch.Tensor, K: torch.Tensor, W: int, H: int):
    """
    Project (N,3) world-space points into pixel coords for one camera.
    Returns (coords (M,2) int [u,v], valid_mask (N,) bool).
    """
    if len(points_3d) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=bool)

    pts_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1).T
    vm = viewmat.cpu().numpy()
    K_np = K.cpu().numpy()

    cam = vm @ pts_h
    z = cam[2]
    valid = z > 0.01

    cam_xy = cam[:2, valid] / z[valid]
    pix_h = K_np @ np.vstack([cam_xy, np.ones((1, valid.sum()))])
    u = pix_h[0].astype(int)
    v = pix_h[1].astype(int)

    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    coords = np.stack([u, v], axis=1)[in_frame]

    final_valid = np.zeros(len(points_3d), dtype=bool)
    final_valid[np.where(valid)[0][in_frame]] = True

    return coords, final_valid


def generate_sam_masks(
    rendered_images: torch.Tensor,
    prompt_points_3d: torch.Tensor,
    cams: CameraState,
    args,
    device: torch.device,
    label: str = "object"
) -> SegmentationResult:
    """
    Run SAM3 image segmentation on each rendered view independently.
    
    Uses text prompt + projected 3D points to select the best mask per view.
    This avoids SAM2's video propagation which can lose track across views.
    
    Args:
        rendered_images: (N, H, W, 3) tensor of rendered views
        prompt_points_3d: (M, 3) tensor of 3D points inside the target object
        cams: Camera state with viewmats and Ks
        args: Config with num_views, width, height
        device: Torch device
        label: Text label for SAM3 text prompt (e.g., "truck", "car")
    
    Returns:
        SegmentationResult with masks and blended visualization images
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    num_views = len(rendered_images)
    H, W = args.height, args.width
    
    # Convert prompt points to numpy
    prompt_pts_np = prompt_points_3d.detach().cpu().numpy()
    
    print(f"[SAM3] Loading model...")
    model = build_sam3_image_model(device="cuda")
    processor = Sam3Processor(model)
    
    masks_list = []
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for view_idx in range(num_views):
            print(f"[SAM3] View {view_idx + 1}/{num_views}...")
            
            # Get image as numpy/PIL
            img_np = (rendered_images[view_idx].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Project 3D prompt points to this view
            vm = cams.viewmats[view_idx]
            K = cams.Ks[view_idx]
            add_coords, _ = project_points_to_image(prompt_pts_np, vm, K, W, H)
            
            # Run SAM3 text prompt
            state = processor.set_image(pil_img)
            
            if label.strip():
                output = processor.set_text_prompt(state=state, prompt=label)
                masks = output["masks"]    # (K, 1, H, W)
                scores = output["scores"]  # (K,)
                
                if scores.numel() == 0:
                    print(f"  [warn] no detections for '{label}' in view {view_idx}")
                    # Use empty mask for this view
                    masks_list.append(np.zeros((H, W), dtype=bool))
                    continue
                
                n_detections = masks.shape[0]
                print(f"  [info] detected {n_detections} '{label}' instances")
                
                # Select mask containing the most projected prompt points
                best = int(scores.argmax())  # default: highest score
                
                if len(add_coords) > 0:
                    masks_np = masks[:, 0].cpu().numpy()  # (K, H, W)
                    best_count = 0
                    best_score = -1
                    
                    for k in range(masks_np.shape[0]):
                        # Count how many prompt points fall inside this mask
                        points_inside = 0
                        for (u, v) in add_coords:
                            if 0 <= v < H and 0 <= u < W and masks_np[k, v, u]:
                                points_inside += 1
                        
                        # Prefer mask with more points; break ties by score
                        if points_inside > best_count or (points_inside == best_count and scores[k].item() > best_score):
                            best_count = points_inside
                            best_score = scores[k].item()
                            best = k
                    
                    if best_count > 0:
                        print(f"  [info] selected mask {best} (contains {best_count}/{len(add_coords)} pts)")
                    else:
                        print(f"  [warn] no mask contains projected points — using highest score (mask {best})")
                
                mask = masks[best, 0].cpu().numpy().astype(bool)
            else:
                # No label - use full image as mask
                mask = np.ones((H, W), dtype=bool)
            
            masks_list.append(mask)
    
    # Convert to numpy array
    target_masks = np.stack(masks_list, axis=0)
    
    # Generate blended images (green tint) for visualization
    original_renders = rendered_images.detach().cpu().numpy()
    blended_images = []
    
    for i in range(num_views):
        tinted = np.zeros((*original_renders.shape[1:3], 3), dtype=np.float32)
        tinted[..., 1] = 1.0  # Green tint
        alpha_map = (target_masks[i] * 0.5)[..., None].astype(np.float32)
        blended = original_renders[i] * (1.0 - alpha_map) + tinted * alpha_map
        blended_images.append(blended)
    
    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return SegmentationResult(
        masks=target_masks,
        blended_images=blended_images
    )