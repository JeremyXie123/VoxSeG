import os
import sys
import gc
import pathlib
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass

from core.camera import CameraState, project_points


@dataclass
class SegmentationResult:
    """Container for the output masks and visualization renders."""
    masks: np.ndarray              # (N_valid, H, W) bool
    blended_images: list[np.ndarray]  # N_valid RGB images
    valid_indices: np.ndarray      # (N_valid,) indices into original views


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
    Rejects views where the mask is empty or doesn't contain the center point.
    
    Args:
        rendered_images: (N, H, W, 3) tensor of rendered views
        prompt_points_3d: (M, 3) tensor of 3D points inside the target object
        cams: Camera state with viewmats and Ks
        args: Config with num_views, width, height
        device: Torch device
        label: Text label for SAM3 text prompt (e.g., "truck", "car")
    
    Returns:
        SegmentationResult with masks, blended images, and valid view indices
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    num_views = len(rendered_images)
    H, W = args.height, args.width
    
    # Convert prompt points to numpy
    prompt_pts_np = prompt_points_3d.detach().cpu().numpy()
    
    # Workaround: compute bpe_path directly to avoid pkg_resources issues
    # Use the model_builder module's location to find the sam3 assets
    model_builder = sys.modules.get("sam3.model_builder")
    if model_builder and hasattr(model_builder, "__file__") and model_builder.__file__:
        # model_builder is at sam3/model_builder.py, so go up one level
        sam3_root = pathlib.Path(model_builder.__file__).parent
        bpe_path = str(sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    else:
        # Fallback: use known location relative to this file
        bpe_path = str(pathlib.Path(__file__).parent.parent / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    
    print(f"[SAM3] Loading model...")
    model = build_sam3_image_model(bpe_path=bpe_path, device="cuda")
    threshold = getattr(args, "sam_threshold", 0.5)
    processor = Sam3Processor(model, confidence_threshold=threshold)
    print(f"[SAM3] Confidence threshold: {threshold}")
    
    masks_list = []
    valid_indices = []
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for view_idx in range(num_views):
            print(f"[SAM3] View {view_idx + 1}/{num_views}...", end=" ")
            
            # Get image as numpy/PIL
            img_np = (rendered_images[view_idx].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Project 3D prompt points to this view
            vm = cams.viewmats[view_idx]
            K = cams.Ks[view_idx]
            add_coords, _ = project_points_to_image(prompt_pts_np, vm, K, W, H)
            
            # Run SAM3 text prompt
            state = processor.set_image(pil_img)
            
            if not label.strip():
                # No label - use full image as mask (all views valid)
                mask = np.ones((H, W), dtype=bool)
                masks_list.append(mask)
                valid_indices.append(view_idx)
                print("full mask (no label)")
                continue
            
            output = processor.set_text_prompt(state=state, prompt=label)
            masks = output["masks"]    # (K, 1, H, W)
            scores = output["scores"]  # (K,)
            
            # Check if any detections
            if scores.numel() == 0:
                print(f"REJECTED - no '{label}' detected")
                continue
            
            n_detections = masks.shape[0]
            masks_np = masks[:, 0].cpu().numpy()  # (K, H, W)
            
            # Find mask containing the most projected prompt points
            best_mask_idx = None
            best_count = 0
            best_score = -1
            
            if len(add_coords) > 0:
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
                        best_mask_idx = k
            
            # Reject if no mask contains the center point
            if best_count == 0:
                print(f"REJECTED - {n_detections} detections but none contain center point")
                continue
            
            mask = masks_np[best_mask_idx].astype(bool)
            
            # Reject empty masks (shouldn't happen if best_count > 0, but safety check)
            if not mask.any():
                print(f"REJECTED - empty mask")
                continue
            
            masks_list.append(mask)
            valid_indices.append(view_idx)
            print(f"OK - mask {best_mask_idx} ({best_count} pts, score {best_score:.3f})")
    
    valid_indices = np.array(valid_indices, dtype=np.int64)
    num_valid = len(valid_indices)
    num_rejected = num_views - num_valid
    
    print(f"\n[SAM3] {num_valid}/{num_views} views accepted, {num_rejected} rejected")
    
    if num_valid == 0:
        raise RuntimeError("All views were rejected! Check that the label matches the object and the box is properly positioned.")
    
    # Convert to numpy array
    target_masks = np.stack(masks_list, axis=0)
    
    # Generate blended images (green tint) for visualization - only for valid views
    original_renders = rendered_images.detach().cpu().numpy()
    blended_images = []
    
    for i, view_idx in enumerate(valid_indices):
        tinted = np.zeros((H, W, 3), dtype=np.float32)
        tinted[..., 1] = 1.0  # Green tint
        alpha_map = (target_masks[i] * 0.5)[..., None].astype(np.float32)
        blended = original_renders[view_idx] * (1.0 - alpha_map) + tinted * alpha_map
        blended_images.append(blended)
    
    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return SegmentationResult(
        masks=target_masks,
        blended_images=blended_images,
        valid_indices=valid_indices
    )