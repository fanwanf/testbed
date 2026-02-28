"""
cvTools_isaacgym.py - Convex hull candidate point detection using PyTorch

Fully vectorized implementation supporting multi-environment parallel processing,
without for loops over environments.
Uses boundary detection and corner detection to approximate convex hull vertices.
"""

import torch
import torch.nn.functional as F
import numpy as np


def find_boundary_points_vectorized(mask: torch.Tensor) -> torch.Tensor:
    """
    Find boundary points of a mask (vectorized implementation)

    Args:
        mask: [N, H, W] binary mask

    Returns:
        boundary: [N, H, W] boundary mask (1 for boundary points, 0 otherwise)
    """
    # Use morphological operation: boundary = mask - erode(mask)
    # Add channel dimension
    mask_4d = mask.unsqueeze(1).float()  # [N, 1, H, W]

    # Erosion operation (inverse of min-value pooling)
    # Use -max_pool(-x) to implement min_pool
    eroded = -F.max_pool2d(-mask_4d, kernel_size=3, stride=1, padding=1)

    # Boundary = original mask - eroded mask
    boundary = (mask_4d - eroded).squeeze(1)
    boundary = (boundary > 0).float()

    return boundary


def compute_curvature_vectorized(boundary: torch.Tensor) -> torch.Tensor:
    """
    Compute curvature of boundary points (for corner detection)

    Args:
        boundary: [N, H, W] boundary mask

    Returns:
        curvature: [N, H, W] curvature map (0 for non-boundary points)
    """
    device = boundary.device

    # Compute Sobel gradients to estimate boundary direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=torch.float32).view(1, 1, 3, 3)

    boundary_4d = boundary.unsqueeze(1)  # [N, 1, H, W]

    # Compute gradients
    grad_x = F.conv2d(boundary_4d, sobel_x, padding=1)
    grad_y = F.conv2d(boundary_4d, sobel_y, padding=1)

    # Compute direction angle
    angle = torch.atan2(grad_y, grad_x)  # [N, 1, H, W]

    # Use Laplacian to approximate curvature (second derivative of direction angle)
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                    device=device, dtype=torch.float32).view(1, 1, 3, 3)

    # Apply Laplacian to the angle map
    curvature = torch.abs(F.conv2d(angle, laplacian_kernel, padding=1))
    curvature = curvature.squeeze(1)  # [N, H, W]

    # Keep curvature only at boundary points
    curvature = curvature * boundary

    return curvature


def find_corner_points_vectorized(mask: torch.Tensor, threshold_ratio: float = 0.3) -> torch.Tensor:
    """
    Find corner points (convex vertex candidates) of a mask - fully vectorized

    Args:
        mask: [N, H, W] binary mask
        threshold_ratio: curvature threshold ratio (relative to maximum curvature)

    Returns:
        corners: [N, H, W] corner point mask
    """
    N, H, W = mask.shape
    device = mask.device

    # Find boundary
    boundary = find_boundary_points_vectorized(mask)

    # Compute curvature
    curvature = compute_curvature_vectorized(boundary)

    # Non-maximum suppression (in 3x3 window)
    max_pool = F.max_pool2d(curvature.unsqueeze(1), kernel_size=3, stride=1, padding=1)
    is_local_max = (curvature.unsqueeze(1) == max_pool) & (curvature.unsqueeze(1) > 0)
    is_local_max = is_local_max.squeeze(1)

    # Use adaptive threshold - vectorized computation
    curvature_flat = curvature.view(N, -1)  # [N, H*W]
    max_curvature = curvature_flat.max(dim=1, keepdim=True)[0]  # [N, 1]
    threshold = (max_curvature * threshold_ratio).view(N, 1, 1)  # [N, 1, 1]

    # Corners = local maxima & curvature exceeds threshold & on boundary
    corners = is_local_max & (curvature > threshold) & (boundary > 0)

    return corners.float()


def find_all_corners_vectorized(height_map: torch.Tensor,
                                mask: torch.Tensor,
                                height_resolution: float = 0.01,
                                num_height_levels: int = 20) -> torch.Tensor:
    """
    Find corners across multiple height levels - fully vectorized (no environment loops)

    Splits the height map into multiple levels and processes them simultaneously
    using vectorized operations.

    Args:
        height_map: [N, H, W] height map (posZ)
        mask: [N, H, W] valid position mask
        height_resolution: height resolution
        num_height_levels: number of height levels to process

    Returns:
        corner_weights: [N, H, W] corner weights (considering all height levels)
    """
    device = height_map.device
    N, H, W = height_map.shape

    # Discretize heights
    height_int = (height_map / height_resolution).long()
    height_int = torch.where(mask > 0, height_int, torch.tensor(-1, device=device, dtype=torch.long))

    # Find height range
    valid_heights = height_int[height_int >= 0]
    if len(valid_heights) == 0:
        return torch.zeros_like(mask)

    min_h = valid_heights.min().item()
    max_h = valid_heights.max().item()

    # Limit number of height levels to process
    step = max(1, (max_h - min_h + 1) // num_height_levels)
    height_levels = torch.arange(min_h, max_h + 1, step, device=device)
    num_levels = len(height_levels)

    # Create mask for each height level: [N, num_levels, H, W]
    height_int_exp = height_int.unsqueeze(1)  # [N, 1, H, W]
    levels_exp = height_levels.view(1, num_levels, 1, 1)  # [1, num_levels, 1, 1]

    # Each height level includes points at that height and nearby
    level_masks = ((height_int_exp >= levels_exp) &
                   (height_int_exp < levels_exp + step)).float()  # [N, num_levels, H, W]

    # Reshape to [N * num_levels, H, W] for batch corner detection
    level_masks_flat = level_masks.view(N * num_levels, H, W)

    # Batch corner detection
    corners_flat = find_corner_points_vectorized(level_masks_flat, threshold_ratio=0.2)

    # Reshape back to [N, num_levels, H, W]
    corners = corners_flat.view(N, num_levels, H, W)

    # Merge corners from all height levels (take max)
    all_corners = corners.max(dim=1)[0]  # [N, H, W]

    # Also add boundary points as candidates
    boundary = find_boundary_points_vectorized(mask)

    # Merge: corner weight 1.0, boundary point weight 0.5
    corner_weights = torch.maximum(all_corners, boundary * 0.5)

    return corner_weights


def extract_topk_candidates_vectorized(corner_weights: torch.Tensor,
                                       height_map: torch.Tensor,
                                       valid_mask: torch.Tensor,
                                       k: int) -> tuple:
    """
    Extract top-k candidate points from corner weight map - fully vectorized

    Args:
        corner_weights: [N, H, W] corner/boundary point weights
        height_map: [N, H, W] height map
        valid_mask: [N, H, W] valid position mask
        k: number of points to extract per image

    Returns:
        coords_x: [N, k] X coordinates
        coords_y: [N, k] Y coordinates
        heights: [N, k] height values
        valid: [N, k] validity flags
    """
    device = corner_weights.device
    N, H, W = corner_weights.shape

    # Only consider corner points at valid positions
    weights = corner_weights * valid_mask

    # Flatten
    weights_flat = weights.view(N, -1)  # [N, H*W]
    height_flat = height_map.view(N, -1)  # [N, H*W]
    valid_flat = valid_mask.view(N, -1)  # [N, H*W]

    # Compute composite score: higher corner weight + lower height -> higher score
    # Use normalized height as penalty term
    height_max = height_flat.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
    height_normalized = height_flat / height_max

    score = weights_flat - height_normalized * 0.1
    score = torch.where(weights_flat > 0, score, torch.tensor(-1e9, device=device))

    # Get top-k
    actual_k = min(k, H * W)
    top_scores, top_indices = torch.topk(score, k=actual_k, dim=1)  # [N, k]

    # Convert to 2D coordinates
    coords_y = top_indices // W  # [N, k]
    coords_x = top_indices % W   # [N, k]

    # Use gather to get height values
    heights = torch.gather(height_flat, 1, top_indices)  # [N, k]

    # Validity: weight > 0 and within valid mask
    valid = (torch.gather(weights_flat, 1, top_indices) > 0) & \
            (torch.gather(valid_flat, 1, top_indices) > 0)

    # If k > actual_k, pad
    if k > actual_k:
        pad_size = k - actual_k
        coords_x = F.pad(coords_x, (0, pad_size), value=0)
        coords_y = F.pad(coords_y, (0, pad_size), value=0)
        heights = F.pad(heights, (0, pad_size), value=0)
        valid = F.pad(valid, (0, pad_size), value=False)

    return coords_x, coords_y, heights, valid


def getConvexHullActionsVectorized(posZValid: torch.Tensor,
                                   mask: torch.Tensor,
                                   heightResolution: float,
                                   selectedAction: int,
                                   bin_height: float,
                                   device: str = 'cuda:0') -> torch.Tensor:
    """
    Get convex hull candidate action points - fully vectorized (no for loops over environments)

    Args:
        posZValid: [num_envs, rotNum, rangeX_A, rangeY_A] height map
        mask: [num_envs, rotNum, rangeX_A, rangeY_A] mask of possible placement positions
        heightResolution: height resolution
        selectedAction: number of candidate points to select per environment
        bin_height: container height, used to fill invalid candidate points
        device: PyTorch device

    Returns:
        candidates: [num_envs, selectedAction, 5] candidate points
                   5 columns are [ROT, X, Y, H, V]
    """
    # Ensure inputs are tensors
    if isinstance(posZValid, np.ndarray):
        posZValid = torch.tensor(posZValid, device=device, dtype=torch.float32)
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, device=device, dtype=torch.float32)

    posZValid = posZValid.to(device)
    mask = mask.to(device)

    num_envs = posZValid.shape[0]
    rotNum = posZValid.shape[1]
    rangeX_A = posZValid.shape[2]
    rangeY_A = posZValid.shape[3]

    # Reshape [num_envs, rotNum, H, W] to [num_envs * rotNum, H, W]
    N = num_envs * rotNum
    posZ_flat = posZValid.view(N, rangeX_A, rangeY_A)
    mask_flat = mask.view(N, rangeX_A, rangeY_A)

    # Find corners across all height levels - fully vectorized
    corner_weights = find_all_corners_vectorized(
        posZ_flat, mask_flat, heightResolution, num_height_levels=10
    )

    # Number of candidate points to extract per (env, rot)
    k_per_rot = max(selectedAction // rotNum + 2, 5)

    # Extract top-k candidate points - fully vectorized
    coords_x, coords_y, heights, valid = extract_topk_candidates_vectorized(
        corner_weights, posZ_flat, mask_flat, k=k_per_rot
    )
    # coords_x, coords_y, heights, valid: [N, k_per_rot]

    # Reshape back to [num_envs, rotNum, k_per_rot]
    coords_x = coords_x.view(num_envs, rotNum, k_per_rot)
    coords_y = coords_y.view(num_envs, rotNum, k_per_rot)
    heights = heights.view(num_envs, rotNum, k_per_rot)
    valid = valid.view(num_envs, rotNum, k_per_rot)

    # Get V values (from original mask) - vectorized using advanced indexing
    env_idx = torch.arange(num_envs, device=device).view(num_envs, 1, 1).expand(-1, rotNum, k_per_rot)
    rot_idx = torch.arange(rotNum, device=device).view(1, rotNum, 1).expand(num_envs, -1, k_per_rot)
    V = mask[env_idx, rot_idx, coords_y, coords_x]  # [num_envs, rotNum, k_per_rot]

    # Create ROT indices
    rot_values = torch.arange(rotNum, device=device, dtype=torch.float32).view(1, rotNum, 1)
    rot_values = rot_values.expand(num_envs, -1, k_per_rot)  # [num_envs, rotNum, k_per_rot]

    # Combine candidates: [num_envs, rotNum, k_per_rot, 5] -> [num_envs, rotNum * k_per_rot, 5]
    # Note: consistent with original - column 1 is row index (Y) as lx, column 2 is col index (X) as ly
    # In original: row index -> X coordinate, col index -> Y coordinate (opposite of conventional)
    candidates_full = torch.stack([
        rot_values,                    # ROT
        coords_y.float(),              # row index -> lx (physical X)
        coords_x.float(),              # col index -> ly (physical Y)
        heights,                       # H
        V                              # V
    ], dim=-1)  # [num_envs, rotNum, k_per_rot, 5]

    candidates_full = candidates_full.view(num_envs, rotNum * k_per_rot, 5)
    valid_full = valid.view(num_envs, rotNum * k_per_rot)

    # Sort by height, select the lowest selectedAction valid candidate points
    # Set invalid candidates' height to large value so they sort to the end
    heights_for_sort = candidates_full[:, :, 3].clone()
    heights_for_sort = torch.where(valid_full, heights_for_sort,
                                   torch.tensor(1e9, device=device))

    # Sort
    sorted_indices = torch.argsort(heights_for_sort, dim=1)  # [num_envs, rotNum * k_per_rot]

    # Use gather to select sorted candidates
    sorted_indices_exp = sorted_indices.unsqueeze(-1).expand(-1, -1, 5)
    candidates_sorted = torch.gather(candidates_full, 1, sorted_indices_exp)

    # Select the first selectedAction candidates
    output = candidates_sorted[:, :selectedAction, :]  # [num_envs, selectedAction, 5]

    # Handle case where there are not enough valid candidate points
    # Check number of valid candidates per environment
    valid_sorted = torch.gather(valid_full, 1, sorted_indices)
    valid_counts = valid_sorted[:, :selectedAction].sum(dim=1)  # [num_envs]

    # For environments without enough valid candidates, use fallback
    # Fallback: select the selectedAction positions with lowest posZ
    needs_fallback = valid_counts < selectedAction

    if needs_fallback.any():
        # Flatten posZ and mask
        posZ_all = posZValid.view(num_envs, -1)  # [num_envs, rotNum * H * W]
        mask_all = mask.view(num_envs, -1)
        total_size = rotNum * rangeX_A * rangeY_A

        # Set invalid positions' height to large value (ensure they sort to the end)
        # Also filter out posZ >= 100 (these are initialization values or invalid)
        valid_posZ = (mask_all > 0) & (posZ_all < 100)
        posZ_masked = torch.where(valid_posZ, posZ_all,
                                  torch.tensor(1e9, device=device))

        # Sort to find lowest positions
        fallback_indices = torch.argsort(posZ_masked, dim=1)[:, :selectedAction]

        # Convert indices to ROT, X, Y
        fallback_rot = fallback_indices // (rangeX_A * rangeY_A)
        remainder = fallback_indices % (rangeX_A * rangeY_A)
        fallback_x = remainder // rangeY_A
        fallback_y = remainder % rangeY_A

        # Get heights and V
        fallback_h = torch.gather(posZ_all, 1, fallback_indices)
        fallback_v = torch.gather(mask_all, 1, fallback_indices)

        # Set invalid positions' height to bin_height (not 1e3)
        fallback_h = torch.where(fallback_h < 100, fallback_h,
                                 torch.tensor(bin_height, device=device))

        # Build fallback candidates with same coordinate mapping as normal candidates
        fallback_candidates = torch.stack([
            fallback_rot.float(),
            fallback_x.float(),  # lx
            fallback_y.float(),  # ly
            fallback_h,
            fallback_v
        ], dim=-1)  # [num_envs, selectedAction, 5]

        # Use where to mix original candidates and fallback candidates
        needs_fallback_exp = needs_fallback.view(num_envs, 1, 1).expand(-1, selectedAction, 5)
        output = torch.where(needs_fallback_exp, fallback_candidates, output)

    return output


def getConvexHullActions(posZValid, mask, heightResolution, selectedAction, bin_height, device='cuda:0'):
    """
    Get convex hull candidate action points (compatibility interface)

    Args:
        posZValid: [num_envs, rotNum, rangeX_A, rangeY_A] height map
        mask: [num_envs, rotNum, rangeX_A, rangeY_A] mask of possible placement positions
        heightResolution: height resolution
        selectedAction: number of candidate points to select per environment
        bin_height: container height
        device: PyTorch device

    Returns:
        candidates: [num_envs, selectedAction, 5] candidate point tensor
    """
    return getConvexHullActionsVectorized(
        posZValid, mask, heightResolution, selectedAction, bin_height, device
    )


