import numpy as np
from typing import Optional, Tuple


def sample_patch_depth(
    depth_map: np.ndarray,
    u: float,
    v: float,
    radius: int = 5,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    percentile: float = 50.0,
) -> Optional[float]:
    """
    Robustly sample depth around (u, v) pixel using a local patch median/percentile
    to filter out edge bleeding, sensor noise, or invalid NaN/0 depth pixels.

    Args:
        depth_map: 2D numpy array (H, W) in meters
        u: pixel column coordinate (horizontal)
        v: pixel row coordinate (vertical)
        radius: patch half-width in pixels
        min_depth: minimum valid depth in meters
        max_depth: maximum valid depth in meters
        percentile: percentile to select (50.0 for median)

    Returns:
        Filtered depth in meters, or None if no valid depth pixels in patch.
    """
    h, w = depth_map.shape[:2]
    u_int = int(round(u))
    v_int = int(round(v))

    u_min = max(0, u_int - radius)
    u_max = min(w, u_int + radius + 1)
    v_min = max(0, v_int - radius)
    v_max = min(h, v_int + radius + 1)

    patch = depth_map[v_min:v_max, u_min:u_max]
    
    # Filter valid range
    valid_mask = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth) & (patch > 0)
    valid_depths = patch[valid_mask]

    if len(valid_depths) == 0:
        # Fallback to broader patch if initial was empty
        broader_radius = radius * 3
        u_min = max(0, u_int - broader_radius)
        u_max = min(w, u_int + broader_radius + 1)
        v_min = max(0, v_int - broader_radius)
        v_max = min(h, v_int + broader_radius + 1)
        patch = depth_map[v_min:v_max, u_min:u_max]
        valid_mask = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth) & (patch > 0)
        valid_depths = patch[valid_mask]
        if len(valid_depths) == 0:
            return None

    return float(np.percentile(valid_depths, percentile))
