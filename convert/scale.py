from libs import *
from geometry import index_nearest_time, quaternion_to_rotation_matrix

import cv2


DOWNSCALE = 2
FRAME_GAP = 2
FRAME_STEP = 3
HIGHPASS_SIGMA = 4.0
MIN_SHARPNESS = 0.15
GRID_FORWARD, GRID_LATERAL = 120, 80
FORWARD_RANGE = (0.8, 3.2)
LATERAL_RANGE = (-0.75, 0.75)
HEIGHT_CANDIDATES = 90
HEIGHT_RANGE = (0.25, 6.0)
MIN_OVERLAP = 0.3
MIN_STRONG_PAIRS = 5


def high_passed_frame(bag, image_times_ns, image_row_ids, pose_time_ns, tolerance_ns):
    image_index = index_nearest_time(image_times_ns, pose_time_ns)
    if abs(int(image_times_ns[image_index]) - pose_time_ns) > tolerance_ns:
        return None

    grey = np.array(bag.image(image_row_ids[image_index]).convert("L"), np.float32)
    if DOWNSCALE > 1:
        grey = cv2.resize(
            grey,
            (grey.shape[1] // DOWNSCALE, grey.shape[0] // DOWNSCALE),
            interpolation=cv2.INTER_AREA,
        )
    return grey - cv2.GaussianBlur(grey, (0, 0), HIGHPASS_SIGMA)


def sample_grid(image, points_camera, intrinsics):
    depth = points_camera[:, 2]
    visible = depth > 1e-3
    safe_depth = np.maximum(depth, 1e-9)
    pixels_u = np.where(visible,
                        intrinsics[0, 0] * points_camera[:, 0] / safe_depth
                        + intrinsics[0, 2], -1.0)
    pixels_v = np.where(visible,
                        intrinsics[1, 1] * points_camera[:, 1] / safe_depth
                        + intrinsics[1, 2], -1.0)

    height, width = image.shape
    visible &= ((pixels_u >= 1) & (pixels_u < width - 2)
                & (pixels_v >= 1) & (pixels_v < height - 2))
    values = cv2.remap(
        image,
        pixels_u.astype(np.float32).reshape(-1, 1),
        pixels_v.astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR, borderValue=0,
    ).ravel()
    return values, visible


def grid_match_cost(first_image, second_image, grid_in_first, grid_in_second, intrinsics):
    first_samples, first_visible = sample_grid(first_image, grid_in_first, intrinsics)
    second_samples, second_visible = sample_grid(second_image, grid_in_second, intrinsics)
    both_visible = first_visible & second_visible
    if both_visible.sum() < MIN_OVERLAP * len(grid_in_first):
        return np.nan

    first_centered = first_samples[both_visible] - first_samples[both_visible].mean()
    second_centered = second_samples[both_visible] - second_samples[both_visible].mean()
    denominator = math.sqrt(float((first_centered * first_centered).sum()
                                  * (second_centered * second_centered).sum())) + 1e-9
    return 1.0 - float((first_centered * second_centered).sum()) / denominator


def measure_height(
    bag,
    image_times_ns, image_row_ids,
    pose_times_ns, positions, quaternions,
    up,
    intrinsics,
    tolerance_ms,
):
    intrinsics_scaled = intrinsics / DOWNSCALE
    intrinsics_scaled[2, 2] = 1.0
    tolerance_ns = int(tolerance_ms * 1e6)
    rotations = [quaternion_to_rotation_matrix(quaternion) for quaternion in quaternions]

    ahead, across = np.meshgrid(
        np.linspace(*FORWARD_RANGE, GRID_FORWARD),
        np.linspace(*LATERAL_RANGE, GRID_LATERAL),
        indexing="ij",
    )
    ahead, across = ahead.ravel()[:, None], across.ravel()[:, None]
    height_candidates = np.exp(np.linspace(
        math.log(HEIGHT_RANGE[0]), math.log(HEIGHT_RANGE[1]), HEIGHT_CANDIDATES))

    estimates, n_pairs = [], 0
    for first in range(0, len(pose_times_ns) - FRAME_GAP - 1, FRAME_STEP):
        second = first + FRAME_GAP
        first_image = high_passed_frame(bag, image_times_ns, image_row_ids,
                                        int(pose_times_ns[first]), tolerance_ns)
        second_image = high_passed_frame(bag, image_times_ns, image_row_ids,
                                         int(pose_times_ns[second]), tolerance_ns)
        if first_image is None or second_image is None:
            continue
        n_pairs += 1

        floor_normal = rotations[first].T.dot(up)
        forward = np.array([0.0, 0.0, 1.0]) - float(floor_normal[2]) * floor_normal
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            continue
        forward /= forward_norm
        right = np.cross(forward, floor_normal)

        rotation_second_from_first = rotations[second].T.dot(rotations[first])
        translation_second_from_first = rotations[second].T.dot(positions[first]
                                                                - positions[second])

        costs = np.full(len(height_candidates), np.nan)
        for candidate_index, height in enumerate(height_candidates):
            grid_in_first = ((-height * floor_normal)[None, :]
                             + (ahead * height) * forward
                             + (across * height) * right)
            grid_in_second = (grid_in_first.dot(rotation_second_from_first.T)
                              + translation_second_from_first)
            costs[candidate_index] = grid_match_cost(first_image, second_image, grid_in_first,
                                                     grid_in_second, intrinsics_scaled)

        if np.all(np.isnan(costs)):
            continue
        best_index = int(np.nanargmin(costs))
        if best_index in (0, len(height_candidates) - 1):
            continue

        sharpness = float(np.nanmedian(costs) - costs[best_index])
        estimates.append((float(height_candidates[best_index]), sharpness))

    strong_estimates = [estimate for estimate in estimates if estimate[1] >= MIN_SHARPNESS]
    if len(strong_estimates) < MIN_STRONG_PAIRS:
        raise ValueError(
            f"auto-scale failed: only {len(strong_estimates)} of {n_pairs} frame pairs had a "
            "clear minimum. Usually a floor with no texture, or a camera that cannot see it. "
            "Pin this bag in SCALE_OVERRIDES in .env instead: {'path_length': "
            "<real metres walked>} or {'traj_scale': <factor>}.")

    heights = np.array([height for height, _ in strong_estimates])
    median_height = float(np.median(heights))
    spread = float(np.median(np.abs(heights - median_height))) / max(median_height, 1e-9)
    return median_height, spread
