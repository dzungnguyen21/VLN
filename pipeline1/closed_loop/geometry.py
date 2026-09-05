import re

import numpy as np

from .config import (
    ANTI_BACKTRACK_TOLERANCE_DEG,
    COLLISION_CLEAR,
    COLLISION_NEAR,
    DEPTH_MAX,
    DEPTH_MIN,
    SEARCH_PLAN_HEADING_DEG,
    SIM_TURN_ANGLE,
    TURN_LEFT,
    TURN_RIGHT,
)


def unproject_pixel(x, y, depth_val, sensor_state):
    import quaternion
    WIDTH, HEIGHT = 256, 256
    HFOV = np.pi / 2.0
    fx = (WIDTH / 2.0) / np.tan(HFOV / 2.0)
    fy = fx
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    Z = depth_val * 10.0
    X_cam = (x - cx) * Z / fx
    Y_cam = -(y - cy) * Z / fy
    Z_cam = -Z
    local_pt = np.array([X_cam, Y_cam, Z_cam])

    pos = np.array(sensor_state.position)
    rot_matrix = quaternion.as_rotation_matrix(sensor_state.rotation)
    return pos + rot_matrix @ local_pt


def horiz_dist(agent_pos, target_pt):
    """
    Floor-plane (X/Z) distance only -- Habitat is Y-up, and a grounded landmark
    pixel (e.g. a tabletop) usually sits well above floor height. A full 3D
    distance would keep the vertical offset baked in forever, so the agent
    could stand right against the table and still never register "reached".
    "Reached the landmark" means horizontally close to it, not the same height.
    """
    return float(np.linalg.norm([agent_pos[0] - target_pt[0], agent_pos[2] - target_pt[2]]))


def parse_pixel_target(landmark_str, img_shape):
    match = re.search(r"\[(\d+),\s*(\d+)\]", landmark_str)
    if not match:
        return None
    y_norm, x_norm = int(match.group(1)), int(match.group(2))
    # Model-reported norms are nominally [0, 1000] but aren't guaranteed to stay
    # in range (e.g. exactly 1000, or a stray out-of-range value) -- clamp so the
    # pixel indices always stay valid for a img_shape[0] x img_shape[1] image.
    y_pixel = min(max(int((y_norm / 1000.0) * img_shape[0]), 0), img_shape[0] - 1)
    x_pixel = min(max(int((x_norm / 1000.0) * img_shape[1]), 0), img_shape[1] - 1)
    return x_pixel, y_pixel


def valid_depth(depth_img, x_pixel, y_pixel):
    d_val = depth_img[y_pixel, x_pixel, 0]
    if DEPTH_MIN < d_val < DEPTH_MAX / 10.0:
        return d_val
    return None


def heading_clearance(depth_img):
    """
    0-1 score for how physically open the path straight ahead is -- from the
    depth image's central forward band, the region the agent will actually
    walk through if it commits to this heading and steps forward. This is
    independent of where the model's pointed-to pixel lands (that pixel might
    be off to one side, e.g. a doorway across the room, while a wall sits
    directly in front). 0 = blocked immediately ahead (or no usable depth at
    all in that band), 1 = comfortably clear.
    """
    h, w = depth_img.shape[:2]
    band = depth_img[h // 2:, w // 3: 2 * w // 3, 0]
    valid = band[band > 0]
    if valid.size == 0:
        return 0.0
    near = float(np.min(valid))
    return float(np.clip((near - COLLISION_NEAR) / (COLLISION_CLEAR - COLLISION_NEAR), 0.0, 1.0))


def turns_to_heading(from_heading_idx, to_heading_idx):
    """(turn_action, turn_count) of SIM_TURN_ANGLE physical turns to reorient
    from one SEARCH_PLAN heading index to another, shortest direction."""
    delta = SEARCH_PLAN_HEADING_DEG[to_heading_idx] - SEARCH_PLAN_HEADING_DEG[from_heading_idx]
    delta = (delta + 180) % 360 - 180   # normalize to (-180, 180]
    turns = round(abs(delta) / SIM_TURN_ANGLE)
    if turns == 0:
        return None, 0
    return (TURN_LEFT if delta > 0 else TURN_RIGHT), turns


def is_backtracking(candidate_deg, last_explore_deg):
    """True if candidate_deg is close to the exact opposite of last_explore_deg --
    i.e. would send the agent back the way it just came, the oscillation pattern
    that leaves it stuck ping-ponging between the same two spots while exploring."""
    if last_explore_deg is None:
        return False
    reverse_deg = (last_explore_deg + 180) % 360
    diff = abs(candidate_deg - reverse_deg) % 360
    diff = min(diff, 360 - diff)
    return diff < ANTI_BACKTRACK_TOLERANCE_DEG
