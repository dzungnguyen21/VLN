from libs import *


MIN_POSES = 4


def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def up_direction_from_pitch(pitch_deg):
    pitch_rad = math.radians(pitch_deg)
    return np.array([0.0, -math.cos(pitch_rad), -math.sin(pitch_rad)])


def synthesize_camera_pose(x, y, yaw, height_m, pitch_deg):
    pitch_rad = math.radians(pitch_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ])

    optical_forward = yaw_rotation.dot([math.cos(pitch_rad), 0.0, -math.sin(pitch_rad)])
    optical_right = yaw_rotation.dot([0.0, -1.0, 0.0])
    optical_down = np.cross(optical_forward, optical_right)

    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 0] = optical_right
    camera_to_world[:3, 1] = optical_down
    camera_to_world[:3, 2] = optical_forward
    camera_to_world[:3, 3] = (x, y, height_m)
    return camera_to_world


def center_crop_box(source_width, source_height, out_width, out_height):
    source_aspect = source_width / source_height
    out_aspect = out_width / out_height
    if abs(source_aspect - out_aspect) < 1e-6:
        return None

    if source_aspect > out_aspect:
        crop_width = int(round(source_height * out_width / out_height))
        left = (source_width - crop_width) // 2
        return (left, 0, left + crop_width, source_height)

    crop_height = int(round(source_width * out_height / out_width))
    top = (source_height - crop_height) // 2
    return (0, top, source_width, top + crop_height)


def crop_resize_intrinsics(
    intrinsics, crop_box,
    source_width, source_height,
    out_width, out_height,
):
    intrinsics_out = np.asarray(intrinsics, np.float64).copy()
    if crop_box is not None:
        left, top, right, bottom = crop_box
        intrinsics_out[0, 2] -= left
        intrinsics_out[1, 2] -= top
        source_width, source_height = right - left, bottom - top

    intrinsics_out[0, :] *= out_width / source_width
    intrinsics_out[1, :] *= out_height / source_height
    return intrinsics_out


def index_nearest_time(timestamps_ns, target_ns):
    insert_at = int(np.searchsorted(timestamps_ns, target_ns))
    if insert_at == 0:
        return 0
    if insert_at >= len(timestamps_ns):
        return len(timestamps_ns) - 1

    distance_after = abs(timestamps_ns[insert_at] - target_ns)
    distance_before = abs(timestamps_ns[insert_at - 1] - target_ns)
    return insert_at if distance_after < distance_before else insert_at - 1


def resample_indices(timestamps_ns, target_hz, skip_if_already_slower=False):
    n_samples = len(timestamps_ns)
    if target_hz <= 0 or n_samples < 2:
        return np.arange(n_samples)

    period_ns = 1e9 / target_hz
    median_gap_ns = float(np.median(np.diff(timestamps_ns)))
    if skip_if_already_slower and period_ns <= median_gap_ns:
        return np.arange(n_samples)

    span_ns = int(timestamps_ns[-1] - timestamps_ns[0])
    tick_times_ns = timestamps_ns[0] + period_ns * np.arange(int(span_ns / period_ns) + 1)

    index_after = np.clip(np.searchsorted(timestamps_ns, tick_times_ns), 1, n_samples - 1)
    index_before = index_after - 1
    before_is_nearer = (np.abs(timestamps_ns[index_before] - tick_times_ns)
                        <= np.abs(timestamps_ns[index_after] - tick_times_ns))

    return np.unique(np.where(before_is_nearer, index_before, index_after))


def load_trajectory(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.replace(",", " ").split())

    if len(rows) < MIN_POSES:
        raise ValueError(f"{path} holds {len(rows)} pose(s), need at least {MIN_POSES}")
    if any(len(row) < 8 for row in rows):
        raise ValueError(f"{path}: every line needs 8 columns (t_ns x y z qx qy qz qw)")

    timestamps_ns = np.array([int(float(row[0])) for row in rows], np.int64)
    positions = np.array([[float(value) for value in row[1:4]] for row in rows], np.float64)
    quaternions = np.array([[float(value) for value in row[4:8]] for row in rows], np.float64)
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)

    if np.any(np.diff(timestamps_ns) <= 0):
        order = np.argsort(timestamps_ns, kind="stable")
        timestamps_ns = timestamps_ns[order]
        positions = positions[order]
        quaternions = quaternions[order]
    return timestamps_ns, positions, quaternions


def path_length(positions):
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def world_from_trajectory(positions, quaternions, up, height_m, scale):
    up_unit = up / np.linalg.norm(up)

    drift = np.diff(positions, axis=0).sum(0)
    forward = drift - np.dot(drift, up_unit) * up_unit
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        forward = np.array([0.0, 0.0, 1.0]) - up_unit[2] * up_unit
        forward_norm = np.linalg.norm(forward)
    forward = forward / forward_norm

    rotation_world_from_first = np.stack([forward, np.cross(up_unit, forward), up_unit])

    n_poses = len(positions)
    camera_to_world = np.zeros((n_poses, 4, 4), np.float64)
    camera_to_world[:, 3, 3] = 1.0
    camera_to_world[:, :3, 3] = (rotation_world_from_first.dot((scale * positions).T).T
                                 + np.array([0.0, 0.0, height_m]))

    yaws = np.zeros(n_poses)
    for index in range(n_poses):
        rotation_world_camera = rotation_world_from_first.dot(
            quaternion_to_rotation_matrix(quaternions[index]))
        camera_to_world[index, :3, :3] = rotation_world_camera

        optical_axis_world = rotation_world_camera[:, 2]
        if math.hypot(optical_axis_world[0], optical_axis_world[1]) < 1e-6:
            optical_axis_world = -rotation_world_camera[:, 1]
        yaws[index] = math.atan2(optical_axis_world[1], optical_axis_world[0])

    return camera_to_world, yaws
