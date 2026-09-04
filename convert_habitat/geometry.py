from libs import *

MIN_POSES = 4  # unused placeholder kept for parity with convert/geometry.py's constant


def yaw_delta_deg(yaw, previous_yaw):
    """Signed turn in degrees, wrapped to (-180, 180]. Copied unmodified from
    convert/geometry.py — pure trig on two scalars, no world-frame assumptions."""
    delta = yaw - previous_yaw
    return math.degrees(math.atan2(math.sin(delta), math.cos(delta)))


def project_to_pixel(camera_to_world, point_world, intrinsics):
    """Pixel (u, v) of a world point, or None if it is behind the camera or off-image.
    Copied unmodified from convert/episode.py — plain pinhole projection, agnostic to which
    physical axes are 'up': it only needs point_world expressed in the SAME world frame as
    camera_to_world's translation column, which habitat_pose_to_camera_to_world guarantees."""
    rotation_world_camera = camera_to_world[:3, :3]
    camera_origin = camera_to_world[:3, 3]
    x, y, z = rotation_world_camera.T.dot(point_world - camera_origin)
    if z <= 1e-6:
        return None

    pixel_u = int(round(intrinsics[0, 0] * x / z + intrinsics[0, 2]))
    pixel_v = int(round(intrinsics[1, 1] * y / z + intrinsics[1, 2]))
    from config import HABITAT_OUT_W, HABITAT_OUT_H
    if 0 <= pixel_u < HABITAT_OUT_W and 0 <= pixel_v < HABITAT_OUT_H:
        return pixel_u, pixel_v
    return None


def sensor_intrinsics(width, height, hfov_deg):
    """Pinhole intrinsics derived from the ACTUAL configured sensor spec — deliberately not
    hardcoded like vln_subgoal_pipeline/test/closed_loop/geometry.py's unproject_pixel (which
    bakes in 256x256/hfov=90), so this stays correct if HABITAT_OUT_W/H/HFOV ever change."""
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def habitat_pose_to_camera_to_world(position, rotation):
    """4x4 camera-to-world from a Habitat AgentState/SensorState's (position, rotation).

    Habitat's sensor-local frame looks down -Z with +Y up, +X right (verified against
    vln_subgoal_pipeline/test/closed_loop/geometry.py's unproject_pixel, and directly against
    the live sim: rgb sensor_state.rotation matches env.sim.get_agent_state().rotation exactly
    here since benchmark/nav/vln_r2r.yaml's rgb_sensor.orientation=[0,0,0], i.e. no relative
    tilt between agent and sensor). convert/'s camera_to_world convention has [:,0]=right,
    [:,1]=down, [:,2]=forward (OpenCV/pinhole), which project_to_pixel assumes. So: right is
    unchanged, forward = -local_Z_world_direction, down = -local_Y_world_direction. Negating
    two columns of a proper rotation matrix preserves its determinant (+1), so the result
    stays a valid rotation.
    """
    import quaternion  # noqa: F401 (registers the quaternion dtype/ops with numpy)

    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = rotation_matrix[:, 0]
    camera_to_world[:3, 1] = -rotation_matrix[:, 1]
    camera_to_world[:3, 2] = -rotation_matrix[:, 2]
    camera_to_world[:3, 3] = np.asarray(position, dtype=np.float64)
    return camera_to_world.astype(np.float32)


def ground_plane_yaw(camera_to_world):
    """Heading such that INCREASING yaw = turning LEFT, matching convert/episode.py's
    discretize_actions convention (LEFT if turn_deg > 0 else RIGHT). Derived from the world
    forward direction (column 2 of camera_to_world, per the convention above) projected onto
    Habitat's ground plane (X, Z; Y is up): forward = (-sin(yaw), *, -cos(yaw)) by
    construction at yaw=0 forward=(0,*,-1), and yaw increases as forward sweeps toward -X
    (the agent's own left, since right = column 0 = +X at identity)."""
    forward = camera_to_world[:3, 2]
    return math.atan2(-float(forward[0]), -float(forward[2]))


def region_aabb_bounds(region):
    lo, hi = region.aabb.min, region.aabb.max
    return np.array([lo.x, lo.y, lo.z]), np.array([hi.x, hi.y, hi.z])


def region_for_point(scene, point):
    """Manual 3D AABB containment over every region, smallest-volume wins on overlap, nearest-
    center as a fallback when the point is in zero regions (common: hallways are frequently
    not their own MP3D region). Manual because SemanticRegion.contains()/
    get_regions_for_point() don't work on these .house files — verified directly: every
    region's floor_height/extrusion_height reads back as 0.0 and poly_loop_points is empty,
    so both return False/[] even for a region's own AABB center. region.aabb IS populated
    correctly (sane room-sized boxes, confirmed against the live sim)."""
    point = np.asarray(point, dtype=np.float64)
    regions = list(scene.regions)
    if not regions:
        return None

    containing = []
    for region in regions:
        lo, hi = region_aabb_bounds(region)
        if np.all(point >= lo) and np.all(point <= hi):
            volume = float(np.prod(np.maximum(hi - lo, 1e-9)))
            containing.append((volume, region))
    if containing:
        containing.sort(key=lambda pair: pair[0])
        return containing[0][1]

    def center_distance(region):
        lo, hi = region_aabb_bounds(region)
        return float(np.linalg.norm((lo + hi) / 2.0 - point))

    return min(regions, key=center_distance)


def instance_pixel_stats(semantic_frame, instance_id):
    """(pixel_count, row_mean, col_mean) for one instance id in a (H, W) semantic frame, or
    None if the instance doesn't appear. row/col are the on-screen mask centroid — not a
    reprojected 3D center, which can land off-image/occluded even when the object genuinely
    is visible elsewhere in frame (e.g. a long table whose center leg is out of view)."""
    mask = semantic_frame == instance_id
    count = int(mask.sum())
    if count == 0:
        return None
    rows, cols = np.nonzero(mask)
    return count, float(rows.mean()), float(cols.mean())


def to_norm_yx(pixel_uv, width, height):
    """(u, v) pixel -> [y, x] normalized to a 0-1000 grid, matching
    Cosmos3Reasoner.DETECT_PROMPT_TEMPLATE's response convention (verified against
    vln_subgoal_pipeline/test/closed_loop/geometry.py's parse_pixel_target)."""
    u, v = pixel_uv
    y_norm = int(round((v / height) * 1000))
    x_norm = int(round((u / width) * 1000))
    return [max(0, min(999, y_norm)), max(0, min(999, x_norm))]
