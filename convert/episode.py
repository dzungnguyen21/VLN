from libs import *
from config import *
from geometry import (center_crop_box, crop_resize_intrinsics, index_nearest_time,
                      load_trajectory, path_length, resample_indices, synthesize_camera_pose,
                      up_direction_from_pitch, world_from_trajectory)

MIN_GOAL_K = 3
CHUNK_SIZE = 1000
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3


def require_rig():
    """Camera height (cm) and downward tilt (deg), measured by hand."""
    if RIG_HEIGHT_CM is None or RIG_PITCH_DEG is None:
        raise ValueError("set RIG_HEIGHT_CM and RIG_PITCH_DEG in .env — measure the camera's "
                         "optical centre above the floor and its downward tilt. They cannot "
                         "be recovered from these bags.")
    return float(RIG_HEIGHT_CM), float(RIG_PITCH_DEG)


def setting_tag():
    """The <H>cm_<P>deg tag naming every label column and image directory."""
    height_cm, pitch_deg = require_rig()
    return f"{round(height_cm)}cm_{round(pitch_deg)}deg"


def yaw_delta_deg(yaw, previous_yaw):
    """Signed turn in degrees, wrapped to (-180, 180]."""
    delta = yaw - previous_yaw
    return math.degrees(math.atan2(math.sin(delta), math.cos(delta)))


def resolve_scale(raw_path_length, pinned_traj_scale, pinned_path_m,
                  measured_height=None, rig_height_cm=0.0):
    """Trajectory units -> metres, as (scale, reason).

    Mono VO fixes translation only up to an unknown multiplier. Defaulting to 1.0 would put
    every label off by a constant while every check still passes, so an unpinned bag raises.
    """
    if pinned_traj_scale is not None:
        return float(pinned_traj_scale), f"pinned: traj_scale {pinned_traj_scale:g}"

    if pinned_path_m:
        scale = float(pinned_path_m) / max(raw_path_length, 1e-9)
        return scale, f"path_length {pinned_path_m:g} m / {raw_path_length:.2f} raw units"

    if measured_height:
        height_units, spread = measured_height
        rig_height_m = rig_height_cm / 100.0
        scale = rig_height_m / height_units
        return scale, (f"floor match: {rig_height_m:.3f} m / {height_units:.3f} raw units, "
                       f"spread +/-{spread * 100:.0f}%")

    raise ValueError("no metric scale for this bag. Mono VO is scale-free, so either set "
                     "AUTO_SCALE=true in .env, or pin it in SCALE_OVERRIDES there: "
                     "{'path_length': <real metres walked>} or {'traj_scale': <factor>}.")


def select_keyframes(image_times_ns, image_row_ids, pose_times_ns, world_poses, yaws,
                     height_m, pitch_deg):
    """Poses worth a training frame, as a list of dicts: on the TARGET_FPS grid, with an
    image within TOL_MS, and having actually moved since the last kept frame."""
    tolerance_ns = int(TOL_MS * 1e6)
    frames = []

    for pose_index in resample_indices(pose_times_ns, TARGET_FPS).tolist():
        pose_time_ns = int(pose_times_ns[pose_index])

        image_index = index_nearest_time(image_times_ns, pose_time_ns)
        if abs(int(image_times_ns[image_index]) - pose_time_ns) > tolerance_ns:
            continue

        x = float(world_poses[pose_index, 0, 3])
        y = float(world_poses[pose_index, 1, 3])
        yaw = float(yaws[pose_index])

        if frames:
            last = frames[-1]
            moved = math.hypot(x - last["x"], y - last["y"])
            turned = abs(yaw_delta_deg(yaw, last["yaw"]))
            if moved < MIN_MOVE and turned < MIN_TURN_DEG:
                continue

        frames.append({
            "time_ns": pose_time_ns,
            "row_id": image_row_ids[image_index],
            "x": x,
            "y": y,
            "yaw": yaw,
            "pose": synthesize_camera_pose(x, y, yaw, height_m, pitch_deg),
        })

    return frames


def discretize_actions(frames):
    """action[i] is what was done to ARRIVE AT frame i.

    Frame 0 gets STOP: the loader shifts this column left by one, so that entry is always
    discarded, and STOP rather than -1 keeps every value in the loader's id table.
    """
    actions = [STOP]
    for index in range(1, len(frames)):
        turn_deg = yaw_delta_deg(frames[index]["yaw"], frames[index - 1]["yaw"])
        if abs(turn_deg) < MIN_TURN_DEG:
            actions.append(FORWARD)
        else:
            actions.append(LEFT if turn_deg > 0 else RIGHT)
    return np.array(actions, np.int64)


def find_subgoal_frames(frames, actions):
    """Frames worth aiming at: every turn, every SUBGOAL_DIST metres, and the last one."""
    subgoal_frames = {index for index in range(1, len(actions))
                      if actions[index] in (LEFT, RIGHT) and actions[index - 1] == FORWARD}

    walked = 0.0
    for index in range(1, len(frames)):
        walked += math.hypot(frames[index]["x"] - frames[index - 1]["x"],
                             frames[index]["y"] - frames[index - 1]["y"])
        if walked >= SUBGOAL_DIST:
            subgoal_frames.add(index)
            walked = 0.0

    subgoal_frames.add(len(frames) - 1)
    return sorted(subgoal_frames)


def project_to_pixel(camera_to_world, point_world, intrinsics):
    """Pixel (u, v) of a world point, or None if it is behind the camera or off-image."""
    rotation_world_camera = camera_to_world[:3, :3]
    camera_origin = camera_to_world[:3, 3]
    x, y, z = rotation_world_camera.T.dot(point_world - camera_origin)
    if z <= 1e-6:
        return None

    pixel_u = int(round(intrinsics[0, 0] * x / z + intrinsics[0, 2]))
    pixel_v = int(round(intrinsics[1, 1] * y / z + intrinsics[1, 2]))
    if 0 <= pixel_u < OUT_WIDTH and 0 <= pixel_v < OUT_HEIGHT:
        return pixel_u, pixel_v
    return None


def build_labels(frames, intrinsics_out):
    """(action, goal, relative_goal_frame_id) for one episode."""
    n_frames = len(frames)
    actions = discretize_actions(frames)
    subgoal_frames = find_subgoal_frames(frames, actions)

    goals = np.full((n_frames, 2), -1, np.int64)
    relative_goal_frame_ids = np.full(n_frames, -1, np.int64)

    for frame_index, frame in enumerate(frames):
        goal_frame = next((subgoal for subgoal in subgoal_frames
                           if subgoal - frame_index >= MIN_GOAL_K), None)
        if goal_frame is None or (goal_frame - frame_index) > MAX_GOAL_FRAMES:
            continue

        waypoint_world = np.array([frames[goal_frame]["x"], frames[goal_frame]["y"], 0.0])
        pixel = project_to_pixel(frame["pose"], waypoint_world, intrinsics_out)
        if pixel is not None:
            goals[frame_index] = pixel
            relative_goal_frame_ids[frame_index] = goal_frame - frame_index

    return actions, goals, relative_goal_frame_ids


def check_goal_reach(speed_mps=1.2):
    """Warn if SUBGOAL_DIST is too near to survive the loader's k >= 3 filter. Nothing else
    reports this: the dataset builds fine and the loader just comes out nearly empty."""
    if TARGET_FPS <= 0:
        return None

    frames_ahead = SUBGOAL_DIST / (speed_mps / TARGET_FPS)
    if frames_ahead >= MIN_GOAL_K:
        return None

    needed_dist = MIN_GOAL_K * speed_mps / TARGET_FPS
    return (f"SUBGOAL_DIST {SUBGOAL_DIST:g} m at {TARGET_FPS:g} fps and {speed_mps:.1f} m/s "
            f"puts the goal only {frames_ahead:.1f} frames ahead; the loader keeps k >= 3 "
            f"only. Raise it to >= {needed_dist:.1f} m, or raise TARGET_FPS.")


def measure_scale(bag, image_times_ns, image_row_ids, pose_times_ns, positions, quaternions,
                  up, intrinsics, traj_scale, path_m):
    """Floor-matched height in trajectory units, or None when the bag is pinned by hand."""
    if not AUTO_SCALE or traj_scale is not None or path_m:
        return None

    from scale import measure_height
    return measure_height(bag, image_times_ns, image_row_ids, pose_times_ns, positions,
                          quaternions, up, intrinsics, TOL_MS)


def output_intrinsics(camera_info):
    """Intrinsics after the same crop and resize the written images get."""
    crop_box = center_crop_box(camera_info["width"], camera_info["height"],
                               OUT_WIDTH, OUT_HEIGHT)
    return crop_resize_intrinsics(camera_info["intrinsics"], crop_box,
                                  camera_info["width"], camera_info["height"],
                                  OUT_WIDTH, OUT_HEIGHT)


def sanity_warnings(speed_mps, measured_height):
    warnings = []
    if not SPEED_MIN <= speed_mps <= SPEED_MAX:
        warnings.append(f"implied speed {speed_mps:.2f} m/s outside "
                        f"{SPEED_MIN:g}-{SPEED_MAX:g} m/s — check the scale")
    if measured_height and measured_height[1] > MAX_SPREAD:
        warnings.append(f"auto-scale spread +/-{measured_height[1] * 100:.0f}% "
                        f"(> {MAX_SPREAD * 100:.0f}%) — pin this bag by hand")
    return warnings


def build_episode(bag, traj_path, traj_scale=None, path_m=None):
    """trajectory + bag -> everything one episode needs. Nothing is written."""
    height_cm, pitch_deg = require_rig()
    height_m = height_cm / 100.0

    pose_times_ns, positions, quaternions = load_trajectory(traj_path)
    camera_info = bag.camera_info()
    image_times_ns, image_row_ids = bag.image_index()

    up = up_direction_from_pitch(pitch_deg)
    raw_path_length = path_length(positions)

    measured_height = measure_scale(bag, image_times_ns, image_row_ids, pose_times_ns,
                                    positions, quaternions, up, camera_info["intrinsics"],
                                    traj_scale, path_m)
    scale, scale_reason = resolve_scale(raw_path_length, traj_scale, path_m,
                                        measured_height, height_cm)
    world_poses, yaws = world_from_trajectory(positions, quaternions, up, height_m, scale)

    frames = select_keyframes(image_times_ns, image_row_ids, pose_times_ns, world_poses, yaws,
                              height_m, pitch_deg)
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"only {len(frames)} keyframe(s), need {MIN_FRAMES} — "
                         "lower MIN_MOVE, or check TOL_MS")

    actions, goals, relative_goal_frame_ids = build_labels(frames, output_intrinsics(camera_info))

    duration_s = float(pose_times_ns[-1] - pose_times_ns[0]) / 1e9
    speed_mps = raw_path_length * scale / duration_s if duration_s > 0 else 0.0

    return {
        "stamps": np.array([frame["time_ns"] for frame in frames], np.int64),
        "row_ids": [frame["row_id"] for frame in frames],
        "pose": np.array([frame["pose"] for frame in frames], np.float32),
        "action": actions,
        "goal": goals,
        "relative_goal_frame_id": relative_goal_frame_ids,
        "n_goal": int((relative_goal_frame_ids >= 0).sum()),
        "scale": scale,
        "reason": scale_reason,
        "speed": speed_mps,
        "warnings": sanity_warnings(speed_mps, measured_height),
    }


def parquet_path(scene_dir, episode_index):
    return (Path(scene_dir) / "data" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / f"episode_{episode_index:06d}.parquet")


def episode_parquets(scene_dir):
    """(episode_index, path) for every episode in this scene, in index order."""
    return sorted((int(path.stem[len("episode_"):]), path)
                  for path in Path(scene_dir).glob("data/chunk-*/episode_*.parquet"))


def save_frame_images(bag, row_ids, scene_dir, episode_index):
    """One cropped JPEG per frame, plus a blank depth PNG.

    The loader wants two RGB streams and one depth stream; this robot has one camera and no
    depth sensor, so depth is a 16-bit zero image. Every path must exist or __getitem__ crashes.
    """
    tag = setting_tag()
    chunk_dir = Path(scene_dir) / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
    rgb_dir = chunk_dir / f"observation.images.rgb.{tag}"
    depth_dir = chunk_dir / f"observation.images.depth.{tag}"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    stem = f"episode_{episode_index:06d}"
    blank_depth = Image.fromarray(np.zeros((OUT_HEIGHT, OUT_WIDTH), np.uint16))

    for frame, row_id in enumerate(row_ids):
        image = bag.image(row_id)
        crop_box = center_crop_box(image.width, image.height, OUT_WIDTH, OUT_HEIGHT)
        if crop_box is not None:
            image = image.crop(crop_box)
        image = image.resize((OUT_WIDTH, OUT_HEIGHT), Image.BILINEAR)

        image.save(rgb_dir / f"{stem}_{frame}.jpg", quality=JPEG_QUALITY)
        blank_depth.save(depth_dir / f"{stem}_{frame}.png", optimize=True)


def episode_table(episode, episode_index, task_index):
    """The parquet table for one episode.

    Two dtypes are load-bearing: action and relative_goal_frame_id are int64 SCALARS, not (1,)
    lists (as lists the loader raises "unhashable type: numpy.ndarray"), and pose stays a
    nested 4x4 because the loader does not reshape. `index` is fixed by fix_index_column.
    """
    tag = setting_tag()
    n_frames = len(episode["stamps"])
    timestamps_s = (episode["stamps"] - episode["stamps"][0]) / 1e9

    return pa.table({
        "action": pa.array(episode["action"], pa.int64()),
        f"pose.{tag}": pa.array([pose.tolist() for pose in episode["pose"]],
                                pa.list_(pa.list_(pa.float32()))),
        f"goal.{tag}": pa.array([goal.tolist() for goal in episode["goal"]],
                                pa.list_(pa.int64(), 2)),
        f"relative_goal_frame_id.{tag}": pa.array(episode["relative_goal_frame_id"], pa.int64()),
        "timestamp": pa.array(timestamps_s.astype(np.float32), pa.float32()),
        "frame_index": pa.array(np.arange(n_frames, dtype=np.int64), pa.int64()),
        "episode_index": pa.array(np.full(n_frames, episode_index, np.int64), pa.int64()),
        "index": pa.array(np.arange(n_frames, dtype=np.int64), pa.int64()),
        "task_index": pa.array(np.full(n_frames, task_index, np.int64), pa.int64()),
    })


def write_episode(bag, episode, scene_dir, episode_index, task_index):
    """Images and parquet for one episode. meta/ describes the whole scene and is written
    separately, by rebuild_meta."""
    save_frame_images(bag, episode["row_ids"], scene_dir, episode_index)

    path = parquet_path(scene_dir, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_table(episode, episode_index, task_index), path)


def fix_index_column(scene_dir):
    """Make each episode's `index` column continue from the episode before it.

    `index` runs across the WHOLE scene, so it depends on every lower-numbered episode's
    length. Bags arrive out of order, and a late low one invalidates every offset above it.
    """
    renumbered, offset = [], 0
    for episode_index, path in episode_parquets(scene_dir):
        table = pq.read_table(path)
        n_rows = table.num_rows

        if n_rows and table.column("index")[0].as_py() != offset:
            corrected = pa.array(np.arange(offset, offset + n_rows, dtype=np.int64), pa.int64())
            column_position = table.schema.get_field_index("index")
            pq.write_table(table.set_column(column_position, "index", corrected), path)
            renumbered.append(episode_index)

        offset += n_rows
    return renumbered


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def column_stats(values, n_rows):
    matrix = np.asarray([np.asarray(value).ravel() for value in values],
                        np.float64).reshape(n_rows, -1)
    return {
        "min": matrix.min(0).tolist(),
        "max": matrix.max(0).tolist(),
        "mean": matrix.mean(0).tolist(),
        "std": matrix.std(0).tolist(),
        "count": [n_rows],
    }


def collect_tasks(episode_tables, episode_info):
    """task_index -> instruction. NOT episode_index: one instruction has several takes, and
    they must stay in the same train/val split."""
    tasks = {}
    for episode_index, _ in episode_tables:
        info = episode_info[episode_index]
        task_index, instruction = info["task_index"], info["instruction"]
        if tasks.setdefault(task_index, instruction) != instruction:
            raise ValueError(f"task_index {task_index} maps to two instructions:\n"
                             f"  {tasks[task_index]!r}\n  {instruction!r}")
    return tasks


def measured_fps(episode_tables):
    """Read back from the timestamps actually written, so nothing is carried in state."""
    rates = []
    for _, table in episode_tables:
        timestamps_s = table.column("timestamp").to_pylist()
        if len(timestamps_s) > 1 and timestamps_s[-1] > timestamps_s[0]:
            rates.append((len(timestamps_s) - 1) / (timestamps_s[-1] - timestamps_s[0]))
    return sum(rates) / len(rates) if rates else TARGET_FPS


def write_meta_info(meta_dir, episode_tables, tasks):
    tag = setting_tag()
    highest_episode = max(episode_index for episode_index, _ in episode_tables)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "custom",
        "total_episodes": len(episode_tables),
        "total_frames": int(sum(table.num_rows for _, table in episode_tables)),
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": highest_episode // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": measured_fps(episode_tables),
        "splits": {"train": f"0:{len(episode_tables)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/"
                       "episode_{episode_index:06d}_{frame_index}.jpg"),
        "features": {
            "action": {"dtype": "int64", "shape": [1], "names": ["action_index"]},
            f"pose.{tag}": {"dtype": "float32", "shape": [4, 4], "names": ["row", "col"]},
            f"goal.{tag}": {"dtype": "int64", "shape": [2], "names": ["u", "v"]},
            f"relative_goal_frame_id.{tag}": {"dtype": "int64", "shape": [1], "names": ["k"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def rebuild_meta(scene_dir, episode_info):
    """Regenerate the whole of meta/ from the parquets on disk.

    Reading them back, instead of carrying labels in memory, is what lets a bag recorded weeks
    later be appended. episode_info: episode_index -> {instruction, task_index, repeat_index,
    run_id}, from the instruction log.
    """
    tag = setting_tag()
    meta_dir = Path(scene_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_tables = [(episode_index, pq.read_table(path))
                      for episode_index, path in episode_parquets(scene_dir)]
    if not episode_tables:
        raise ValueError(f"scene '{scene_dir}' holds no parquet — nothing to describe")

    orphans = [episode_index for episode_index, _ in episode_tables
               if episode_index not in episode_info]
    if orphans:
        raise ValueError(f"episodes {orphans} are on disk with no instruction in the log — "
                         "fix instruction_log.json, or delete those episodes")

    tasks = collect_tasks(episode_tables, episode_info)
    stat_columns = ("action", f"goal.{tag}", f"relative_goal_frame_id.{tag}")

    write_jsonl(meta_dir / "episodes.jsonl", [
        {
            "episode_index": episode_index,
            "tasks": [episode_info[episode_index]["instruction"]],
            "length": table.num_rows,
            "task_index": episode_info[episode_index]["task_index"],
            "repeat_index": episode_info[episode_index]["repeat_index"],
            "success": True,
            "run_id": episode_info[episode_index]["run_id"],
        }
        for episode_index, table in episode_tables
    ])

    write_jsonl(meta_dir / "tasks.jsonl", [
        {"task_index": task_index, "task": tasks[task_index]}
        for task_index in sorted(tasks)
    ])

    write_jsonl(meta_dir / "episodes_stats.jsonl", [
        {
            "episode_index": episode_index,
            "stats": {name: column_stats(table.column(name).to_pylist(), table.num_rows)
                      for name in stat_columns},
        }
        for episode_index, table in episode_tables
    ])

    write_meta_info(meta_dir, episode_tables, tasks)
