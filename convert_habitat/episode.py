"""Task D (pixel-goal) episode assembly — mirrors convert/episode.py's schema/algorithms as
closely as Habitat's discrete-stepping, ground-truth-pose simulator allows. Two functions are
NOT a straight copy of convert/episode.py, for reasons that matter:

- select_keyframes: convert/'s version resamples onto a TARGET_FPS/TOL_MS timestamp grid,
  which doesn't apply here — every Habitat sim step already produces a synchronized
  rgb/depth/semantic/pose tuple, so this instead just re-implements the moved/turned-since-
  last-kept-frame filter directly over the step stream.
- build_labels: convert/'s version projects the goal frame's floor point as (x, y, 0.0) —
  valid there because that pipeline assumes a flat floor at world z=0 with height added only
  to the camera. Habitat is Y-up and each frame's own floor_position (the AGENT's position,
  not the elevated sensor's) is already the true floor height at that point — using (x, y,
  0.0) here would silently zero out the wrong axis and mis-project every goal pixel on any
  scene where the floor isn't exactly at Y=0 (i.e. almost always). So this version projects
  frames[goal_frame]["floor_position"] directly.

discretize_actions, find_subgoal_frames, project_to_pixel (in geometry.py), episode_table,
parquet_path, episode_parquets, fix_index_column, rebuild_meta, write_meta_info,
collect_tasks, column_stats, write_jsonl, measured_fps are copied with only the constant
names swapped (HABITAT_* instead of convert/'s bare names) — they only ever consume the
abstract {x, y, yaw, pose} frame shape or plain parquet tables, no Habitat/ROS dependency.
"""
from libs import *
from config import (HABITAT_HFOV, HABITAT_JPEG_QUALITY, HABITAT_MAX_DEPTH_M,
                    HABITAT_MAX_GOAL_FRAMES, HABITAT_MIN_FRAMES, HABITAT_MIN_MOVE,
                    HABITAT_MIN_TURN_DEG, HABITAT_OUT_H, HABITAT_OUT_W, HABITAT_RIG_HEIGHT_CM,
                    HABITAT_RIG_PITCH_DEG, HABITAT_SUBGOAL_DIST)
from geometry import (ground_plane_yaw, habitat_pose_to_camera_to_world, project_to_pixel,
                      sensor_intrinsics, yaw_delta_deg)
from rollout import walk_episode

MIN_GOAL_K = 3
CHUNK_SIZE = 1000
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3

# Frames have no real wall-clock time in a step-based simulator; a nominal 1 Hz spacing is
# written purely so the parquet schema's `timestamp` column stays populated the way
# finetuning/data.py's loader expects the column to exist (it doesn't read the values).
NOMINAL_FRAME_PERIOD_NS = 1_000_000_000


def setting_tag():
    return f"{round(HABITAT_RIG_HEIGHT_CM)}cm_{round(HABITAT_RIG_PITCH_DEG)}deg"


def _frame_from_step(step):
    pose = habitat_pose_to_camera_to_world(step["position"], step["rotation"])
    floor_position = step["agent_position"]
    return {
        "x": float(floor_position[0]),
        "y": float(floor_position[2]),
        "yaw": ground_plane_yaw(pose),
        "pose": pose,
        "floor_position": floor_position,
        "rgb": step["rgb"],
        "depth": step["depth"],
        "semantic": step["semantic"],
    }


def select_keyframes(steps):
    """Frames worth a training sample: moved >= HABITAT_MIN_MOVE or turned >=
    HABITAT_MIN_TURN_DEG since the last kept frame — same filter shape as
    convert/episode.py's select_keyframes, minus the timestamp-grid machinery it needs and
    this doesn't. Always keeps the very last step of the episode, even if it wouldn't
    otherwise pass the filter."""
    frames = []
    last_step, last_kept = None, False

    for step in steps:
        last_step = step
        frame = _frame_from_step(step)

        keep = True
        if frames:
            moved = math.hypot(frame["x"] - frames[-1]["x"], frame["y"] - frames[-1]["y"])
            turned = abs(yaw_delta_deg(frame["yaw"], frames[-1]["yaw"]))
            keep = moved >= HABITAT_MIN_MOVE or turned >= HABITAT_MIN_TURN_DEG

        last_kept = keep
        if keep:
            frames.append(frame)

    if last_step is not None and not last_kept:
        frames.append(_frame_from_step(last_step))

    return frames


def discretize_actions(frames):
    """action[i] is what was done to ARRIVE AT frame i. Copied from convert/episode.py."""
    actions = [STOP]
    for index in range(1, len(frames)):
        turn_deg = yaw_delta_deg(frames[index]["yaw"], frames[index - 1]["yaw"])
        if abs(turn_deg) < HABITAT_MIN_TURN_DEG:
            actions.append(FORWARD)
        else:
            actions.append(LEFT if turn_deg > 0 else RIGHT)
    return np.array(actions, np.int64)


def find_subgoal_frames(frames, actions):
    """Frames worth aiming at: every turn, every HABITAT_SUBGOAL_DIST metres, and the last
    one. Copied from convert/episode.py — only consumes frame['x']/['y'] (ground-plane)."""
    subgoal_frames = {index for index in range(1, len(actions))
                      if actions[index] in (LEFT, RIGHT) and actions[index - 1] == FORWARD}

    walked = 0.0
    for index in range(1, len(frames)):
        walked += math.hypot(frames[index]["x"] - frames[index - 1]["x"],
                             frames[index]["y"] - frames[index - 1]["y"])
        if walked >= HABITAT_SUBGOAL_DIST:
            subgoal_frames.add(index)
            walked = 0.0

    subgoal_frames.add(len(frames) - 1)
    return sorted(subgoal_frames)


def build_labels(frames, intrinsics):
    """(action, goal, relative_goal_frame_id, subgoal_frames) for one episode. See module
    docstring for why the goal world point is frames[goal_frame]['floor_position'], not
    convert/episode.py's (x, y, 0.0)."""
    n_frames = len(frames)
    actions = discretize_actions(frames)
    subgoal_frames = find_subgoal_frames(frames, actions)

    goals = np.full((n_frames, 2), -1, np.int64)
    relative_goal_frame_ids = np.full(n_frames, -1, np.int64)

    for frame_index, frame in enumerate(frames):
        goal_frame = next((subgoal for subgoal in subgoal_frames
                           if subgoal - frame_index >= MIN_GOAL_K), None)
        if goal_frame is None or (goal_frame - frame_index) > HABITAT_MAX_GOAL_FRAMES:
            continue

        waypoint_world = frames[goal_frame]["floor_position"]
        pixel = project_to_pixel(frame["pose"], waypoint_world, intrinsics)
        if pixel is not None:
            goals[frame_index] = pixel
            relative_goal_frame_ids[frame_index] = goal_frame - frame_index

    return actions, goals, relative_goal_frame_ids, subgoal_frames


def build_episode(env, episode):
    """Habitat rollout -> everything one episode needs. Nothing is written."""
    frames = select_keyframes(walk_episode(env, episode))
    if len(frames) < HABITAT_MIN_FRAMES:
        raise ValueError(f"only {len(frames)} keyframe(s), need {HABITAT_MIN_FRAMES}")

    intrinsics = sensor_intrinsics(HABITAT_OUT_W, HABITAT_OUT_H, HABITAT_HFOV)
    actions, goals, relative_goal_frame_ids, subgoal_frames = build_labels(frames, intrinsics)

    n_frames = len(frames)
    stamps = np.arange(n_frames, dtype=np.int64) * NOMINAL_FRAME_PERIOD_NS

    return {
        "frames": frames,
        "stamps": stamps,
        "action": actions,
        "goal": goals,
        "relative_goal_frame_id": relative_goal_frame_ids,
        "subgoal_frames": subgoal_frames,
        "intrinsics": intrinsics,
        "n_goal": int((relative_goal_frame_ids >= 0).sum()),
    }


def parquet_path(scene_dir, episode_index):
    return (Path(scene_dir) / "data" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / f"episode_{episode_index:06d}.parquet")


def episode_parquets(scene_dir):
    """(episode_index, path) for every episode in this scene, in index order."""
    return sorted((int(path.stem[len("episode_"):]), path)
                  for path in Path(scene_dir).glob("data/chunk-*/episode_*.parquet"))


def rgb_image_path(scene_dir, episode_index, frame_index):
    tag = setting_tag()
    chunk_dir = Path(scene_dir) / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
    return chunk_dir / f"observation.images.rgb.{tag}" / f"episode_{episode_index:06d}_{frame_index}.jpg"


def save_frame_images(frames, scene_dir, episode_index):
    """One JPEG + one 16-bit depth PNG (millimetres, real — Habitat renders true depth, so
    unlike convert/'s all-zero placeholder there's no reason to blank it out) per frame.
    Rendered natively at HABITAT_OUT_W x HABITAT_OUT_H already, so no crop/resize step."""
    tag = setting_tag()
    chunk_dir = Path(scene_dir) / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
    rgb_dir = chunk_dir / f"observation.images.rgb.{tag}"
    depth_dir = chunk_dir / f"observation.images.depth.{tag}"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    stem = f"episode_{episode_index:06d}"
    for frame_index, frame in enumerate(frames):
        Image.fromarray(frame["rgb"]).save(rgb_dir / f"{stem}_{frame_index}.jpg",
                                           quality=HABITAT_JPEG_QUALITY)
        depth_mm = np.clip(frame["depth"] * HABITAT_MAX_DEPTH_M * 1000.0, 0, 65535) \
                     .astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_dir / f"{stem}_{frame_index}.png", optimize=True)


def episode_table(episode, episode_index, task_index):
    """Same dtypes as convert/episode.py's episode_table: action and
    relative_goal_frame_id.<tag> are int64 SCALARS (lists raise "unhashable type:
    numpy.ndarray" in finetuning/data.py's loader), pose.<tag> stays a nested 4x4."""
    tag = setting_tag()
    n_frames = len(episode["stamps"])
    timestamps_s = episode["stamps"].astype(np.float64) / 1e9

    return pa.table({
        "action": pa.array(episode["action"], pa.int64()),
        f"pose.{tag}": pa.array([frame["pose"].tolist() for frame in episode["frames"]],
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


def write_episode(episode, scene_dir, episode_index, task_index):
    """Images and parquet for one episode. meta/ is written separately by rebuild_meta."""
    save_frame_images(episode["frames"], scene_dir, episode_index)
    path = parquet_path(scene_dir, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_table(episode, episode_index, task_index), path)


def fix_index_column(scene_dir):
    """Copied from convert/episode.py — makes each episode's `index` column continue from
    the episode before it, across the whole scene, regardless of build order."""
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
        "min": matrix.min(0).tolist(), "max": matrix.max(0).tolist(),
        "mean": matrix.mean(0).tolist(), "std": matrix.std(0).tolist(), "count": [n_rows],
    }


def collect_tasks(episode_tables, episode_info):
    tasks = {}
    for episode_index, _ in episode_tables:
        info = episode_info[episode_index]
        task_index, instruction = info["task_index"], info["instruction"]
        if tasks.setdefault(task_index, instruction) != instruction:
            raise ValueError(f"task_index {task_index} maps to two instructions:\n"
                             f"  {tasks[task_index]!r}\n  {instruction!r}")
    return tasks


def measured_fps(episode_tables):
    rates = []
    for _, table in episode_tables:
        timestamps_s = table.column("timestamp").to_pylist()
        if len(timestamps_s) > 1 and timestamps_s[-1] > timestamps_s[0]:
            rates.append((len(timestamps_s) - 1) / (timestamps_s[-1] - timestamps_s[0]))
    return sum(rates) / len(rates) if rates else 1.0


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
    """Regenerate the whole of meta/ from the parquets on disk — copied from
    convert/episode.py. episode_info: episode_index -> {instruction, task_index,
    repeat_index, run_id}, from r2r_data.episode_info_by_index."""
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
        raise ValueError(f"episodes {orphans} are on disk with no matching R2R episode in "
                         "the split file")

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
