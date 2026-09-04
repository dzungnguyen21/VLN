from libs import *
from config import RIG_HEIGHT_CM, RIG_PITCH_DEG, SCENE_DIR

from torch.utils.data import Dataset

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
CHUNK_SIZE = 1000  # must match convert/episode.py CHUNK_SIZE


def setting_tag():
    """Same <H>cm_<P>deg tag convert/episode.py used when writing pose.<tag>/goal.<tag>."""
    if RIG_HEIGHT_CM is None or RIG_PITCH_DEG is None:
        raise ValueError("set RIG_HEIGHT_CM and RIG_PITCH_DEG in .env — must match the values "
                         "used to build this dataset, or pose.<tag>/goal.<tag> won't resolve.")
    return f"{round(RIG_HEIGHT_CM)}cm_{round(RIG_PITCH_DEG)}deg"


def load_meta(scene_dir):
    meta_dir = Path(scene_dir) / "meta"
    tasks = {
        int(row["task_index"]): row["task"]
        for row in (json.loads(line) for line in (meta_dir / "tasks.jsonl").read_text().splitlines())
    }
    episodes = [
        json.loads(line) for line in (meta_dir / "episodes.jsonl").read_text().splitlines()
    ]
    info = json.loads((meta_dir / "info.json").read_text())
    return tasks, episodes, info


def parquet_path(scene_dir, episode_index):
    return (Path(scene_dir) / "data" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / f"episode_{episode_index:06d}.parquet")


def image_path(scene_dir, episode_index, frame_index, tag):
    chunk_dir = Path(scene_dir) / "videos" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
    stem = f"episode_{episode_index:06d}_{frame_index}"
    return chunk_dir / f"observation.images.rgb.{tag}" / f"{stem}.jpg"


def split_tasks(task_indices, val_fraction, seed):
    """Task-level split, never episode-level — a task's repeats must stay on one side."""
    task_indices = sorted(task_indices)
    rng = random.Random(seed)
    rng.shuffle(task_indices)
    n_val = max(1, int(len(task_indices) * val_fraction)) if task_indices else 0
    return set(task_indices[n_val:]), set(task_indices[:n_val])  # train, val


class ScopedTableReader:
    """Bounded-LRU parquet access shared by both dataset classes below — one scene's episodes
    are read over and over by index, so re-opening a whole table per row would dominate I/O."""

    def __init__(self, scene_dir, cache_size=16):
        self.scene_dir = Path(scene_dir)
        self.cache_size = cache_size
        self._cache = {}

    def table(self, episode_index):
        if episode_index not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            import pyarrow.parquet as pq
            self._cache[episode_index] = pq.read_table(parquet_path(self.scene_dir, episode_index))
        return self._cache[episode_index]


class PixelGoalTextDataset(Dataset):
    """System 2 training data: one sample per frame with a VALID goal — (image, instruction) ->
    the pixel-goal text Cosmos should generate. Frames with relative_goal_frame_id < 0 (no
    subgoal in range, per convert/episode.py) carry no supervision for this head and are
    dropped rather than trained toward a fabricated target.
    """

    def __init__(self, scene_dir=SCENE_DIR, task_filter=None):
        self.scene_dir = Path(scene_dir)
        self.tag = setting_tag()
        self.tasks, episodes, _ = load_meta(self.scene_dir)
        self.reader = ScopedTableReader(self.scene_dir)

        self.episodes = {
            episode["episode_index"]: episode
            for episode in episodes
            if task_filter is None or episode["task_index"] in task_filter
        }
        self.index = self._build_index()

    def _build_index(self):
        flat = []
        for episode_index in self.episodes:
            table = self.reader.table(episode_index)
            goal_valid = table.column(f"relative_goal_frame_id.{self.tag}").to_numpy() >= 0
            flat.extend((episode_index, frame_index)
                       for frame_index in np.nonzero(goal_valid)[0].tolist())
        return flat

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from PIL import Image

        episode_index, frame_index = self.index[index]
        table = self.reader.table(episode_index)

        task_index = table.column("task_index")[frame_index].as_py()
        goal_u, goal_v = table.column(f"goal.{self.tag}")[frame_index].as_py()

        return {
            "image": Image.open(image_path(self.scene_dir, episode_index, frame_index, self.tag))
                          .convert("RGB"),
            "instruction": self.tasks[task_index],
            "goal_u": int(goal_u),
            "goal_v": int(goal_v),
        }


class ActionFrameDataset(Dataset):
    """One sample = one frame's (image, goal pixel) paired with the action taken to reach the
    NEXT frame. action[i] in the parquet is "what was done to arrive at frame i"
    (convert/episode.py), so an observation at frame i pairs with action[i + 1]; the final frame
    of each episode has no next action and is dropped.

    Serves two trainers with different needs, hence the two knobs:
      * System 1 (system1_train.py) — `image_transform` to a normalized tensor, every frame kept:
        the controller is trained to cope with an absent goal, so `goal_valid` is a real input.
      * System 2's action task (system2_train.py) — raw PIL image, `require_valid_goal=True`:
        that task PUTS the goal pixel in the prompt, so a frame without one has nothing to say.
    """

    def __init__(self, scene_dir=SCENE_DIR, task_filter=None, image_transform=None,
                 require_valid_goal=False):
        self.scene_dir = Path(scene_dir)
        self.tag = setting_tag()
        self.tasks, episodes, _ = load_meta(self.scene_dir)
        self.reader = ScopedTableReader(self.scene_dir)
        self.image_transform = image_transform

        self.episodes = {
            episode["episode_index"]: episode
            for episode in episodes
            if task_filter is None or episode["task_index"] in task_filter
        }
        self.index = self._build_index(require_valid_goal)

    def _build_index(self, require_valid_goal):
        flat = []
        for episode_index, episode in self.episodes.items():
            frame_indices = range(episode["length"] - 1)  # last frame has no next action
            if require_valid_goal:
                goal_valid = self.reader.table(episode_index) \
                    .column(f"relative_goal_frame_id.{self.tag}").to_numpy() >= 0
                frame_indices = [frame_index for frame_index in frame_indices
                                 if goal_valid[frame_index]]
            flat.extend((episode_index, frame_index) for frame_index in frame_indices)
        return flat

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from PIL import Image

        episode_index, frame_index = self.index[index]
        table = self.reader.table(episode_index)

        image = Image.open(image_path(self.scene_dir, episode_index, frame_index, self.tag)) \
                     .convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        task_index = table.column("task_index")[frame_index].as_py()
        goal_uv = table.column(f"goal.{self.tag}")[frame_index].as_py()
        relative_goal_frame_id = table.column(f"relative_goal_frame_id.{self.tag}")[frame_index].as_py()
        next_action = table.column("action")[frame_index + 1].as_py()

        return {
            "image": image,
            "instruction": self.tasks[task_index],
            "goal_uv": torch.tensor(goal_uv, dtype=torch.float32),
            "goal_valid": torch.tensor(relative_goal_frame_id >= 0, dtype=torch.bool),
            "action": torch.tensor(next_action, dtype=torch.int64),
        }


def detect_jsonl_paths(scene_dir):
    """One file per episode, written by convert_habitat/landmarks.py — same chunk-XXX layout
    as the parquets so both stay resumable/reviewable per episode."""
    return sorted(Path(scene_dir).glob("detect/chunk-*/episode_*.jsonl"))


def episode_task_index(scene_dir):
    """episode_index -> task_index, from meta/episodes.jsonl — the same map load_meta's
    episodes list carries, pulled out here since DetectLandmarksDataset filters by task_index
    but its rows are written keyed by episode_index (convert_habitat/landmarks.py's rows)."""
    _, episodes, _ = load_meta(scene_dir)
    return {episode["episode_index"]: episode["task_index"] for episode in episodes}


class DetectLandmarksDataset(Dataset):
    """System 2's "detect" task training data: object-pointing + scene understanding, one
    sample per row convert_habitat/landmarks.py wrote — (image, candidate landmarks) -> which
    are visible + pixel, plus a best-guess exploration point/label/confidence and the current
    room. Rows already carry everything system2_train.py's detect_prompt/detect_target need;
    this class only loads the image and passes the row through.

    Real-recording scene_dirs (built by convert/, not convert_habitat/) have no detect/
    directory at all — dataset comes back empty rather than raising, so a SCENE_DIR without
    this data just contributes nothing to the "detect" task's split rather than breaking the
    multi-task run outright.
    """

    def __init__(self, scene_dir=SCENE_DIR, task_filter=None):
        self.scene_dir = Path(scene_dir)
        task_index_by_episode = episode_task_index(self.scene_dir)

        self.rows = []
        for path in detect_jsonl_paths(self.scene_dir):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                task_index = task_index_by_episode.get(row["episode_index"])
                if task_filter is None or task_index in task_filter:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(self.scene_dir / row["image_path"]).convert("RGB")

        return {
            "image": image,
            "instruction": row["instruction"],
            "guided_direction": row["guided_direction"],
            "candidates": row["candidates"],
            "visible": row["visible"],
            "guess_pixel_norm": row["guess_pixel_norm"],
            "guess_label": row["guess_label"],
            "guess_confidence": row["guess_confidence"],
            "current_location": row["current_location"],
        }


def build_split_datasets(dataset_class, scene_dir=SCENE_DIR, val_fraction=0.1, seed=0, **kwargs):
    tasks, _, _ = load_meta(scene_dir)
    train_tasks, val_tasks = split_tasks(list(tasks), val_fraction, seed)
    train_set = dataset_class(scene_dir, task_filter=train_tasks, **kwargs)
    val_set = dataset_class(scene_dir, task_filter=val_tasks, **kwargs)
    return train_set, val_set
