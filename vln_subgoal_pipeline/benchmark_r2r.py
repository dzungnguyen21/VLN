import argparse
import gzip
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image

# Ensure package root is in sys.path
package_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(package_root)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# IMPORTANT: import habitat_sim at module level BEFORE torch/transformers are loaded.
# habitat_sim registers LLVM command-line options via C++ static initialisers.
# If it is imported a second time (e.g. after a CUDA OOM recovery path re-triggers
# native library loading), LLVM aborts with "Option 'default' already exists!".
# Importing it here first guarantees it is only registered once.
try:
    import habitat_sim as _habitat_sim_preload  # noqa: F401
except ImportError:
    _habitat_sim_preload = None  # noqa: F841

from vln_subgoal_pipeline.inference_client import InferenceClient
from vln_subgoal_pipeline.perception.grounding_3d import Grounding3D

from vln_subgoal_pipeline.metrics import (
    evaluate_trajectory,
    aggregate_metrics,
    compute_distance_to_goal,
    euclidean_distance,
)


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def xyz_pitch_to_tf_matrix(xyz: np.ndarray, pitch: float) -> np.ndarray:
    x, y, z = xyz
    return np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch), x],
            [0.0, 1.0, 0.0, y],
            [-np.sin(pitch), 0.0, np.cos(pitch), z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def xyz_yaw_pitch_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    rot1 = xyz_yaw_to_tf_matrix(xyz, yaw)[:3, :3]
    rot2 = xyz_pitch_to_tf_matrix(xyz, pitch)[:3, :3]
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot1 @ rot2
    transformation_matrix[:3, 3] = xyz
    return transformation_matrix


def get_axis_align_matrix() -> np.ndarray:
    return np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def get_intrinsic_matrix(sensor_cfg) -> np.ndarray:
    width = sensor_cfg.width
    height = sensor_cfg.height
    fov = sensor_cfg.hfov
    fx = (width / 2.0) / math.tan(math.radians(fov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def pixel_to_gps(pixel: List[int], depth: np.ndarray, intrinsic: np.ndarray, tf_camera_to_episodic: np.ndarray) -> Tuple[float, float]:
    v, u = pixel
    z = float(depth[v, u])
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    point_camera = np.array([x, y, z, 1.0])
    point_episodic = tf_camera_to_episodic @ point_camera
    point_episodic = point_episodic[:3] / point_episodic[3]
    return (float(point_episodic[0]), float(point_episodic[1]))


def load_r2r_episodes(
    split: str = "val_seen",
    dataset_dir: str = "data/vln_ce/raw_data/r2r",
) -> List[Dict[str, Any]]:
    """Load episodes from R2R json.gz file."""
    gz_path = os.path.join(dataset_dir, split, f"{split}.json.gz")
    json_path = os.path.join(dataset_dir, split, f"{split}.json")

    # Fallback to datasets/vln/mp3d/r2r/v1 if needed
    if not os.path.exists(gz_path) and not os.path.exists(json_path):
        alt_gz = os.path.join("data/datasets/vln/mp3d/r2r/v1", split, f"{split}.json.gz")
        if os.path.exists(alt_gz):
            gz_path = alt_gz

    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"R2R dataset not found at {gz_path} or {json_path}")

    episodes = data.get("episodes", []) if isinstance(data, dict) else data
    return episodes


# =========================================================================
# Mode 1: Online Habitat Simulation Benchmarker (Live 3D Renderer)
# =========================================================================
# Uses raw habitat_sim directly (no habitat-lab, no YAML config) so that
# only ONE copy of habitat_sim is ever loaded, avoiding the LLVM double-
# registration crash ("Option 'default' already exists!").
# =========================================================================

# Default test-scene included in the habitat-test-scenes download.
_DEFAULT_SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
_MP3D_SCENES_DIR = "data/scene_data/mp3d"   # full Matterport3D dataset
_SENSOR_RES = (360, 640)   # (height, width)
_SENSOR_HEIGHT = 1.25      # metres above ground
_STEP_SIZE = 0.25          # metres per forward step
_TURN_ANGLE = 15.0         # degrees per turn action


class HabitatR2RBenchmarker:
    """
    Online Habitat Simulator Benchmarker.

    Uses raw ``habitat_sim`` (not ``habitat.Env``/habitat-lab) so that
    exactly ONE copy of the native library is loaded into the process.
    Each episode is executed inside a freshly-configured Habitat-Sim
    environment and the agent receives live-rendered RGB + Depth frames.

    Scene selection priority
    ------------------------
    1.  The episode's ``scene_id`` field, if the file exists on disk.
    2.  ``scene_override`` constructor argument.
    3.  The built-in test scene (apartment_1.glb).
    """

    def __init__(
        self,
        habitat_config_path: str = "scripts/eval/configs/vln_r2r_lowmem.yaml",  # kept for CLI compat, ignored
        split: str = "val_seen",
        use_mock_models: bool = False,
        max_steps_per_episode: int = 500,
        max_steps_per_subgoal: int = 80,
        scene_override: Optional[str] = None,
        r2r_dataset_dir: str = "data/vln_ce/raw_data/r2r",
        save_video: bool = False,
    ):
        import habitat_sim  # single import – no habitat-lab involved

        self._habitat_sim = habitat_sim
        self.split = split
        self.use_mock_models = use_mock_models
        self.max_steps_per_episode = max_steps_per_episode
        self.max_steps_per_subgoal = max_steps_per_subgoal
        self.scene_override = scene_override
        self.save_video = save_video

        self.inference_client = InferenceClient(use_mock=use_mock_models)
        self.inference_client.start()

        self.projector = Grounding3D(
            fx=_SENSOR_RES[1] / 2.0, fy=_SENSOR_RES[1] / 2.0,
            cx=_SENSOR_RES[1] / 2.0, cy=_SENSOR_RES[0] / 2.0,
            camera_offset_x=0.15, camera_offset_y=0.0, camera_offset_z=0.50,
            standoff_dist=0.50,
        )

        # Load R2R episodes from the annotation JSON/GZ
        self.episodes = load_r2r_episodes(split=split, dataset_dir=r2r_dataset_dir)
        print(f"Loaded {len(self.episodes)} R2R episodes for split '{split}'.")

    def close(self):
        self.inference_client.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_scene(self, episode: Dict[str, Any]) -> str:
        """Return the GLB path to use for this episode.

        Resolution order
        ----------------
        1. scene_override (if set and exists on disk)
        2. Exact scene_id path (if the file exists as-is)
        3. data/scene_data/mp3d/<scene_id> – real MP3D dataset
        4. Fallback test scene (apartment_1.glb)
        """
        if self.scene_override and os.path.isfile(self.scene_override):
            return self.scene_override

        scene_id = episode.get("scene_id", "")

        # As-is (absolute or already correct relative path)
        if scene_id and os.path.isfile(scene_id):
            return scene_id

        # R2R episodes store scene_id as e.g. "mp3d/2t7WUuJeko7/2t7WUuJeko7.glb"
        # Strip the leading "mp3d/" prefix and prepend our local scenes dir.
        stripped = scene_id.lstrip("/")
        if stripped.startswith("mp3d/"):
            stripped = stripped[len("mp3d/"):]  # -> "2t7WUuJeko7/2t7WUuJeko7.glb"
        candidate = os.path.join(_MP3D_SCENES_DIR, stripped)
        if os.path.isfile(candidate):
            return candidate

        # Fallback: extract just the scene folder name, try <dir>/<name>/<name>.glb
        base = os.path.basename(scene_id)           # "2t7WUuJeko7.glb"
        folder = os.path.splitext(base)[0]          # "2t7WUuJeko7"
        candidate2 = os.path.join(_MP3D_SCENES_DIR, folder, base)
        if os.path.isfile(candidate2):
            return candidate2

        # Last resort: one of the downloaded test scenes
        return _DEFAULT_SCENE

    def _make_sim(self, scene_path: str):
        """Create and return a configured habitat_sim.Simulator."""
        hs = self._habitat_sim
        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = str(pathlib.Path(scene_path).resolve())
        sim_cfg.gpu_device_id = 0

        agent_cfg = hs.AgentConfiguration()
        agent_cfg.action_space = {
            "move_forward": hs.ActionSpec("move_forward", hs.ActuationSpec(amount=_STEP_SIZE)),
            "turn_left":    hs.ActionSpec("turn_left",    hs.ActuationSpec(amount=_TURN_ANGLE)),
            "turn_right":   hs.ActionSpec("turn_right",   hs.ActuationSpec(amount=_TURN_ANGLE)),
        }

        rgb_spec = hs.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = hs.SensorType.COLOR
        rgb_spec.resolution = list(_SENSOR_RES)
        rgb_spec.position = [0.0, _SENSOR_HEIGHT, 0.0]

        depth_spec = hs.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = hs.SensorType.DEPTH
        depth_spec.resolution = list(_SENSOR_RES)
        depth_spec.position = [0.0, _SENSOR_HEIGHT, 0.0]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        cfg = hs.Configuration(sim_cfg, [agent_cfg])
        return hs.Simulator(cfg)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run_episode(self, episode: Dict[str, Any], ep_idx: int) -> Dict[str, Any]:
        hs = self._habitat_sim

        scene_path = self._resolve_scene(episode)
        episode_id = episode.get("episode_id", ep_idx)
        instruction = episode.get("instruction", {}).get("instruction_text", "").strip()
        start_position = episode.get("start_position", None)
        goals = episode.get("goals", [])
        geodesic_dist = episode.get("info", {}).get("geodesic_distance", None)

        sim = self._make_sim(scene_path)

        try:
            # Place agent at the episode start position (or a random nav point)
            if start_position and sim.pathfinder.is_navigable(start_position):
                start_pt = np.array(start_position, dtype=np.float32)
            else:
                nav_pt = sim.pathfinder.get_random_navigable_point()
                start_pt = np.array([float(nav_pt[0]), float(nav_pt[1]), float(nav_pt[2])], dtype=np.float32)
            sim.get_agent(0).set_state(hs.AgentState(position=start_pt))

            # Compute geodesic distance to goal if not supplied
            if geodesic_dist is None and goals:
                goal_pos = goals[0].get("position", [0.0, 0.0, 0.0])
                sp = hs.ShortestPath()
                sp.requested_start = start_pt
                sp.requested_end = np.array(goal_pos, dtype=np.float32)
                sim.pathfinder.find_path(sp)
                geodesic_dist = float(sp.geodesic_distance)

            frames = []
            # Initial live-rendered frame
            obs = sim.get_sensor_observations()
            rgb_img = Image.fromarray(obs["rgb"][:, :, :3])
            depth_map = obs["depth"]
            if self.save_video:
                frames.append(np.array(rgb_img))

            # -- Step 1 : subgoal decomposition --------------------------
            subgoals = self.inference_client.decompose(instruction=instruction, image=rgb_img)

            trajectory = [[float(start_pt[0]), float(start_pt[1]), float(start_pt[2])]]
            current_pos = start_pt.copy()
            subgoals_grounded = 0
            total_steps = 0

            # -- Step 2 : sequential closed-loop navigation ---------------
            for sg in subgoals:
                if total_steps >= self.max_steps_per_episode:
                    break

                # Re-render from current location
                obs = sim.get_sensor_observations()
                rgb_img = Image.fromarray(obs["rgb"][:, :, :3])
                depth_map = obs["depth"]
                if self.save_video:
                    frames.append(np.array(rgb_img))

                landmark = sg.get("target_landmark", "")
                grounding = self.inference_client.ground(rgb_img, landmark)
                if grounding is None:
                    continue

                proj = self.projector.project_2d_to_3d(
                    u=grounding.point_uv[0],
                    v=grounding.point_uv[1],
                    depth_map=depth_map,
                    confidence=grounding.confidence,
                    robot_pose={"x": float(current_pos[0]), "y": float(current_pos[2]), "yaw": 0.0},
                )
                if proj is None:
                    continue

                subgoals_grounded += 1
                target_pos = np.array([proj.x_map, float(current_pos[1]), proj.y_map], dtype=np.float32)
                if not sim.pathfinder.is_navigable(target_pos):
                    target_pos = np.array(sim.pathfinder.snap_point(target_pos), dtype=np.float32)

                budget = min(self.max_steps_per_subgoal, self.max_steps_per_episode - total_steps)
                for _ in range(budget):
                    pos = sim.get_agent(0).get_state().position
                    vec = (target_pos - np.array(pos)) * 0.3
                    nxt = pos + vec
                    if sim.pathfinder.is_navigable(nxt):
                        sim.get_agent(0).set_state(hs.AgentState(position=nxt))
                        if getattr(self, "save_video", False):
                            obs_nxt = sim.get_sensor_observations()
                            frames.append(obs_nxt["rgb"][:, :, :3])
                    p = sim.get_agent(0).get_state().position
                    trajectory.append([float(p[0]), float(p[1]), float(p[2])])
                    total_steps += 1

                current_pos = np.array(sim.get_agent(0).get_state().position, dtype=np.float32)

        finally:
            if getattr(self, "save_video", False) and len(frames) > 0:
                import imageio
                out_dir = os.path.dirname(getattr(self, 'output_json_path', 'results/vln_subgoal_pipeline/dummy.json'))
                os.makedirs(out_dir, exist_ok=True)
                vid_path = os.path.join(out_dir, f"sim_episode_{episode_id}.mp4")
                try:
                    imageio.mimsave(vid_path, frames, fps=5)
                    print(f"Saved simulation video to {vid_path}")
                except Exception as e:
                    print(f"Failed to save video: {e}")
            sim.close()

        metrics = evaluate_trajectory(
            trajectory=trajectory,
            goals=goals,
            shortest_path_distance=float(geodesic_dist) if geodesic_dist else None,
            start_position=[float(start_pt[0]), float(start_pt[1]), float(start_pt[2])],
            success_threshold=3.0,
        )

        return {
            "scene_id": scene_path,
            "episode_id": episode_id,
            "instruction": instruction,
            "start_position": trajectory[0],
            "goals": goals,
            "path_length": metrics["path_length"],
            "success": metrics["SR"],
            "spl": metrics["SPL"],
            "os": metrics["OS"],
            "ne": metrics["NE"],
            "metrics": metrics,
            "subgoals_total": len(subgoals),
            "subgoals_grounded": subgoals_grounded,
            "trajectory_points": len(trajectory),
        }

    def run(self, max_episodes: Optional[int] = None) -> Dict[str, Any]:
        total = len(self.episodes)
        eval_count = min(max_episodes, total) if max_episodes else total

        print("=" * 80)
        print(f"  Habitat-Sim Online Benchmark (Split: {self.split}, Episodes: {eval_count})")
        print("=" * 80)

        per_episode = []
        for idx in range(eval_count):
            result = self.run_episode(self.episodes[idx], idx)
            per_episode.append(result)
            print(
                f"[{idx + 1:3d}/{eval_count:3d}] scene={os.path.basename(result['scene_id'])} "
                f"ep={result['episode_id']} | "
                f"SR: {result['success']:.1f} | NE: {result['ne']:.2f}m | "
                f"OS: {result['os']:.1f} | SPL: {result['spl']:.3f} | "
                f"Subgoals: {result['subgoals_grounded']}/{result['subgoals_total']}"
            )

        summary_metrics = aggregate_metrics([r["metrics"] for r in per_episode])
        return {
            "split": self.split,
            "mode": "habitat_simulation",
            "num_episodes": len(per_episode),
            "metrics": summary_metrics,
            "per_episode": per_episode,
        }


# =========================================================================
# Mode 2: Offline / Dataset Benchmarker (Sim-Free Mode)
# =========================================================================

class DatasetR2RBenchmarker:
    """
    Offline R2R Benchmarker:
    Evaluates the reasoning, landmark decomposition, and 3D projection
    directly on R2R dataset annotations without requiring 3D rendering meshes.
    """

    def __init__(
        self,
        split: str = "val_seen",
        use_mock_models: bool = True,
        max_steps_per_subgoal: int = 80,
    ):
        self.split = split
        self.use_mock_models = use_mock_models
        self.max_steps_per_subgoal = max_steps_per_subgoal

        self.inference_client = InferenceClient(use_mock=use_mock_models)
        self.inference_client.start()

        self.projector_3d = Grounding3D(
            fx=384.0, fy=384.0, cx=320.0, cy=180.0,
            camera_offset_x=0.15, camera_offset_y=0.0, camera_offset_z=0.50,
            standoff_dist=0.60
        )
        self.episodes = load_r2r_episodes(split=split)

    def close(self):
        self.inference_client.stop()

    def run_episode(self, episode: Dict[str, Any], ep_idx: int) -> Dict[str, Any]:
        episode_id = episode.get("episode_id", ep_idx)
        scene_id = episode.get("scene_id", "")
        start_pos = episode.get("start_position", [0.0, 0.0, 0.0])
        goals = episode.get("goals", [])
        geodesic_dist = episode.get("info", {}).get("geodesic_distance")
        if geodesic_dist is None and goals:
            geodesic_dist = compute_distance_to_goal(start_pos, goals)
        instruction = episode.get("instruction", {}).get("instruction_text", "").strip()

        # Step 1: Subgoal decomposition
        sample_img = Image.new("RGB", (640, 360), color=(120, 140, 160))
        depth_map = np.full((360, 640), 2.5, dtype=np.float32)

        subgoals = self.inference_client.decompose(instruction=instruction, image=sample_img)

        # Step 2: Track trajectory as robot navigates to each subgoal
        trajectory = [list(start_pos)]
        current_pos = np.array(start_pos, dtype=np.float64)
        subgoals_grounded = 0
        grounded_subgoals_info = []

        for sg in subgoals:
            landmark = sg.get("target_landmark", "")
            ground_res = self.inference_client.ground(sample_img, landmark)
            if ground_res is None:

                continue

            subgoals_grounded += 1
            proj_3d = self.projector_3d.project_2d_to_3d(
                u=ground_res.point_uv[0],
                v=ground_res.point_uv[1],
                depth_map=depth_map,
                confidence=ground_res.confidence,
                robot_pose={"x": current_pos[0], "y": current_pos[2], "yaw": 0.0}
            )

            if proj_3d:
                target_3d = np.array([proj_3d.x_map, current_pos[1], proj_3d.y_map], dtype=np.float64)
                for step in range(1, 4):
                    interp_pos = current_pos + (target_3d - current_pos) * (step / 3.0)
                    trajectory.append(interp_pos.tolist())
                current_pos = target_3d

                grounded_subgoals_info.append({
                    "id": sg.get("id"),
                    "landmark": landmark,
                    "point_uv": ground_res.point_uv,
                    "target_3d": target_3d.tolist(),
                })

        metrics = evaluate_trajectory(
            trajectory=trajectory,
            goals=goals,
            shortest_path_distance=geodesic_dist,
            start_position=start_pos,
            success_threshold=3.0,
        )

        return {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "instruction": instruction,
            "start_position": start_pos,
            "goals": goals,
            "shortest_path_distance": float(geodesic_dist) if geodesic_dist else 0.0,
            "path_length": metrics["path_length"],
            "success": metrics["SR"],
            "spl": metrics["SPL"],
            "os": metrics["OS"],
            "ne": metrics["NE"],
            "metrics": metrics,
            "subgoals_total": len(subgoals),
            "subgoals_grounded": subgoals_grounded,
            "grounded_subgoals": grounded_subgoals_info,
            "trajectory_points": len(trajectory),
        }

    def run(self, max_episodes: Optional[int] = None) -> Dict[str, Any]:
        total = len(self.episodes)
        eval_count = min(max_episodes, total) if max_episodes else total

        print("=" * 80)
        print(f"  Dataset Offline Benchmark (Split: {self.split}, Episodes: {eval_count})")
        print("=" * 80)

        per_episode = []
        for idx in range(eval_count):
            ep = self.episodes[idx]
            res = self.run_episode(ep, idx)
            per_episode.append(res)
            print(
                f"[{idx + 1:3d}/{eval_count:3d}] ep_id={res['episode_id']} | "
                f"SR: {res['success']:.1f} | NE: {res['ne']:.2f}m | "
                f"OS: {res['os']:.1f} | SPL: {res['spl']:.3f} | "
                f"Subgoals: {res['subgoals_grounded']}/{res['subgoals_total']}"
            )

        summary_metrics = aggregate_metrics([r["metrics"] for r in per_episode])

        return {
            "split": self.split,
            "mode": "dataset",
            "num_episodes": len(per_episode),
            "metrics": {
                "SR": summary_metrics["SR"],
                "NE": summary_metrics["NE"],
                "OS": summary_metrics["OS"],
                "SPL": summary_metrics["SPL"],
                "avg_path_length": summary_metrics["avg_path_length"],
                "avg_shortest_path_distance": summary_metrics["avg_shortest_path_distance"],
            },
            "per_episode": per_episode,
            "models": {
                "reasoner": "Cosmos3Reasoner (mock)" if self.use_mock_models else "Cosmos3Reasoner (live)",
                "grounder": "LocateAnythingGrounder (mock)" if self.use_mock_models else "LocateAnythingGrounder (live)",
            }
        }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vln_subgoal_pipeline on R2R with SR/NE/OS/SPL")
    parser.add_argument(
        "--habitat-config",
        type=str,
        default="scripts/eval/configs/vln_r2r_lowmem.yaml",
        help="Habitat config path (for Habitat mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["dataset", "habitat", "auto"],
        help="Evaluation mode: 'habitat' (runs live in Habitat sim), 'dataset' (evaluates on R2R annotations), or 'auto'",
    )
    parser.add_argument("--split", type=str, default="val_seen", choices=["val_seen", "val_unseen", "train"])
    parser.add_argument("--max-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=80)
    parser.add_argument("--use-mock-models", action="store_true", help="Use mock reasoner/grounder for fast evaluation")
    parser.add_argument("--save-video", action="store_true", help="Save MP4 video of the simulation runs")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/vln_subgoal_pipeline/r2r_benchmark_result.json",
        help="Output path for benchmark json",
    )
    args = parser.parse_args()

    benchmarker = None
    if args.mode in ["habitat", "auto"]:
        try:
            print("Attempting to initialize Habitat-Sim 3D environment...")
            benchmarker = HabitatR2RBenchmarker(
                habitat_config_path=args.habitat_config,
                split=args.split,
                use_mock_models=args.use_mock_models,
                max_steps_per_episode=args.max_steps_per_episode,
                max_steps_per_subgoal=args.max_steps_per_subgoal,
                save_video=args.save_video,
            )
            print("Habitat-Sim environment successfully initialized.")
        except Exception as e:
            if args.mode == "habitat":
                print(f"[Error] Failed to initialize Habitat simulation: {e}")
                sys.exit(1)
            else:
                print(f"[Info] Habitat 3D scene meshes not found locally ({e}). Falling back to Dataset mode.")
                benchmarker = DatasetR2RBenchmarker(
                    split=args.split,
                    use_mock_models=args.use_mock_models,
                    max_steps_per_subgoal=args.max_steps_per_subgoal,
                save_video=args.save_video,
                )
    else:
        benchmarker = DatasetR2RBenchmarker(
            split=args.split,
            use_mock_models=args.use_mock_models,
            max_steps_per_subgoal=args.max_steps_per_subgoal,
                save_video=args.save_video,
        )

    try:
        result = benchmarker.run(max_episodes=args.max_episodes)
    finally:
        benchmarker.close()

    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 50)
    print("           Final R2R Benchmark Results")
    print("=" * 50)
    print(f"  Evaluation Mode:        {result.get('mode', args.mode)}")
    print(f"  Split:                  {result['split']}")
    print(f"  Episodes Evaluated:     {result['num_episodes']}")
    print(f"  SR  (Success Rate):     {result['metrics']['SR'] * 100:.2f}%  ({result['metrics']['SR']:.4f})")
    print(f"  NE  (Navigation Error): {result['metrics']['NE']:.2f} meters")
    print(f"  OS  (Oracle Success):   {result['metrics']['OS'] * 100:.2f}%  ({result['metrics']['OS']:.4f})")
    print(f"  SPL (Path Efficiency):  {result['metrics']['SPL'] * 100:.2f}%  ({result['metrics']['SPL']:.4f})")
    print(f"  Avg Trajectory Length:  {result['metrics']['avg_path_length']:.2f} meters")
    print("=" * 50)
    print(f"Saved benchmark result JSON to: {args.output_json}\n")


if __name__ == "__main__":
    main()
