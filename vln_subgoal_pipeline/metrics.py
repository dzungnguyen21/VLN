import math
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def euclidean_distance(pos_a: Sequence[float], pos_b: Sequence[float]) -> float:
    """Calculate Euclidean distance between two 2D or 3D positions."""
    a = np.asarray(pos_a, dtype=np.float64)
    b = np.asarray(pos_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def compute_path_length(trajectory: Sequence[Sequence[float]]) -> float:
    """Calculate total path length traversed along a sequence of (x, y, z) or (x, y) points."""
    if len(trajectory) < 2:
        return 0.0
    points = np.asarray(trajectory, dtype=np.float64)
    deltas = np.diff(points, axis=0)
    step_distances = np.linalg.norm(deltas, axis=1)
    return float(np.sum(step_distances))


def compute_distance_to_goal(position: Sequence[float], goals: Sequence[Union[Sequence[float], Dict[str, Any]]]) -> float:
    """
    Calculate minimum Euclidean distance from a position to any of the target goal positions.
    Goals can be a list of coordinate triples [x, y, z] or list of dicts [{'position': [x, y, z], 'radius': ...}].
    """
    min_dist = float("inf")
    for g in goals:
        if isinstance(g, dict) and "position" in g:
            g_pos = g["position"]
        else:
            g_pos = g
        dist = euclidean_distance(position, g_pos)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def compute_navigation_error(
    final_position: Sequence[float],
    goals: Sequence[Union[Sequence[float], Dict[str, Any]]],
) -> float:
    """
    Compute Navigation Error (NE):
    Distance in meters between final agent position and closest target goal.
    """
    return compute_distance_to_goal(final_position, goals)


def compute_success(
    final_position: Sequence[float],
    goals: Sequence[Union[Sequence[float], Dict[str, Any]]],
    success_threshold: float = 3.0,
) -> float:
    """
    Compute Success (SR indicator):
    1.0 if final navigation error <= success_threshold (default 3.0m), else 0.0.
    """
    ne = compute_navigation_error(final_position, goals)
    return 1.0 if ne <= success_threshold else 0.0


def compute_oracle_success(
    trajectory: Sequence[Sequence[float]],
    goals: Sequence[Union[Sequence[float], Dict[str, Any]]],
    success_threshold: float = 3.0,
) -> float:
    """
    Compute Oracle Success (OS / OSR indicator):
    1.0 if the agent was within success_threshold of any goal at ANY point along trajectory, else 0.0.
    """
    if not trajectory:
        return 0.0
    min_dist = min(compute_distance_to_goal(pos, goals) for pos in trajectory)
    return 1.0 if min_dist <= success_threshold else 0.0


def compute_spl(
    success: float,
    path_length: float,
    shortest_path_distance: float,
) -> float:
    """
    Compute SPL (Success weighted by Path Length):
    SPL = S * (L / max(P, L))
    where S is success (0 or 1), L is shortest path distance, and P is actual path length.
    """
    if success <= 0.0:
        return 0.0
    if shortest_path_distance <= 0.0:
        return 1.0 if path_length <= 0.0 else 0.0
    denom = max(path_length, shortest_path_distance)
    return float(success * (shortest_path_distance / denom))


def evaluate_trajectory(
    trajectory: Sequence[Sequence[float]],
    goals: Sequence[Union[Sequence[float], Dict[str, Any]]],
    shortest_path_distance: Optional[float] = None,
    start_position: Optional[Sequence[float]] = None,
    success_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    Evaluate an episode trajectory and compute standard VLN metrics:
    - SR (Success Rate indicator: 1.0 or 0.0)
    - NE (Navigation Error: distance to goal in meters)
    - OS (Oracle Success indicator: 1.0 or 0.0)
    - SPL (Success weighted by Path Length: [0, 1])
    - path_length (Total length traveled in meters)
    - shortest_path_distance (Reference shortest path distance)
    - oracle_navigation_error (Minimum distance to goal during episode)
    """
    if not trajectory:
        if start_position is not None:
            trajectory = [start_position]
        else:
            raise ValueError("Trajectory cannot be empty without a start_position.")

    final_pos = trajectory[-1]
    path_len = compute_path_length(trajectory)

    # Compute reference shortest path distance if not provided
    if shortest_path_distance is None:
        if start_position is not None:
            shortest_path_distance = compute_distance_to_goal(start_position, goals)
        else:
            shortest_path_distance = compute_distance_to_goal(trajectory[0], goals)

    ne = compute_navigation_error(final_pos, goals)
    sr = compute_success(final_pos, goals, success_threshold=success_threshold)
    osr = compute_oracle_success(trajectory, goals, success_threshold=success_threshold)
    spl = compute_spl(sr, path_len, shortest_path_distance)

    min_dist_to_goal = min(compute_distance_to_goal(p, goals) for p in trajectory)

    return {
        "SR": sr,
        "NE": ne,
        "OS": osr,
        "SPL": spl,
        "path_length": path_len,
        "shortest_path_distance": float(shortest_path_distance),
        "oracle_navigation_error": min_dist_to_goal,
    }


def aggregate_metrics(per_episode_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate per-episode evaluation dictionaries into dataset-level average metrics:
    SR, NE, OS, SPL, avg_path_length, avg_shortest_path_distance.
    """
    if not per_episode_results:
        return {
            "SR": 0.0,
            "NE": 0.0,
            "OS": 0.0,
            "SPL": 0.0,
            "avg_path_length": 0.0,
            "avg_shortest_path_distance": 0.0,
            "num_episodes": 0,
        }

    srs = [r.get("SR", r.get("success", 0.0)) for r in per_episode_results]
    nes = [r.get("NE", r.get("ne", 0.0)) for r in per_episode_results]
    oss = [r.get("OS", r.get("os", 0.0)) for r in per_episode_results]
    spls = [r.get("SPL", r.get("spl", 0.0)) for r in per_episode_results]
    pls = [r.get("path_length", 0.0) for r in per_episode_results]
    sps = [r.get("shortest_path_distance", 0.0) for r in per_episode_results]

    return {
        "SR": float(np.mean(srs)),
        "NE": float(np.mean(nes)),
        "OS": float(np.mean(oss)),
        "SPL": float(np.mean(spls)),
        "avg_path_length": float(np.mean(pls)),
        "avg_shortest_path_distance": float(np.mean(sps)),
        "num_episodes": len(per_episode_results),
    }

