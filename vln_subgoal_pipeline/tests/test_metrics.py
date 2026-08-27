import pytest
from vln_subgoal_pipeline.metrics import (
    euclidean_distance,
    compute_path_length,
    compute_distance_to_goal,
    compute_navigation_error,
    compute_success,
    compute_oracle_success,
    compute_spl,
    evaluate_trajectory,
    aggregate_metrics,
)


def test_euclidean_distance():
    assert abs(euclidean_distance([0, 0, 0], [3, 4, 0]) - 5.0) < 1e-6
    assert abs(euclidean_distance([1, 2, 3], [1, 2, 3]) - 0.0) < 1e-6


def test_path_length():
    traj = [[0, 0, 0], [3, 0, 0], [3, 4, 0]]
    assert abs(compute_path_length(traj) - 7.0) < 1e-6
    assert compute_path_length([[0, 0, 0]]) == 0.0
    assert compute_path_length([]) == 0.0


def test_distance_to_goal_and_success():
    goals = [{"position": [10.0, 0.0, 0.0], "radius": 2.0}]
    
    # Exactly on goal
    assert abs(compute_distance_to_goal([10.0, 0.0, 0.0], goals) - 0.0) < 1e-6
    assert compute_success([10.0, 0.0, 0.0], goals, success_threshold=3.0) == 1.0

    # Within 3.0m threshold (e.g., 2.5m away)
    assert compute_success([7.5, 0.0, 0.0], goals, success_threshold=3.0) == 1.0

    # Outside threshold (e.g., 3.5m away)
    assert compute_success([6.5, 0.0, 0.0], goals, success_threshold=3.0) == 0.0


def test_oracle_success():
    goals = [{"position": [10.0, 0.0, 0.0]}]
    # Agent passed near the goal at step 1 (dist=1.0m) then wandered off to (0, 0, 0)
    traj = [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    final_success = compute_success(traj[-1], goals, success_threshold=3.0)
    oracle_success = compute_oracle_success(traj, goals, success_threshold=3.0)
    
    assert final_success == 0.0
    assert oracle_success == 1.0


def test_spl_calculation():
    # Direct optimal path: S=1, P=10, L=10 -> SPL = 1.0
    assert abs(compute_spl(success=1.0, path_length=10.0, shortest_path_distance=10.0) - 1.0) < 1e-6

    # Inefficient path (wandered 20m): S=1, P=20, L=10 -> SPL = 10 / 20 = 0.5
    assert abs(compute_spl(success=1.0, path_length=20.0, shortest_path_distance=10.0) - 0.5) < 1e-6

    # Failed path: S=0 -> SPL = 0
    assert compute_spl(success=0.0, path_length=10.0, shortest_path_distance=10.0) == 0.0


def test_evaluate_trajectory_and_aggregation():
    goals = [{"position": [10.0, 0.0, 0.0]}]
    start = [0.0, 0.0, 0.0]

    # Episode 1: Perfect navigation
    traj1 = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]
    res1 = evaluate_trajectory(traj1, goals, shortest_path_distance=10.0, start_position=start)
    assert res1["SR"] == 1.0
    assert abs(res1["NE"] - 0.0) < 1e-6
    assert res1["OS"] == 1.0
    assert abs(res1["SPL"] - 1.0) < 1e-6

    # Episode 2: Failed navigation (wandered and stopped at 20, 0, 0)
    traj2 = [[0, 0, 0], [20, 0, 0]]
    res2 = evaluate_trajectory(traj2, goals, shortest_path_distance=10.0, start_position=start)
    assert res2["SR"] == 0.0
    assert abs(res2["NE"] - 10.0) < 1e-6
    assert res2["OS"] == 0.0
    assert res2["SPL"] == 0.0

    # Aggregate 2 episodes
    agg = aggregate_metrics([res1, res2])
    assert abs(agg["SR"] - 0.5) < 1e-6
    assert abs(agg["NE"] - 5.0) < 1e-6
    assert abs(agg["OS"] - 0.5) < 1e-6
    assert abs(agg["SPL"] - 0.5) < 1e-6
    assert agg["num_episodes"] == 2

