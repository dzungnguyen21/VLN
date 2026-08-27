import pytest
import numpy as np
from vln_subgoal_pipeline.perception.depth_utils import sample_patch_depth
from vln_subgoal_pipeline.perception.grounding_3d import Grounding3D


def test_depth_patch_sampling():
    # 100x100 depth map with constant 3.0m depth and one NaN in the center
    depth_map = np.full((100, 100), 3.0, dtype=np.float32)
    depth_map[50, 50] = np.nan  # noise point

    # Depth sampling at (50, 50) should filter NaN and return median 3.0m
    sampled = sample_patch_depth(depth_map, u=50, v=50, radius=3)
    assert sampled is not None
    assert abs(sampled - 3.0) < 1e-4


def test_grounding_3d_pinhole_projection():
    projector = Grounding3D(
        fx=400.0,
        fy=400.0,
        cx=320.0,
        cy=180.0,
        camera_offset_x=0.0,
        camera_offset_y=0.0,
        camera_offset_z=0.0,
        standoff_dist=0.5,
    )

    # 360x640 depth map with constant 2.0m depth
    depth_map = np.full((360, 640), 2.0, dtype=np.float32)
    robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    # Center pixel (320, 180) -> Camera frame (0, 0, 2.0)
    res = projector.project_2d_to_3d(u=320.0, v=180.0, depth_map=depth_map, robot_pose=robot_pose)

    assert res is not None
    assert abs(res.x_camera - 0.0) < 1e-4
    assert abs(res.y_camera - 0.0) < 1e-4
    assert abs(res.z_camera - 2.0) < 1e-4

    # In map frame (facing forward along +X)
    assert abs(res.x_map - 2.0) < 1e-4
    assert abs(res.y_map - 0.0) < 1e-4

    # Nav2 standoff goal (should stop standoff_dist=0.5m before target -> x=1.5)
    nav_goal = res.to_nav2_goal()
    assert abs(nav_goal["x"] - 1.5) < 1e-4
    assert abs(nav_goal["y"] - 0.0) < 1e-4
