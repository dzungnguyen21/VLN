import pytest
import numpy as np
from PIL import Image
from vln_subgoal_pipeline.pipeline import VLNSubgoalPipeline


def test_end_to_end_pipeline_mock_execution():
    pipeline = VLNSubgoalPipeline(use_mock_models=True, use_mock_nav2=True)

    instruction = "Exit through the corridor, turn left into the kitchen, and stop at the refrigerator."

    def camera_stream():
        rgb = Image.new("RGB", (640, 360), color=(150, 150, 200))
        depth = np.full((360, 640), 2.5, dtype=np.float32)
        return rgb, depth

    results = pipeline.run(
        instruction=instruction,
        get_rgbd_observation_fn=camera_stream,
    )

    assert results["success"] is True
    assert results["all_completed"] is True
    assert len(results["subgoals"]) >= 2
    for sg in results["subgoals"]:
        assert sg["status"] == "REACHED"
        assert sg["target_3d_pose"] is not None
        assert "x" in sg["target_3d_pose"]
        assert "y" in sg["target_3d_pose"]
