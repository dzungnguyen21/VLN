import pytest
import json
from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
from vln_subgoal_pipeline.navigation.subgoal_queue import SubgoalQueue, SubgoalStatus


def test_cosmos3_mock_decomposition():
    reasoner = Cosmos3Reasoner(use_mock=True)
    instruction = "Go out of the bedroom, walk down the hallway, and stop in front of the kitchen refrigerator."
    subgoals = reasoner.decompose(instruction)

    assert isinstance(subgoals, list)
    assert len(subgoals) >= 2
    for sg in subgoals:
        assert "id" in sg
        assert "description" in sg
        assert "target_landmark" in sg
        assert len(sg["target_landmark"]) > 0


def test_cosmos3_json_parser():
    reasoner = Cosmos3Reasoner(use_mock=True)
    json_response = """
    Here is the decomposition:
    ```json
    [
        {"id": 1, "description": "Exit bedroom", "target_landmark": "bedroom door"},
        {"id": 2, "description": "Turn right at mirror", "target_landmark": "wall mirror"}
    ]
    ```
    """
    parsed = reasoner._parse_json_subgoals(json_response, fallback_instruction="test")
    assert len(parsed) == 2
    assert parsed[0]["target_landmark"] == "bedroom door"
    assert parsed[1]["target_landmark"] == "wall mirror"


def test_subgoal_queue_lifecycle():
    queue = SubgoalQueue()
    subgoals = [
        {"id": 1, "description": "Step 1", "target_landmark": "door"},
        {"id": 2, "description": "Step 2", "target_landmark": "table"},
    ]
    queue.load_subgoals(subgoals)

    assert len(queue) == 2
    assert queue.has_more_subgoals() is True

    # Pop step 1
    sg1 = queue.get_next_subgoal()
    assert sg1 is not None
    assert sg1.id == 1
    assert sg1.status == SubgoalStatus.ACTIVE

    queue.mark_current_reached()
    assert sg1.status == SubgoalStatus.REACHED

    # Pop step 2
    sg2 = queue.get_next_subgoal()
    assert sg2 is not None
    assert sg2.id == 2
    queue.mark_current_reached()

    assert queue.has_more_subgoals() is False
    assert queue.is_all_completed() is True
