import pytest
from PIL import Image
from vln_subgoal_pipeline.models.locate_anything import LocateAnythingGrounder, GroundingResult


def test_locate_anything_mock():
    grounder = LocateAnythingGrounder(use_mock=True)
    img = Image.new("RGB", (640, 360), color=(100, 100, 100))
    
    result = grounder.ground(image=img, target_description="refrigerator")
    assert result is not None
    assert isinstance(result, GroundingResult)
    assert len(result.bbox_xyxy) == 4
    # Check bbox within image bounds
    assert 0 <= result.bbox_xyxy[0] < result.bbox_xyxy[2] <= 640
    assert 0 <= result.bbox_xyxy[1] < result.bbox_xyxy[3] <= 360

    # Center point
    u, v = result.point_uv
    assert 0 <= u <= 640
    assert 0 <= v <= 360
    assert result.confidence > 0.5


def test_bbox_parser():
    grounder = LocateAnythingGrounder(use_mock=True)
    sample_output = "[100, 200, 500, 800]"
    bbox, conf = grounder._parse_bbox_prediction(sample_output, width=640, height=360)
    assert len(bbox) == 4
    # Normalized coords conversion check
    assert bbox[0] == (100 / 1000.0) * 640
    assert bbox[1] == (200 / 1000.0) * 360
