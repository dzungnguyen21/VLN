import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn

from ..models.cosmos3_reasoner import Cosmos3Reasoner
from ..models.locate_anything import LocateAnythingGrounder
from ..perception.grounding_3d import Grounding3D

app = FastAPI(title="VLN Subgoal & Grounding API Server", version="1.0.0")

# Global instances initialized on startup
reasoner: Optional[Cosmos3Reasoner] = None
grounder: Optional[LocateAnythingGrounder] = None
projector_3d: Optional[Grounding3D] = None


class DecomposeRequest(BaseModel):
    instruction: str
    image_base64: Optional[str] = None


class GroundAndProjectRequest(BaseModel):
    image_base64: str
    depth_base64: Optional[str] = None  # raw float32 bytes or 16-bit PNG encoded as base64
    target_description: str
    robot_pose: Optional[Dict[str, float]] = None


@app.on_event("startup")
def startup_event():
    global reasoner, grounder, projector_3d
    reasoner = Cosmos3Reasoner(use_mock=False)
    grounder = LocateAnythingGrounder(use_mock=False)
    projector_3d = Grounding3D()


@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "VLN Subgoal & Grounding Service"}


@app.post("/decompose_subgoals")
def decompose_subgoals(req: DecomposeRequest):
    if reasoner is None:
        raise HTTPException(status_code=500, detail="Reasoner model not initialized")

    img = None
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    subgoals = reasoner.decompose(instruction=req.instruction, image=img)
    return {"instruction": req.instruction, "subgoals": subgoals}


@app.post("/ground_and_project")
def ground_and_project(req: GroundAndProjectRequest):
    if grounder is None or projector_3d is None:
        raise HTTPException(status_code=500, detail="Perception models not initialized")

    # Decode RGB image
    img_bytes = base64.b64decode(req.image_base64)
    rgb_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = rgb_img.size

    # Decode or synthesize Depth map
    if req.depth_base64:
        depth_bytes = base64.b64decode(req.depth_base64)
        # Try loading as 16-bit PNG or float array
        try:
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_map = np.array(depth_img, dtype=np.float32) / 1000.0  # mm to meters
        except Exception:
            depth_map = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
    else:
        # Default flat 2.0m depth for testing
        depth_map = np.full((height, width), 2.0, dtype=np.float32)

    # 1. 2D Visual Grounding
    res_2d = grounder.ground(image=rgb_img, target_description=req.target_description)
    if res_2d is None:
        return {"success": False, "message": "Failed to locate landmark"}

    # 2. 3D Grounding & Coordinate Projection
    res_3d = projector_3d.project_2d_to_3d(
        u=res_2d.point_uv[0],
        v=res_2d.point_uv[1],
        depth_map=depth_map,
        robot_pose=req.robot_pose,
        confidence=res_2d.confidence,
    )

    if res_3d is None:
        return {"success": False, "message": "Invalid depth for 3D projection"}

    return {
        "success": True,
        "grounding_2d": res_2d.to_dict(),
        "grounding_3d": res_3d.to_dict(),
        "nav2_goal": res_3d.to_nav2_goal(),
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
