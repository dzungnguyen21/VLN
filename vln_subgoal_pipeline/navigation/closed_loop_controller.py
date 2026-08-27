import logging
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from PIL import Image
import numpy as np

from .subgoal_queue import SubgoalQueue, SubgoalItem, SubgoalStatus
from .nav2_client import BaseNav2Client
from ..models.locate_anything import LocateAnythingGrounder, GroundingResult
from ..perception.grounding_3d import Grounding3D, Projected3DPose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClosedLoopController")


class ExecutionEvent:
    def __init__(self, step_name: str, details: Dict[str, Any]):
        self.timestamp = time.time()
        self.step_name = step_name
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step_name": self.step_name,
            "details": self.details,
        }


class ClosedLoopVLNController:
    """
    Orchestrates the closed-loop navigation execution matching the system architecture:
    [Get Next Subgoal] -> [Locate Anything] -> [3D Grounding] -> [Nav2] -> [Robot]
         ^                                                                  |
         |------------------- Subgoal Reached? (Yes) <----------------------|
         |                         | (No)                                   |
         |                  [Re-ground & Track] <---------------------------|
         |                         |
         |--- More Subgoals? (Yes) |
                 | (No)
          [Task Complete]
    """

    def __init__(
        self,
        subgoal_queue: SubgoalQueue,
        grounder: LocateAnythingGrounder,
        projector_3d: Grounding3D,
        nav2_client: BaseNav2Client,
        max_re_grounding_attempts: int = 3,
        re_ground_distance_threshold: float = 1.5,
    ):
        self.subgoal_queue = subgoal_queue
        self.grounder = grounder
        self.projector_3d = projector_3d
        self.nav2_client = nav2_client
        self.max_re_grounding_attempts = max_re_grounding_attempts
        self.re_ground_distance_threshold = re_ground_distance_threshold
        self.execution_log: List[ExecutionEvent] = []

    def log_event(self, step_name: str, details: Dict[str, Any]):
        event = ExecutionEvent(step_name, details)
        self.execution_log.append(event)
        logger.info(f"[{step_name.upper()}] {details.get('message', '')}")

    def execute_all(
        self,
        get_rgbd_observation_fn: Callable[[], Tuple[Image.Image, np.ndarray]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Main closed-loop execution loop.

        Args:
            get_rgbd_observation_fn: Function returning current (RGB Image, Depth array HxW)
            progress_callback: Optional status callback for monitoring / UI

        Returns:
            True if all subgoals reached successfully, False otherwise.
        """
        self.log_event("pipeline_start", {"message": "Starting closed-loop VLN execution."})

        while self.subgoal_queue.has_more_subgoals():
            current_subgoal: Optional[SubgoalItem] = self.subgoal_queue.get_next_subgoal()
            if current_subgoal is None:
                break

            self.log_event("get_next_subgoal", {
                "message": f"Processing Subgoal #{current_subgoal.id}: '{current_subgoal.description}'",
                "subgoal_id": current_subgoal.id,
                "target_landmark": current_subgoal.target_landmark,
            })

            subgoal_reached = False
            re_ground_count = 0

            while not subgoal_reached and re_ground_count < self.max_re_grounding_attempts:
                # 1. Capture current visual observation
                rgb_img, depth_map = get_rgbd_observation_fn()
                current_robot_pose = self.nav2_client.get_current_robot_pose()

                # 2. Locate Anything / Visual Pointing
                self.log_event("locate_anything", {
                    "message": f"Locating '{current_subgoal.target_landmark}' in camera frame...",
                    "target": current_subgoal.target_landmark,
                })
                grounding_res: Optional[GroundingResult] = self.grounder.ground(
                    image=rgb_img,
                    target_description=current_subgoal.target_landmark,
                )

                if grounding_res is None:
                    self.log_event("grounding_failed", {
                        "message": f"Failed to locate '{current_subgoal.target_landmark}'. Retrying...",
                        "attempt": re_ground_count + 1,
                    })
                    re_ground_count += 1
                    time.sleep(0.5)
                    continue

                # 3. 3D Grounding & Coordinate Projection
                self.log_event("grounding_3d", {
                    "message": f"Projecting 2D point {grounding_res.point_uv} + Depth -> 3D Map Target...",
                    "point_uv": grounding_res.point_uv,
                })
                projected_pose: Optional[Projected3DPose] = self.projector_3d.project_2d_to_3d(
                    u=grounding_res.point_uv[0],
                    v=grounding_res.point_uv[1],
                    depth_map=depth_map,
                    robot_pose=current_robot_pose,
                    confidence=grounding_res.confidence,
                )

                if projected_pose is None:
                    self.log_event("3d_projection_failed", {
                        "message": "Invalid depth measurement. Retrying perception...",
                        "attempt": re_ground_count + 1,
                    })
                    re_ground_count += 1
                    continue

                nav2_goal = projected_pose.to_nav2_goal()
                current_subgoal.target_3d_pose = nav2_goal

                # 4. Nav2 Dispatch
                self.log_event("nav2_dispatch", {
                    "message": f"Nav2 navigating to ({nav2_goal['x']:.2f}, {nav2_goal['y']:.2f})...",
                    "goal": nav2_goal,
                })

                if progress_callback:
                    progress_callback({
                        "event": "nav2_dispatched",
                        "subgoal": current_subgoal.model_dump(),
                        "projected_pose": projected_pose.to_dict(),
                    })

                success = self.nav2_client.navigate_to_pose(target_pose=nav2_goal)

                # 5. Check Subgoal Reached
                if success:
                    subgoal_reached = True
                    self.subgoal_queue.mark_current_reached()
                    self.log_event("subgoal_reached", {
                        "message": f"Subgoal #{current_subgoal.id} successfully REACHED.",
                        "subgoal_id": current_subgoal.id,
                    })
                else:
                    self.log_event("subgoal_not_reached", {
                        "message": "Subgoal not reached yet. Re-grounding landmark...",
                        "subgoal_id": current_subgoal.id,
                    })
                    re_ground_count += 1

            if not subgoal_reached:
                self.subgoal_queue.mark_current_failed()
                self.log_event("subgoal_failed", {
                    "message": f"Subgoal #{current_subgoal.id} FAILED after {re_ground_count} attempts.",
                    "subgoal_id": current_subgoal.id,
                })

        # 6. Final state
        all_done = self.subgoal_queue.is_all_completed()
        if all_done:
            self.log_event("task_complete", {"message": "All navigation subgoals COMPLETED successfully!"})
        else:
            self.log_event("task_incomplete", {"message": "Task finished with uncompleted subgoals."})

        return all_done
