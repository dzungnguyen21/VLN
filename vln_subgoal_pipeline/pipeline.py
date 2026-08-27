import os
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from PIL import Image
import numpy as np

from .models.cosmos3_reasoner import Cosmos3Reasoner
from .models.locate_anything import LocateAnythingGrounder
from .perception.grounding_3d import Grounding3D
from .navigation.subgoal_queue import SubgoalQueue
from .navigation.nav2_client import MockNav2Client, LiveNav2Client, BaseNav2Client
from .navigation.closed_loop_controller import ClosedLoopVLNController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLNSubgoalPipeline")


class VLNSubgoalPipeline:
    """
    Main Orchestrator for the VLN Pipeline with Subgoal Handling.
    Integrates Cosmos 3 Reasoner + LocateAnything + 3D Grounding + Nav2.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        intrinsics_path: Optional[str] = None,
        use_mock_models: bool = False,
        use_mock_nav2: bool = True,
    ):
        # Load configs
        self.config = self._load_yaml(config_path or self._default_config_path("pipeline_config.yaml"))
        self.intrinsics = self._load_yaml(intrinsics_path or self._default_config_path("camera_intrinsics.yaml"))

        # 1. Initialize Cosmos 3 Reasoner
        cosmos_cfg = self.config.get("models", {}).get("cosmos3", {})
        self.reasoner = Cosmos3Reasoner(
            model_id=cosmos_cfg.get("model_id", "nvidia/Cosmos3-Edge"),
            device=cosmos_cfg.get("device", "cuda"),
            torch_dtype=cosmos_cfg.get("torch_dtype", "bfloat16"),
            max_new_tokens=cosmos_cfg.get("max_new_tokens", 512),
            use_mock=use_mock_models or cosmos_cfg.get("use_mock", False),
        )

        # 2. Initialize LocateAnything 2D Visual Grounder
        locate_cfg = self.config.get("models", {}).get("locate_anything", {})
        self.grounder = LocateAnythingGrounder(
            model_id=locate_cfg.get("model_id", "nvidia/LocateAnything-3B"),
            device=locate_cfg.get("device", "cuda"),
            torch_dtype=locate_cfg.get("torch_dtype", "bfloat16"),
            confidence_threshold=locate_cfg.get("confidence_threshold", 0.25),
            use_mock=use_mock_models or locate_cfg.get("use_mock", False),
        )

        # 3. Initialize 3D Grounding
        cam_intro = self.intrinsics.get("intrinsics", {})
        cam_extro = self.intrinsics.get("extrinsics_camera_to_base", {}).get("translation", {})
        self.projector_3d = Grounding3D(
            fx=cam_intro.get("fx", 384.0),
            fy=cam_intro.get("fy", 384.0),
            cx=cam_intro.get("cx", 320.0),
            cy=cam_intro.get("cy", 180.0),
            camera_offset_x=cam_extro.get("x", 0.15),
            camera_offset_y=cam_extro.get("y", 0.0),
            camera_offset_z=cam_extro.get("z", 0.50),
            standoff_dist=0.60,
        )

        # 4. Initialize Subgoal Queue
        self.subgoal_queue = SubgoalQueue()

        # 5. Initialize Nav2 Controller
        if use_mock_nav2 or self.config.get("navigation", {}).get("nav2", {}).get("use_mock", True):
            self.nav2_client: BaseNav2Client = MockNav2Client()
        else:
            try:
                self.nav2_client = LiveNav2Client()
            except Exception as e:
                logger.warning(f"Could not connect to live ROS 2 Nav2: {e}. Defaulting to MockNav2Client.")
                self.nav2_client = MockNav2Client()

        # 6. Initialize Closed-Loop Controller
        ctrl_cfg = self.config.get("navigation", {}).get("subgoal_control", {})
        self.controller = ClosedLoopVLNController(
            subgoal_queue=self.subgoal_queue,
            grounder=self.grounder,
            projector_3d=self.projector_3d,
            nav2_client=self.nav2_client,
            max_re_grounding_attempts=ctrl_cfg.get("max_re_grounding_attempts", 3),
            re_ground_distance_threshold=ctrl_cfg.get("re_ground_distance_threshold", 1.5),
        )

    def _default_config_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), "configs", filename)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def plan(self, instruction: str, initial_image: Optional[Image.Image] = None) -> SubgoalQueue:
        """Step 1: Fine-tuned / Prompted Cosmos 3 Subgoal Decomposition."""
        logger.info(f"Decomposing long-horizon instruction: '{instruction}'")
        subgoals = self.reasoner.decompose(instruction=instruction, image=initial_image)
        self.subgoal_queue.load_subgoals(subgoals)
        logger.info(f"Enqueued {len(self.subgoal_queue)} subgoals into SubgoalQueue.")
        return self.subgoal_queue

    def run(
        self,
        instruction: str,
        get_rgbd_observation_fn: Callable[[], Tuple[Image.Image, np.ndarray]],
        initial_image: Optional[Image.Image] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline from long-horizon instruction to task complete.

        Args:
            instruction: Long-horizon command text
            get_rgbd_observation_fn: Function providing live/simulated (RGB, Depth)
            initial_image: Optional initial visual frame for reasoner
            progress_callback: Optional status callback

        Returns:
            Execution summary dict.
        """
        # Step 1: Subgoal Decomposition & Enqueue
        self.plan(instruction=instruction, initial_image=initial_image)

        # Step 2: Closed-Loop Execution Loop
        success = self.controller.execute_all(
            get_rgbd_observation_fn=get_rgbd_observation_fn,
            progress_callback=progress_callback,
        )

        return {
            "success": success,
            "instruction": instruction,
            "subgoals": self.subgoal_queue.to_list(),
            "execution_log": [e.to_dict() for e in self.controller.execution_log],
            "all_completed": self.subgoal_queue.is_all_completed(),
        }
