import math
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from .depth_utils import sample_patch_depth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Grounding3D")


class Projected3DPose:
    def __init__(
        self,
        x_camera: float,
        y_camera: float,
        z_camera: float,
        x_base: float,
        y_base: float,
        z_base: float,
        x_map: float,
        y_map: float,
        z_map: float,
        yaw_map: float,
        standoff_x_map: float,
        standoff_y_map: float,
        confidence: float,
    ):
        self.x_camera = x_camera
        self.y_camera = y_camera
        self.z_camera = z_camera
        
        self.x_base = x_base
        self.y_base = y_base
        self.z_base = z_base

        self.x_map = x_map
        self.y_map = y_map
        self.z_map = z_map
        self.yaw_map = yaw_map

        # Nav2 Waypoint (offset by standoff distance so robot doesn't hit object)
        self.standoff_x_map = standoff_x_map
        self.standoff_y_map = standoff_y_map
        self.confidence = confidence

    def to_nav2_goal(self) -> Dict[str, float]:
        """Format as 2D navigation goal dictionary for Nav2."""
        return {
            "x": self.standoff_x_map,
            "y": self.standoff_y_map,
            "z": 0.0,
            "yaw": self.yaw_map,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_frame": {"x": self.x_camera, "y": self.y_camera, "z": self.z_camera},
            "base_frame": {"x": self.x_base, "y": self.y_base, "z": self.z_base},
            "map_frame": {"x": self.x_map, "y": self.y_map, "z": self.z_map, "yaw": self.yaw_map},
            "nav2_standoff_goal": {"x": self.standoff_x_map, "y": self.standoff_y_map, "yaw": self.yaw_map},
            "confidence": self.confidence,
        }


class Grounding3D:
    """
    Projects 2D pixel coordinates (u, v) + aligned Depth map into 3D Robot Base & Map coordinates.
    Computes navigable target poses with standoff distance for Nav2.
    """

    def __init__(
        self,
        fx: float = 384.0,
        fy: float = 384.0,
        cx: float = 320.0,
        cy: float = 180.0,
        camera_offset_x: float = 0.15,
        camera_offset_y: float = 0.0,
        camera_offset_z: float = 0.50,
        standoff_dist: float = 0.60,  # stop 0.6m in front of target object
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_offset_z = camera_offset_z
        self.standoff_dist = standoff_dist

    def project_2d_to_3d(
        self,
        u: float,
        v: float,
        depth_map: np.ndarray,
        robot_pose: Optional[Dict[str, float]] = None,
        confidence: float = 1.0,
    ) -> Optional[Projected3DPose]:
        """
        Backproject 2D (u, v) pixel with depth to 3D Camera, Base, and Map frames.

        Args:
            u: pixel x coordinate
            v: pixel y coordinate
            depth_map: 2D numpy array (H, W) in meters
            robot_pose: Current robot pose in map frame {'x': float, 'y': float, 'yaw': float}
            confidence: detection confidence

        Returns:
            Projected3DPose or None if depth is invalid.
        """
        z_c = sample_patch_depth(depth_map, u, v)
        if z_c is None or z_c <= 0:
            logger.warning(f"Failed to obtain valid depth at pixel ({u:.1f}, {v:.1f})")
            return None

        # 1. Pinhole backprojection to Camera Optical Frame
        # Optical frame: X=right, Y=down, Z=forward
        x_c = (u - self.cx) * z_c / self.fx
        y_c = (v - self.cy) * z_c / self.fy

        # 2. Transform Camera Optical Frame -> Robot Base Frame (ROS REP-103 standard)
        # Base frame: X=forward, Y=left, Z=up
        x_b = z_c + self.camera_offset_x
        y_b = -x_c + self.camera_offset_y
        z_b = -y_c + self.camera_offset_z

        # 3. Transform Robot Base Frame -> Global Map Frame
        rx = robot_pose.get("x", 0.0) if robot_pose else 0.0
        ry = robot_pose.get("y", 0.0) if robot_pose else 0.0
        ryaw = robot_pose.get("yaw", 0.0) if robot_pose else 0.0

        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        x_map = rx + (x_b * cos_yaw - y_b * sin_yaw)
        y_map = ry + (x_b * sin_yaw + y_b * cos_yaw)
        z_map = z_b

        # Compute heading facing towards the landmark
        dx = x_map - rx
        dy = y_map - ry
        dist_to_landmark = math.hypot(dx, dy)
        landmark_yaw = math.atan2(dy, dx)

        # 4. Compute standoff goal (stop before colliding with the landmark)
        if dist_to_landmark > self.standoff_dist:
            ratio = (dist_to_landmark - self.standoff_dist) / max(dist_to_landmark, 1e-6)
            standoff_x = rx + dx * ratio
            standoff_y = ry + dy * ratio
        else:
            standoff_x = rx
            standoff_y = ry

        return Projected3DPose(
            x_camera=x_c,
            y_camera=y_c,
            z_camera=z_c,
            x_base=x_b,
            y_base=y_b,
            z_base=z_b,
            x_map=x_map,
            y_map=y_map,
            z_map=z_map,
            yaw_map=landmark_yaw,
            standoff_x_map=standoff_x,
            standoff_y_map=standoff_y,
            confidence=confidence,
        )
