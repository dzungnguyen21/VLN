import time
import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Nav2Client")


class Nav2Status:
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class BaseNav2Client(ABC):
    @abstractmethod
    def navigate_to_pose(
        self,
        target_pose: Dict[str, float],
        feedback_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """Send target 3D goal pose and wait/monitor until completion or failure."""
        pass

    @abstractmethod
    def cancel_goal(self) -> None:
        """Cancel current active navigation goal."""
        pass

    @abstractmethod
    def get_current_robot_pose(self) -> Dict[str, float]:
        """Return current estimated robot pose in map frame {'x': ..., 'y': ..., 'yaw': ...}."""
        pass


class MockNav2Client(BaseNav2Client):
    """
    Simulated kinematic Nav2 controller for offline testing and verification.
    Simulates robot motion towards goal coordinates.
    """
    def __init__(
        self,
        initial_pose: Optional[Dict[str, float]] = None,
        linear_speed: float = 1.0,  # meters/sec in simulation
        sim_step_dt: float = 0.05,
    ):
        self.pose = initial_pose or {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.linear_speed = linear_speed
        self.sim_step_dt = sim_step_dt
        self._is_active = False
        self._canceled = False

    def get_current_robot_pose(self) -> Dict[str, float]:
        return dict(self.pose)

    def navigate_to_pose(
        self,
        target_pose: Dict[str, float],
        feedback_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        self._is_active = True
        self._canceled = False

        tx = target_pose["x"]
        ty = target_pose["y"]
        tyaw = target_pose.get("yaw", 0.0)

        logger.info(f"[MockNav2] Starting navigation to target ({tx:.2f}, {ty:.2f}, yaw={tyaw:.2f})")

        while self._is_active and not self._canceled:
            dx = tx - self.pose["x"]
            dy = ty - self.pose["y"]
            dist = math.hypot(dx, dy)

            if dist < 0.15:  # arrival threshold
                self.pose["x"] = tx
                self.pose["y"] = ty
                self.pose["yaw"] = tyaw
                logger.info(f"[MockNav2] Goal reached successfully at ({tx:.2f}, {ty:.2f})")
                self._is_active = False
                return True

            # Step motion
            step = min(dist, self.linear_speed * self.sim_step_dt)
            angle = math.atan2(dy, dx)
            self.pose["x"] += step * math.cos(angle)
            self.pose["y"] += step * math.sin(angle)
            self.pose["yaw"] = angle

            if feedback_callback:
                feedback_callback({
                    "current_pose": dict(self.pose),
                    "distance_remaining": dist - step,
                })

            time.sleep(0.01)  # small pause for simulated progress

        return False

    def cancel_goal(self) -> None:
        logger.info("[MockNav2] Canceling active goal.")
        self._canceled = True
        self._is_active = False


class LiveNav2Client(BaseNav2Client):
    """
    ROS 2 Nav2 Action Client using rclpy and nav2_msgs.action.NavigateToPose.
    """
    def __init__(self, node_name: str = "vln_nav2_client"):
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.node import Node
            from nav2_msgs.action import NavigateToPose
            from geometry_msgs.msg import PoseStamped

            if not rclpy.ok():
                rclpy.init()

            self.node = Node(node_name)
            self._action_client = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
            self._goal_handle = None
            self._current_robot_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
            logger.info("ROS 2 Nav2 Action Client initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ROS 2 Nav2 Client: {e}")

    def get_current_robot_pose(self) -> Dict[str, float]:
        return dict(self._current_robot_pose)

    def navigate_to_pose(
        self,
        target_pose: Dict[str, float],
        feedback_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        import rclpy
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            logger.error("Nav2 Action Server 'navigate_to_pose' not available.")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target_pose["x"])
        goal_msg.pose.pose.position.y = float(target_pose["y"])
        goal_msg.pose.pose.position.z = 0.0

        yaw = target_pose.get("yaw", 0.0)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        logger.info(f"[LiveNav2] Sending goal: x={target_pose['x']:.2f}, y={target_pose['y']:.2f}, yaw={yaw:.2f}")
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, send_goal_future)

        self._goal_handle = send_goal_future.result()
        if not self._goal_handle.accepted:
            logger.warning("[LiveNav2] Goal rejected by Nav2 server.")
            return False

        result_future = self._goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result()

        return result.status == 4  # SUCCEEDED status code in ROS 2 action

    def cancel_goal(self) -> None:
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
