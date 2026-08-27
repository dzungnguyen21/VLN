from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import threading


class SubgoalStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    REACHED = "REACHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SubgoalItem(BaseModel):
    id: int
    description: str
    target_landmark: str
    status: SubgoalStatus = SubgoalStatus.PENDING
    target_3d_pose: Optional[Dict[str, float]] = None  # {"x": ..., "y": ..., "z": ..., "yaw": ...}
    attempts: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubgoalQueue:
    """
    Thread-safe Subgoal Queue to manage sequential subgoal decomposition
    and execution state transitions.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._queue: List[SubgoalItem] = []
        self._current_index: int = -1

    def load_subgoals(self, subgoals: List[Dict[str, Any]]) -> None:
        """Populate the queue from a list of decomposed subgoal dictionaries."""
        with self._lock:
            self._queue.clear()
            for idx, item in enumerate(subgoals):
                subgoal_obj = SubgoalItem(
                    id=item.get("id", idx + 1),
                    description=item.get("description", item.get("subgoal", "")),
                    target_landmark=item.get("target_landmark", item.get("target", item.get("description", ""))),
                    metadata=item.get("metadata", {}),
                )
                self._queue.append(subgoal_obj)
            self._current_index = -1

    def enqueue(self, item: SubgoalItem) -> None:
        with self._lock:
            self._queue.append(item)

    def get_next_subgoal(self) -> Optional[SubgoalItem]:
        """Advance to and return the next pending subgoal."""
        with self._lock:
            for i, item in enumerate(self._queue):
                if item.status == SubgoalStatus.PENDING:
                    self._current_index = i
                    item.status = SubgoalStatus.ACTIVE
                    item.attempts += 1
                    return item
            return None

    def get_current_subgoal(self) -> Optional[SubgoalItem]:
        """Return currently active subgoal without advancing."""
        with self._lock:
            if 0 <= self._current_index < len(self._queue):
                return self._queue[self._current_index]
            return None

    def mark_current_reached(self) -> Optional[SubgoalItem]:
        """Mark the active subgoal as REACHED."""
        with self._lock:
            if 0 <= self._current_index < len(self._queue):
                item = self._queue[self._current_index]
                item.status = SubgoalStatus.REACHED
                return item
            return None

    def mark_current_failed(self) -> Optional[SubgoalItem]:
        """Mark the active subgoal as FAILED."""
        with self._lock:
            if 0 <= self._current_index < len(self._queue):
                item = self._queue[self._current_index]
                item.status = SubgoalStatus.FAILED
                return item
            return None

    def has_more_subgoals(self) -> bool:
        """Check if any pending subgoals remain in queue."""
        with self._lock:
            return any(item.status == SubgoalStatus.PENDING for item in self._queue)

    def is_all_completed(self) -> bool:
        """Check if all subgoals have reached completion."""
        with self._lock:
            if not self._queue:
                return False
            return all(item.status == SubgoalStatus.REACHED for item in self._queue)

    def to_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [item.model_dump() for item in self._queue]

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._current_index = -1

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
