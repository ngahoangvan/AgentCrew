from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)


class TaskStore(ABC):
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        pass

    @abstractmethod
    async def save_task(self, task: Task) -> None:
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> None:
        pass

    @abstractmethod
    async def has_task(self, task_id: str) -> bool:
        pass

    @abstractmethod
    async def get_task_history(self, context_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def save_task_history(
        self, context_id: str, history: List[Dict[str, Any]]
    ) -> None:
        pass

    @abstractmethod
    async def append_task_history_message(
        self, context_id: str, message: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    async def has_task_history(self, context_id: str) -> bool:
        pass

    @abstractmethod
    async def get_task_events(
        self, task_id: str
    ) -> List[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]]:
        pass

    @abstractmethod
    async def append_task_event(
        self,
        task_id: str,
        event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent],
    ) -> None:
        pass

    @abstractmethod
    async def cleanup_task(self, task_id: str) -> None:
        pass

    async def close(self) -> None:
        """Release any resources held by the store (e.g. connection pools).
        Default is a no-op; override in stores that manage connections.
        """
        pass

    @staticmethod
    def deserialize_events(
        raw_events: List[Dict[str, Any]],
    ) -> List[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]]:
        events: List[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]] = []
        for raw in raw_events:
            if "artifact" in raw:
                events.append(TaskArtifactUpdateEvent.model_validate(raw))
            else:
                events.append(TaskStatusUpdateEvent.model_validate(raw))
        return events
