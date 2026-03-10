from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from loguru import logger

from a2a.types import (
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)

if TYPE_CHECKING:
    from typing import Dict, Union
    from .task_store import TaskStore


class TaskStreamingManager:
    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self.streaming_tasks: Dict[str, asyncio.Queue] = {}
        self.streaming_enabled_tasks: set[str] = set()

    def enable_streaming(self, task_id: str) -> asyncio.Queue:
        self.streaming_enabled_tasks.add(task_id)
        queue: asyncio.Queue = asyncio.Queue()
        self.streaming_tasks[task_id] = queue
        return queue

    def is_streaming_enabled(self, task_id: str) -> bool:
        return task_id in self.streaming_enabled_tasks

    async def record_and_emit_event(
        self,
        task_id: str,
        event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent],
    ) -> None:
        await self.store.append_task_event(task_id, event)
        for key, queue in list(self.streaming_tasks.items()):
            if key.startswith(task_id):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for {key}")
                except Exception as e:
                    logger.error(f"Error emitting event to {key}: {e}")

    async def signal_end(self, task_id: str) -> None:
        for key in list(self.streaming_tasks.keys()):
            if key.startswith(task_id):
                await self.streaming_tasks[key].put(None)

    async def signal_cancel(self, task_id: str, task: Task) -> None:
        canceled_status = TaskStatus(
            state=TaskState.canceled,
            timestamp=task.status.timestamp,
        )
        cancel_event = TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=task.context_id,
            status=canceled_status,
            final=True,
        )
        for key in list(self.streaming_tasks.keys()):
            if key.startswith(task_id):
                queue = self.streaming_tasks[key]
                await queue.put(cancel_event)
                await queue.put(None)

    def drain_nowait(self, task_id: str) -> None:
        for key in list(self.streaming_tasks.keys()):
            if key.startswith(task_id):
                try:
                    self.streaming_tasks[key].put_nowait(None)
                except Exception:
                    pass

    def remove(self, task_id: str) -> None:
        self.streaming_tasks.pop(task_id, None)

    def register_subscriber(self, key: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.streaming_tasks[key] = queue
        return queue

    def remove_subscriber(self, key: str) -> None:
        self.streaming_tasks.pop(key, None)

    def cleanup(self, task_id: str) -> None:
        self.streaming_enabled_tasks.discard(task_id)
        self.streaming_tasks.pop(task_id, None)
