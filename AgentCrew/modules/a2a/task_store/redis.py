from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

from .base import TaskStore


class RedisTaskStore(TaskStore):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "a2a_task",
        ttl: int = 3600,
    ):
        self.prefix = prefix
        self.ttl = ttl
        self._redis = None
        self._redis_url = redis_url

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    max_connections=20,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisTaskStore. "
                    "Install it with: uv add redis"
                )
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection pool and release all connections."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _key(self, namespace: str, id: str) -> str:
        return f"{self.prefix}:{namespace}:{id}"

    async def get_task(self, task_id: str) -> Optional[Task]:
        r = await self._get_redis()
        data = await r.get(self._key("task", task_id))
        if data is None:
            return None
        return Task.model_validate_json(data)

    async def save_task(self, task: Task) -> None:
        r = await self._get_redis()
        key = self._key("task", task.id)
        await r.set(key, task.model_dump_json(exclude_none=True), ex=self.ttl)

    async def delete_task(self, task_id: str) -> None:
        r = await self._get_redis()
        await r.delete(self._key("task", task_id))

    async def has_task(self, task_id: str) -> bool:
        r = await self._get_redis()
        return await r.exists(self._key("task", task_id)) > 0

    async def get_task_history(self, context_id: str) -> List[Dict[str, Any]]:
        r = await self._get_redis()
        data = await r.get(self._key("history", context_id))
        if data is None:
            return []
        return json.loads(data)

    async def save_task_history(
        self, context_id: str, history: List[Dict[str, Any]]
    ) -> None:
        r = await self._get_redis()
        key = self._key("history", context_id)
        await r.set(key, json.dumps(history, default=str), ex=self.ttl)

    async def append_task_history_message(
        self, context_id: str, message: Dict[str, Any]
    ) -> None:
        r = await self._get_redis()
        key = self._key("history", context_id)
        data = await r.get(key)
        history = json.loads(data) if data else []
        history.append(message)
        await r.set(key, json.dumps(history, default=str), ex=self.ttl)

    async def has_task_history(self, context_id: str) -> bool:
        r = await self._get_redis()
        return await r.exists(self._key("history", context_id)) > 0

    async def get_task_events(
        self, task_id: str
    ) -> List[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]]:
        r = await self._get_redis()
        data = await r.get(self._key("events", task_id))
        if data is None:
            return []
        raw_events = json.loads(data)
        return self.deserialize_events(raw_events)

    async def append_task_event(
        self,
        task_id: str,
        event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent],
    ) -> None:
        r = await self._get_redis()
        key = self._key("events", task_id)
        data = await r.get(key)
        events = json.loads(data) if data else []
        events.append(json.loads(event.model_dump_json(exclude_none=True)))
        await r.set(key, json.dumps(events), ex=self.ttl)

    async def cleanup_task(self, task_id: str) -> None:
        r = await self._get_redis()
        await r.delete(
            self._key("task", task_id),
            self._key("events", task_id),
        )
