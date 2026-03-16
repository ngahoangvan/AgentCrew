from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING
import tempfile
import os

from a2a.types import (
    CancelTaskResponse,
    JSONRPCError,
    GetTaskResponse,
    GetTaskSuccessResponse,
    JSONRPCErrorResponse,
    SendMessageResponse,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    CancelTaskSuccessResponse,
    SetTaskPushNotificationConfigResponse,
    GetTaskPushNotificationConfigResponse,
    SendMessageSuccessResponse,
    Task,
    TaskStatus,
    TaskState,
    TaskStatusUpdateEvent,
)

from AgentCrew.modules.agents import LocalAgent
from AgentCrew.modules.agents.base import MessageType
from .adapters import convert_a2a_message_to_agent
from .common.server.task_manager import TaskManager
from .errors import A2AError
from .task_store import TaskStore
from .task_cancellation import TaskCancellationManager
from .task_streaming import TaskStreamingManager
from .task_interaction import TaskInteractionHandler
from .task_execution import TaskExecutionEngine

if TYPE_CHECKING:
    from typing import Any, AsyncIterable, Dict, Optional, Union
    from AgentCrew.modules.agents import AgentManager
    from a2a.types import (
        CancelTaskRequest,
        TaskNotCancelableError,
        GetTaskPushNotificationConfigRequest,
        GetTaskRequest,
        SendMessageRequest,
        SendStreamingMessageRequest,
        SetTaskPushNotificationConfigRequest,
        TaskResubscriptionRequest,
        JSONRPCResponse,
    )


TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed}
INPUT_REQUIRED_STATES = {TaskState.input_required}


class AgentTaskManager(TaskManager):
    def __init__(self, agent_name: str, agent_manager: AgentManager, store: TaskStore):
        self.agent_name = agent_name
        self.agent_manager = agent_manager
        self.store = store
        self.file_handler = None

        self.agent = self.agent_manager.get_agent(self.agent_name)
        if self.agent is None or not isinstance(self.agent, LocalAgent):
            raise ValueError(f"Agent {agent_name} not found or is not a LocalAgent")

        memory_service = self.agent.services.get("memory", None)

        self.cancellation = TaskCancellationManager()
        self.streaming = TaskStreamingManager(store)
        self.interaction = TaskInteractionHandler(self.cancellation)
        self.execution = TaskExecutionEngine(
            agent_name,
            store,
            self.streaming,
            self.cancellation,
            self.interaction,
            memory_service,
        )

    def _is_terminal_state(self, state: TaskState) -> bool:
        return state in TERMINAL_STATES

    def _validate_task_not_terminal(
        self, task: Task, operation: str
    ) -> Optional[TaskNotCancelableError]:
        if self._is_terminal_state(task.status.state):
            return A2AError.task_not_cancelable(task.id, task.status.state.value)
        return None

    def _extract_text_from_message(self, message: Dict[str, Any]) -> str:
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)

    async def on_send_message(
        self, request: SendMessageRequest | SendStreamingMessageRequest
    ) -> SendMessageResponse:
        if not self.agent or not isinstance(self.agent, LocalAgent):
            return SendMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=-32001, message=f"Agent {self.agent_name} not found"
                    ),
                )
            )

        task_id = (
            request.params.message.task_id
            or f"task_{request.params.message.message_id}"
        )

        existing_task = await self.store.get_task(task_id)
        if existing_task:
            error = self._validate_task_not_terminal(existing_task, "send message")
            if error:
                return SendMessageResponse(
                    root=JSONRPCErrorResponse(id=request.id, error=error)
                )

            if existing_task.status.state in INPUT_REQUIRED_STATES:
                message = convert_a2a_message_to_agent(request.params.message)
                user_response = self._extract_text_from_message(message)
                self.interaction.submit_answer(task_id, user_response)
                return SendMessageResponse(
                    root=SendMessageSuccessResponse(id=request.id, result=existing_task)
                )

        task = existing_task
        if not task:
            task = Task(
                id=task_id,
                context_id=request.params.message.context_id or f"ctx_{task_id}",
                status=TaskStatus(
                    state=TaskState.working, timestamp=datetime.now().isoformat()
                ),
            )
            await self.store.save_task(task)

        if not await self.store.has_task_history(task.context_id):
            await self.store.save_task_history(task.context_id, [])

        message = convert_a2a_message_to_agent(request.params.message)
        if next(
            (m for m in message.get("content", []) if m.get("type", "text") == "file"),
            None,
        ):
            from AgentCrew.modules.chat.file_handler import FileHandler

            new_parts = []
            if self.file_handler is None:
                self.file_handler = FileHandler()
            for part in message.get("content", []):
                if part.get("type") == "file":
                    temp_file = os.path.join(tempfile.gettempdir(), part["file_name"])
                    with open(temp_file, "wb") as f:
                        f.write(part["file_data"])
                    file_part = self.file_handler.process_file(temp_file)
                    if not file_part:
                        file_part = self.agent.format_message(
                            MessageType.FileContent, {"file_uri": temp_file}
                        )
                    if file_part:
                        new_parts.append(file_part)
                    else:
                        new_parts.append(
                            {
                                "type": "text",
                                "text": f"[Unsupported file: {part['file_name']}]",
                            }
                        )
                else:
                    new_parts.append(part)
            message["content"] = new_parts

        await self.store.append_task_history_message(task.context_id, message)

        bg_task = asyncio.create_task(self.execution.run(self.agent, task))
        self.cancellation.register(task_id, bg_task)

        return SendMessageResponse(
            root=SendMessageSuccessResponse(id=request.id, result=task)
        )

    async def on_send_message_streaming(
        self, request: SendStreamingMessageRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        task_id = (
            request.params.message.task_id
            or f"task_{request.params.message.message_id}"
        )

        queue = self.streaming.enable_streaming(task_id)

        try:
            response = await self.on_send_message(request)

            if isinstance(response.root, JSONRPCErrorResponse):
                yield SendStreamingMessageResponse(root=response.root)
                return

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id=request.id, result=event
                    )
                )
        finally:
            self.streaming.remove(task_id)

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        task_id = request.params.id
        task = await self.store.get_task(task_id)
        if not task:
            return GetTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=A2AError.task_not_found(task_id)
                )
            )
        return GetTaskResponse(root=GetTaskSuccessResponse(id=request.id, result=task))

    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        task_id = request.params.id
        task = await self.store.get_task(task_id)
        if not task:
            return CancelTaskResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=A2AError.task_not_found(task_id)
                )
            )

        error = self._validate_task_not_terminal(task, "cancel")
        if error:
            return CancelTaskResponse(
                root=JSONRPCErrorResponse(id=request.id, error=error)
            )

        self.cancellation.signal_cancel(task_id)
        self.interaction.cancel_pending(task_id)

        task.status.state = TaskState.canceled
        task.status.timestamp = datetime.now().isoformat()
        await self.store.save_task(task)

        await self.streaming.signal_cancel(task_id, task)

        return CancelTaskResponse(
            root=CancelTaskSuccessResponse(id=request.id, result=task)
        )

    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationConfigRequest
    ) -> SetTaskPushNotificationConfigResponse:
        return SetTaskPushNotificationConfigResponse(
            root=JSONRPCErrorResponse(
                id=request.id, error=A2AError.push_notification_not_supported()
            )
        )

    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationConfigRequest
    ) -> GetTaskPushNotificationConfigResponse:
        return GetTaskPushNotificationConfigResponse(
            root=JSONRPCErrorResponse(
                id=request.id, error=A2AError.push_notification_not_supported()
            )
        )

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        task_id = request.params.id

        task = await self.store.get_task(task_id)
        if not task:
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id, error=A2AError.task_not_found(task_id)
                )
            )
            return

        stored_events = await self.store.get_task_events(task_id)
        for event in stored_events:
            yield SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(id=request.id, result=event)
            )

        if self._is_terminal_state(task.status.state):
            return

        if not self.streaming.is_streaming_enabled(task_id):
            yield SendStreamingMessageResponse(
                root=JSONRPCErrorResponse(
                    id=request.id,
                    error=A2AError.unsupported_operation(
                        "Task was not created with streaming enabled"
                    ),
                )
            )
            return

        resubscribe_key = f"{task_id}_resub_{request.id}"
        queue = self.streaming.register_subscriber(resubscribe_key)

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break

                yield SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id=request.id, result=event
                    )
                )

                if isinstance(event, TaskStatusUpdateEvent):
                    if self._is_terminal_state(event.status.state):
                        break
        finally:
            self.streaming.remove_subscriber(resubscribe_key)

    async def on_send_task(self, request: SendMessageRequest) -> SendMessageResponse:
        return await self.on_send_message(request)

    async def cleanup_terminal_tasks(self) -> None:
        task_ids = await self.store.list_task_ids()
        for task_id in task_ids:
            task = await self.store.get_task(task_id)
            if task and self._is_terminal_state(task.status.state):
                await self.store.cleanup_task(task_id)
                self.streaming.cleanup(task_id)

    async def on_send_task_subscribe(
        self, request: SendStreamingMessageRequest
    ) -> Union[AsyncIterable[SendStreamingMessageResponse], JSONRPCResponse]:
        return await self.on_send_message_streaming(request)


class MultiAgentTaskManager:
    def __init__(
        self,
        agent_manager: AgentManager,
        store_type: str = "memory",
        store_options: Optional[Dict[str, Any]] = None,
    ):
        from .task_store import create_task_store

        self.agent_manager = agent_manager
        self.agent_task_managers: Dict[str, AgentTaskManager] = {}

        for agent_name in agent_manager.agents:
            store = create_task_store(store_type, **(store_options or {}))
            self.agent_task_managers[agent_name] = AgentTaskManager(
                agent_name, agent_manager, store
            )

    def get_task_manager(self, agent_name: str) -> Optional[AgentTaskManager]:
        return self.agent_task_managers.get(agent_name)

    async def initialize(self) -> None:
        for manager in self.agent_task_managers.values():
            await manager.cleanup_terminal_tasks()

    async def close(self) -> None:
        for manager in self.agent_task_managers.values():
            await manager.store.close()
