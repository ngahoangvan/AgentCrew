from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from a2a.types import (
    Task,
    TaskState,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from AgentCrew.modules.agents.base import MessageType
from .adapters import convert_agent_response_to_a2a_artifact
from .exceptions import TaskCanceledException

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from AgentCrew.modules.agents import LocalAgent
    from .task_store import TaskStore
    from .task_streaming import TaskStreamingManager
    from .task_cancellation import TaskCancellationManager
    from .task_interaction import TaskInteractionHandler


class TaskExecutionEngine:
    def __init__(
        self,
        agent_name: str,
        store: TaskStore,
        streaming: TaskStreamingManager,
        cancellation: TaskCancellationManager,
        interaction: TaskInteractionHandler,
        memory_service: Any = None,
    ) -> None:
        self.agent_name = agent_name
        self.store = store
        self.streaming = streaming
        self.cancellation = cancellation
        self.interaction = interaction
        self.memory_service = memory_service

    async def run(self, agent: LocalAgent, task: Task) -> None:
        if task.status.state in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
        }:
            logger.warning(
                f"Attempted to process task {task.id} in terminal state {task.status.state}"
            )
            return

        try:
            artifacts: List[Any] = []
            if not await self.store.has_task_history(task.context_id):
                raise ValueError("Task history is not existed")

            task_history = await self.store.get_task_history(task.context_id)
            retried_count = [0]

            current_response = await self._process_task(
                agent, task, task_history, artifacts, retried_count
            )

            if self.cancellation.is_canceled(task.id):
                return

            await self._finalize_task(
                agent, task, current_response, artifacts, task_history
            )

        except TaskCanceledException:
            logger.info(f"Task {task.id} canceled during processing")
            if self.streaming.is_streaming_enabled(task.id):
                self.streaming.drain_nowait(task.id)
        except asyncio.CancelledError:
            logger.info(f"Task {task.id} asyncio task cancelled externally")
            raise
        except Exception as e:
            logger.error(str(e))
            task_history = await self.store.get_task_history(task.context_id)
            logger.debug(task_history)
            task.status.state = TaskState.failed
            task.status.timestamp = datetime.now().isoformat()
            await self.store.save_task(task)
            if self.streaming.is_streaming_enabled(task.id):
                await self.streaming.record_and_emit_event(
                    task.id,
                    TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    ),
                )
                await self.streaming.signal_end(task.id)
        finally:
            self.cancellation.cleanup(task.id)
            self.streaming.cleanup(task.id)
            self.interaction.cleanup(task.id)

    async def _process_task(
        self,
        agent: LocalAgent,
        task: Task,
        task_history: List[Dict[str, Any]],
        artifacts: List[Any],
        retried_count: List[int],
    ) -> str:
        try:
            current_response = ""
            response_message = ""
            thinking_content = ""
            thinking_signature = ""
            tool_uses: List[Dict[str, Any]] = []
            input_tokens = 0
            output_tokens = 0

            def process_result(_tool_uses, _input_tokens, _output_tokens):
                nonlocal tool_uses, input_tokens, output_tokens
                tool_uses = _tool_uses
                input_tokens += _input_tokens
                output_tokens += _output_tokens

            async for (
                response_message,
                chunk_text,
                thinking_chunk,
            ) in agent.process_messages(task_history, callback=process_result):
                if response_message:
                    current_response = response_message

                task.status.state = TaskState.working
                task.status.timestamp = datetime.now().isoformat()

                if thinking_chunk:
                    think_text_chunk, signature = thinking_chunk
                    if think_text_chunk:
                        thinking_content += think_text_chunk
                    if signature:
                        thinking_signature += signature

                if self.streaming.is_streaming_enabled(task.id):
                    await self._handle_streaming_chunk(
                        task, chunk_text, thinking_chunk, artifacts
                    )

                if self.cancellation.is_canceled(task.id):
                    raise TaskCanceledException(
                        f"Task {task.id} was canceled during streaming"
                    )

            if self.cancellation.is_canceled(task.id):
                raise TaskCanceledException(
                    f"Task {task.id} was canceled after streaming"
                )

            if tool_uses:
                if self.streaming.is_streaming_enabled(task.id):
                    tool_artifact = convert_agent_response_to_a2a_artifact(
                        "",
                        artifact_id=f"artifact_{task.id}_{len(artifacts)}",
                        tool_uses=tool_uses,
                    )
                    await self.streaming.record_and_emit_event(
                        task.id,
                        TaskArtifactUpdateEvent(
                            task_id=task.id,
                            context_id=task.context_id,
                            artifact=tool_artifact,
                        ),
                    )
                    await asyncio.sleep(0.7)

                thinking_data: Optional[Tuple[str, str]] = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = agent.format_message(
                    MessageType.Thinking, {"thinking": thinking_data}
                )
                if thinking_message:
                    await self._append_history_message(
                        task.context_id, thinking_message, task_history
                    )

                assistant_message = agent.format_message(
                    MessageType.Assistant,
                    {
                        "message": response_message,
                        "tool_uses": [t for t in tool_uses if t["name"] != "transfer"],
                    },
                )
                if assistant_message:
                    await self._append_history_message(
                        task.context_id, assistant_message, task_history
                    )

                await self._execute_tool_calls(agent, task, tool_uses, task_history)

                return await self._process_task(
                    agent, task, task_history, artifacts, retried_count
                )

            return current_response

        except Exception as e:
            if isinstance(e, TaskCanceledException):
                raise
            from openai import BadRequestError

            if isinstance(e, BadRequestError):
                if (
                    e.code == "model_max_prompt_tokens_exceeded"
                    and retried_count[0] < 5
                ):
                    from AgentCrew.modules.agents import LocalAgent as _LocalAgent
                    from AgentCrew.modules.llm.model_registry import ModelRegistry

                    if isinstance(agent, _LocalAgent):
                        max_token = ModelRegistry.get_model_limit(agent.get_model())
                        agent.input_tokens_usage = max_token
                        retried_count[0] += 1
                        return await self._process_task(
                            agent, task, task_history, artifacts, retried_count
                        )
            raise

    async def _handle_streaming_chunk(
        self,
        task: Task,
        chunk_text: Optional[str],
        thinking_chunk: Any,
        artifacts: List[Any],
    ) -> None:
        if thinking_chunk:
            think_text_chunk, signature = thinking_chunk
            if think_text_chunk:
                thinking_artifact = convert_agent_response_to_a2a_artifact(
                    think_text_chunk,
                    artifact_id=f"thinking_{task.id}_{datetime.now()}",
                )
                await self.streaming.record_and_emit_event(
                    task.id,
                    TaskArtifactUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        artifact=thinking_artifact,
                    ),
                )

        if chunk_text:
            artifact = convert_agent_response_to_a2a_artifact(
                chunk_text,
                artifact_id=f"artifact_{task.id}_{len(artifacts)}",
            )
            await self.streaming.record_and_emit_event(
                task.id,
                TaskArtifactUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    artifact=artifact,
                ),
            )

    async def _execute_tool_calls(
        self,
        agent: LocalAgent,
        task: Task,
        tool_uses: List[Dict[str, Any]],
        task_history: List[Dict[str, Any]],
    ) -> None:
        for tool_use in tool_uses:
            if self.cancellation.is_canceled(task.id):
                raise TaskCanceledException(
                    f"Task {task.id} was canceled during tool execution"
                )
            tool_name = tool_use["name"]

            if tool_name == "ask":
                await self._handle_ask_tool(agent, task, tool_use, task_history)
            else:
                try:
                    tool_result = await agent.execute_tool_call(
                        tool_name, tool_use["input"]
                    )
                    tool_result_message = agent.format_message(
                        MessageType.ToolResult,
                        {"tool_use": tool_use, "tool_result": tool_result},
                    )
                    if tool_result_message:
                        await self._append_history_message(
                            task.context_id, tool_result_message, task_history
                        )
                except Exception as e:
                    error_message = agent.format_message(
                        MessageType.ToolResult,
                        {
                            "tool_use": tool_use,
                            "tool_result": str(e),
                            "is_error": True,
                        },
                    )
                    if error_message:
                        await self._append_history_message(
                            task.context_id, error_message, task_history
                        )

    async def _handle_ask_tool(
        self,
        agent: LocalAgent,
        task: Task,
        tool_use: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> None:
        question = tool_use["input"].get("question", "")
        guided_answers = tool_use["input"].get("guided_answers", [])

        task.status.state = TaskState.input_required
        task.status.timestamp = datetime.now().isoformat()
        task.status.message = self.interaction.create_ask_message(
            question, guided_answers
        )

        await self.store.save_task(task)
        await self.streaming.record_and_emit_event(
            task.id,
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=task.status,
                final=False,
            ),
        )

        user_answer = await self.interaction.wait_for_answer(task.id)

        if self.cancellation.is_canceled(task.id):
            raise TaskCanceledException(
                f"Task {task.id} was canceled while waiting for input"
            )

        tool_result = f"User's answer: {user_answer}"

        task.status.state = TaskState.working
        task.status.timestamp = datetime.now().isoformat()
        task.status.message = None

        tool_result_message = agent.format_message(
            MessageType.ToolResult,
            {"tool_use": tool_use, "tool_result": tool_result},
        )
        if tool_result_message:
            await self._append_history_message(
                task.context_id, tool_result_message, task_history
            )

        await self.streaming.record_and_emit_event(
            task.id,
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=task.status,
                final=False,
            ),
        )

    async def _finalize_task(
        self,
        agent: LocalAgent,
        task: Task,
        current_response: str,
        artifacts: List[Any],
        task_history: List[Dict[str, Any]],
    ) -> None:
        if current_response.strip():
            assistant_message = agent.format_message(
                MessageType.Assistant, {"message": current_response}
            )
            if assistant_message:
                await self.store.append_task_history_message(
                    task.context_id, assistant_message
                )
            if self.memory_service:
                user_message = task_history[0].get("content", [{}])[0].get("text", "")
                self.memory_service.store_conversation(
                    user_message, current_response, self.agent_name
                )

        final_artifact = convert_agent_response_to_a2a_artifact(
            current_response, artifact_id=f"artifact_{task.id}_final"
        )
        artifacts.append(final_artifact)

        task.status.state = TaskState.completed
        task.status.timestamp = datetime.now().isoformat()
        task.artifacts = artifacts
        await self.store.save_task(task)

        if self.streaming.is_streaming_enabled(task.id):
            await self.streaming.record_and_emit_event(
                task.id,
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    status=task.status,
                    final=True,
                ),
            )
            await self.streaming.signal_end(task.id)

    async def _append_history_message(
        self,
        context_id: str,
        message: Dict[str, Any],
        task_history: List[Dict[str, Any]],
    ) -> None:
        await self.store.append_task_history_message(context_id, message)
        task_history.append(message)
