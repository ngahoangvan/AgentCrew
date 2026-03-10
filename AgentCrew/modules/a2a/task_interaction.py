from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
    TextPart,
)

from .exceptions import TaskCanceledException

if TYPE_CHECKING:
    from typing import Dict
    from .task_cancellation import TaskCancellationManager


class TaskInteractionHandler:
    def __init__(self, cancellation: TaskCancellationManager) -> None:
        self.cancellation = cancellation
        self.pending_ask_responses: Dict[str, asyncio.Event] = {}
        self.ask_responses: Dict[str, str] = {}

    def create_ask_message(self, question: str, guided_answers: list[str]) -> Message:
        ask_data = {
            "type": "ask",
            "question": question,
            "guided_answers": guided_answers,
            "instruction": "Please respond with one of the guided answers or provide a custom response.",
        }
        return Message(
            message_id=f"ask_{hash(question)}",
            role=Role.agent,
            parts=[
                Part(root=TextPart(text=f"❓ {question}")),
                Part(root=DataPart(data=ask_data)),
            ],
        )

    async def wait_for_answer(self, task_id: str, timeout: int = 300) -> str:
        wait_event = asyncio.Event()
        self.pending_ask_responses[task_id] = wait_event
        try:
            await asyncio.wait_for(wait_event.wait(), timeout=timeout)
            if self.cancellation.is_canceled(task_id):
                raise TaskCanceledException(
                    f"Task {task_id} was canceled while waiting for input"
                )
            return self.ask_responses.get(task_id, "No response received")
        except asyncio.TimeoutError:
            return "User did not respond in time."
        finally:
            self.pending_ask_responses.pop(task_id, None)
            self.ask_responses.pop(task_id, None)

    def submit_answer(self, task_id: str, answer: str) -> None:
        self.ask_responses[task_id] = answer
        if task_id in self.pending_ask_responses:
            self.pending_ask_responses[task_id].set()

    def cancel_pending(self, task_id: str) -> None:
        if task_id in self.pending_ask_responses:
            self.pending_ask_responses[task_id].set()

    def cleanup(self, task_id: str) -> None:
        self.pending_ask_responses.pop(task_id, None)
        self.ask_responses.pop(task_id, None)
