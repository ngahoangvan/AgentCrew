from typing import Tuple, Optional, List
import asyncio
import os
import re
import shlex
import traceback

from loguru import logger
from AgentCrew.modules.agents.base import MessageType
from AgentCrew.modules.chat.history import ChatHistoryManager
from AgentCrew.modules.agents import AgentManager
from AgentCrew.modules.utils.file_handler import FileHandler

from AgentCrew.modules.memory import (
    BaseMemoryService,
    ContextPersistenceService,
)
from AgentCrew.modules.mcpclient import MCPSessionManager
from .command_processor import CommandProcessor
from .tool_manager import ToolManager
from .conversation import ConversationManager
from .base import Observable


_AT_AGENT_RE = re.compile(r"@([\.\w-]+)")


def _resolve_at_mention(user_input: str, agent_manager) -> tuple:
    match = _AT_AGENT_RE.search(user_input)
    if match:
        target = match.group(1)
        if target in agent_manager.agents:
            llm_content = (
                f"<Tag_Action>Transfer to {target} with the user request: "
                f"{user_input}</Tag_Action>"
            )
            return user_input, llm_content
    return user_input, user_input


class MessageHandler(Observable):
    """
    Handles message processing, interaction with the LLM service, and manages
    conversation history. Uses the Observer pattern to notify UI components
    about relevant events.
    """

    def __init__(
        self,
        memory_service: Optional[BaseMemoryService] = None,
        context_persistent_service: Optional[ContextPersistenceService] = None,
        with_voice: bool = False,
    ):
        """
        Initializes the MessageHandler.

        Args:
            memory_service: Memory service for storing conversations.
            context_persistent_service: Service for persistent conversation storage.
        """
        super().__init__()
        self.agent_manager = AgentManager.get_instance()
        self.mcp_manager = MCPSessionManager.get_instance()
        self.agent = self.agent_manager.get_current_agent()
        self.memory_service = memory_service
        self.persistent_service = context_persistent_service
        self.history_manager = ChatHistoryManager()
        self.latest_assistant_response = ""
        self.conversation_turns = []
        self.current_user_input = None
        self.current_user_input_idx = -1
        self.last_assisstant_response_idx = -1
        self.file_handler: Optional[FileHandler] = None
        self._queued_attached_files = []
        self.stop_streaming = False
        self.streamline_messages = []
        self.current_conversation_id: Optional[str] = None  # ID for persistence
        self.pending_evolution_proposal: Optional[dict] = None

        # Initialize components
        self.command_processor = CommandProcessor(self)
        self.tool_manager = ToolManager(self)
        self.conversation_manager = ConversationManager(self)

        self.conversation_manager.start_new_conversation()  # Initialize first conversation
        self._yolo_mode_check()

        # Voice integration
        self.voice_service = None
        # Check if voice service is available
        from AgentCrew.modules.voice import AUDIO_AVAILABLE

        if AUDIO_AVAILABLE and with_voice:
            if os.getenv("ELEVENLABS_API_KEY"):
                from AgentCrew.modules.voice.elevenlabs_service import (
                    ElevenLabsVoiceService,
                )

                self.voice_service = ElevenLabsVoiceService()
            elif os.getenv("DEEPINFRA_API_KEY"):
                from AgentCrew.modules.voice.deepinfra_service import (
                    DeepInfraVoiceService,
                )

                self.voice_service = DeepInfraVoiceService()

    def _yolo_mode_check(self):
        from AgentCrew.modules.config.global_config import GlobalConfig

        global_config = GlobalConfig().read()
        self.tool_manager.yolo_mode = global_config.get("global_settings", {}).get(
            "yolo_mode", False
        )

    def _messages_append(self, message):
        """Append a message to the agent history and streamline messages."""
        self.streamline_messages.append(message)

        self.agent.append_message(message)

    def _prepare_files_processing(self, file_command):
        file_paths_str: str = file_command[6:].strip()
        file_paths: List[str] = [
            os.path.expanduser(path.strip())
            for path in shlex.split(file_paths_str)
            if path.strip()
        ]

        for file_path in file_paths:
            self._queued_attached_files.append(file_path)
            self._notify("file_processing", {"file_path": file_path})

    async def process_user_input(
        self,
        user_input: str,
    ) -> Tuple[bool, bool]:
        """
        Processes user input, handles commands, and updates message history.

        Args:
            user_input: The input string from the user.

        Returns:
            Tuple of (exit_flag, clear_flag)
        """
        self.history_manager.add_entry(user_input)

        if user_input.startswith("/file "):
            self._prepare_files_processing(user_input)
            return False, True
        if user_input.startswith("/retry"):
            return False, False

        # Process commands first
        command_result = await self.command_processor.process_command(user_input)
        if command_result.handled:
            return command_result.exit_flag, command_result.clear_flag

        # Handle regular user input (non-commands)
        # RAG base on user query
        # IMPORTANT: this actually add more problems than it solves, so it's disabled for now
        #         if await self.memory_service.need_generate_user_context(user_input):
        #             self._notify("user_context_request", None)
        #             self._messages_append(
        #                 {
        #                     "role": "user",
        #                     "content": [
        #                         {
        #                             "type": "text",
        #                             "text": f"""Memories related to the user request:
        # ---
        # {await self.memory_service.generate_user_context(user_input, self.agent.name)}
        # ---
        # INSTRUCTIONS:
        # 1. EXTRACT relevant facts and context from memories
        # 2. PRIORITIZE recent memories when information conflicts
        # 3. INCORPORATE memory insights to enhance your response
        # 4. PRESERVE the user's original intent - memories should supplement, not override
        # 5. RESPOND directly to the current request first and foremost
        #
        # IMPORTANT: Memory serves to enhance responses, never to reinterpret or redirect the user's explicit request.""",
        #                         }
        #                     ],
        #                 }
        #             )

        # Delays file processing until user send message

        while len(self._queued_attached_files) > 0:
            file_command = self._queued_attached_files.pop(0)
            await self.command_processor.process_command(
                f"/file {shlex.quote(file_command)}"
            )

        # Add regular text message
        display_text, llm_content = _resolve_at_mention(user_input, self.agent_manager)

        self._messages_append(
            {
                "role": "user",
                "agent": self.agent.name,
                "content": [{"type": "text", "text": llm_content}],
            }
        )
        self.current_user_input = self.agent.history[-1]
        self.current_user_input_idx = len(self.streamline_messages) - 1
        self._notify(
            "user_message_created",
            {
                "message": self.agent.history[-1],
                "display_text": display_text,
                "with_files": False,
            },
        )

        return False, False

    def start_new_conversation(self):
        """Starts a new persistent conversation."""
        # Reset approved tools for the new conversation
        self.tool_manager.reset_approved_tools()
        self.conversation_manager.start_new_conversation()

    def resolve_tool_confirmation(self, confirmation_id, result):
        """
        Resolve a pending tool confirmation future with the user's decision.
        """
        self.tool_manager.resolve_tool_confirmation(confirmation_id, result)

    async def start_evolution_review(self) -> bool:
        from AgentCrew.modules.agents.prompt_evolution_service import (
            PromptEvolutionService,
        )

        self.prompt_evolution_service = PromptEvolutionService(
            memory_service=self.memory_service,
            persistence_service=self.persistent_service,
        )

        if not isinstance(
            self.agent,
            AgentManager.get_instance().get_local_agent(self.agent.name).__class__,
        ):
            self._notify("error", "/evolve is only supported with LocalAgent.")
            return True

        self._notify(
            "evolution_started",
            {"agent_name": self.agent.name},
        )
        try:
            proposal = await self.prompt_evolution_service.create_evolution_proposal(
                self.agent
            )
        except Exception as e:
            self._notify("evolution_finished", None)
            self._notify("error", f"Prompt evolution failed: {str(e)}")
            return True

        self.pending_evolution_proposal = proposal
        self._notify("evolution_summary_ready", proposal)
        return True

    async def approve_pending_evolution(self) -> bool:
        proposal = self.pending_evolution_proposal
        if not proposal:
            self._notify("error", "No pending evolution proposal to accept.")
            return True
        return await self._apply_pending_evolution(
            proposal.get("approved_summary")
            or proposal.get("generated_summary")
            or proposal.get("user_editable_summary", ""),
            edited_by_user=False,
        )

    async def edit_and_approve_pending_evolution(self, approved_summary: str) -> bool:
        proposal = self.pending_evolution_proposal
        if not proposal:
            self._notify("error", "No pending evolution proposal to edit.")
            return True

        approved_summary = approved_summary.strip()
        if not approved_summary:
            self._notify("error", "Edited evolution summary cannot be empty.")
            return True

        proposal["approved_summary"] = approved_summary
        proposal["user_editable_summary"] = approved_summary
        return await self._apply_pending_evolution(
            approved_summary,
            edited_by_user=True,
        )

    async def decline_pending_evolution(self) -> bool:
        if not self.pending_evolution_proposal:
            self._notify("error", "No pending evolution proposal to decline.")
            return True
        self.pending_evolution_proposal = None
        self._notify("evolution_declined", None)
        self._notify("system_message", "Prompt evolution declined.")
        return True

    async def _apply_pending_evolution(
        self, approved_summary: str, edited_by_user: bool
    ) -> bool:
        proposal = self.pending_evolution_proposal
        if not proposal:
            self._notify("error", "No pending evolution proposal to apply.")
            return True

        if not isinstance(
            self.agent,
            AgentManager.get_instance().get_local_agent(self.agent.name).__class__,
        ):
            self._notify("error", "/evolve is only supported with LocalAgent.")
            return True

        self._notify(
            "evolution_started",
            {"agent_name": self.agent.name},
        )
        try:
            revised_prompt = await self.prompt_evolution_service.build_revised_prompt(
                self.agent,
                approved_summary,
            )
            result = self.prompt_evolution_service.apply_prompt_revision(
                self.agent,
                revised_prompt,
                approved_summary,
                generated_summary=proposal.get("generated_summary")
                or proposal.get("user_editable_summary", ""),
                memory_ids=proposal.get("memory_ids", []),
                edited_by_user=edited_by_user,
            )
        except Exception as e:
            self._notify("error", f"Prompt evolution failed: {str(e)}")
            return True
        finally:
            self._notify("evolution_finished", None)

        self.pending_evolution_proposal = None
        self._notify("evolution_applied", result)
        self._notify(
            "system_message",
            f"Updated persisted system prompt for {result['agent_name']}.",
        )
        return True

    async def get_assistant_response(
        self, input_tokens=0, output_tokens=0
    ) -> Tuple[Optional[str], int, int]:
        """
        Stream the assistant's response and return the response and token usage.

        Returns:
            Tuple of (assistant_response, input_tokens, output_tokens)
        """
        assistant_response = ""
        tool_uses = []
        thinking_content = ""  # Reset thinking content for new response
        thinking_signature = ""  # Store the signature
        start_thinking = False
        end_thinking = False
        has_stop_interupted = False
        voice_mode = self._is_voice_enabled()
        voice_sentence = "" if voice_mode != "disabled" else None
        voice_id = self._get_configured_voice_id() if self.voice_service else None

        if len(self.agent.history) == 0:
            return None, 0, 0

        # Create a reference to the streaming generator
        self.stream_generator = None

        def process_result(_tool_uses, _input_tokens, _output_tokens):
            nonlocal tool_uses, input_tokens, output_tokens
            tool_uses = _tool_uses
            input_tokens += _input_tokens
            output_tokens += _output_tokens

        try:
            # Store the generator in a variable so we can properly close it if needed
            self.stream_generator = self.agent.process_messages(callback=process_result)

            async for (
                assistant_response,
                chunk_text,
                thinking_chunk,
            ) in self.stream_generator:
                # Check if stop was requested
                if self.stop_streaming:
                    # Properly close the generator instead of breaking
                    self.stop_streaming = False  # Reset flag
                    has_stop_interupted = True
                    self._notify("streaming_stopped", assistant_response)
                    await self.stream_generator.aclose()

                # Accumulate thinking content if available
                if thinking_chunk:
                    think_text_chunk, signature = thinking_chunk

                    if not start_thinking:
                        # Notify about thinking process
                        self._notify("thinking_started", self.agent.name)
                        if not self.agent.is_streaming():
                            # Delays it a bit when using without stream
                            await asyncio.sleep(0.5)
                        start_thinking = True
                    if think_text_chunk:
                        thinking_content += think_text_chunk
                        self._notify("thinking_chunk", think_text_chunk)
                    if signature:
                        thinking_signature += signature
                if chunk_text:
                    # End thinking when chunk_text start
                    if not end_thinking and start_thinking:
                        self._notify("thinking_completed", thinking_content)
                        end_thinking = True
                    # Notify about response progress
                    if not self.agent.is_streaming():
                        # Delays it a bit when using without stream
                        await asyncio.sleep(0.3)
                    self._notify("response_chunk", (chunk_text, assistant_response))
                    if voice_sentence is not None:
                        if (
                            "<agent_evaluation>" in assistant_response
                            and "</agent_evaluation>" in assistant_response
                        ) or (
                            # First token of agent evaluation
                            "<agent" not in assistant_response
                            and "</agent_evaluation>" not in assistant_response
                        ):
                            voice_sentence += chunk_text

                    if (
                        voice_sentence
                        and "\n" in voice_sentence.lstrip("\n ").strip("<>_-")
                        and self.voice_service
                        and voice_id
                    ):
                        voice_sentence = (
                            voice_sentence.replace("<agent_evaluation>", "")
                            .replace("</agent_evaluation>", "")
                            .replace("<agent", "")
                            .replace("evaluation>", "")
                            .lstrip("\n ")
                            .strip("<>_-")
                        )
                        print(voice_sentence)
                        if len(voice_sentence.split("\n")) > 1:
                            self.voice_service.text_to_speech_stream(
                                voice_sentence.strip().partition("\n")[0],
                                voice_id=voice_id,
                            )
                            if voice_mode == "partial":
                                voice_sentence = None
                            else:
                                voice_sentence = (
                                    voice_sentence.strip()
                                    .partition("\n")[-1]
                                    .lstrip("\n")
                                )

            self.stream_generator = None

            # End thinking when break the response stream
            if not end_thinking and start_thinking:
                self._notify("thinking_completed", thinking_content)
                end_thinking = True

            if voice_sentence and voice_sentence.strip() and self.voice_service:
                self.voice_service.text_to_speech_stream(
                    voice_sentence.strip().partition("\n")[0], voice_id=voice_id
                )
            # Handle tool use if needed
            if not has_stop_interupted and tool_uses and len(tool_uses) > 0:
                # Add thinking content as a separate message if available
                thinking_data = (
                    (thinking_content, thinking_signature) if thinking_content else None
                )
                thinking_message = self.agent.format_message(
                    MessageType.Thinking, {"thinking": thinking_data}
                )
                if thinking_message:
                    self._messages_append(thinking_message)
                    self._notify("thinking_message_added", thinking_message)

                # Format assistant message with the response and tool uses
                tool_uses_without_transfer = [
                    t for t in tool_uses if t["name"] != "transfer"
                ]
                # only append message if there are tool uses other than transfer
                if len(tool_uses_without_transfer) > 0:
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                            "tool_uses": tool_uses_without_transfer,
                        },
                    )
                    self._messages_append(assistant_message)
                # ignore if message is empty
                elif assistant_response.strip():
                    assistant_message = self.agent.format_message(
                        MessageType.Assistant,
                        {
                            "message": assistant_response,
                        },
                    )
                    self._messages_append(assistant_message)
                self._notify("assistant_message_added", assistant_response)

                self._yolo_mode_check()

                # Process each tool use
                await self.tool_manager.execute_tools_batch(tool_uses)

                self._notify(
                    "update_token_usage",
                    {"input_tokens": input_tokens, "output_tokens": output_tokens},
                )

                if has_stop_interupted:
                    # return as soon as possible
                    self._notify("response_completed", assistant_response)
                    return assistant_response, input_tokens, output_tokens

                return await self.get_assistant_response()

            # Add assistant response to messages
            if assistant_response.strip():
                self._messages_append(
                    self.agent.format_message(
                        MessageType.Assistant, {"message": assistant_response}
                    )
                )
            # Final assistant message
            self._notify("response_completed", assistant_response)

            if self.current_conversation_id and self.last_assisstant_response_idx >= 0:
                try:
                    messages_for_this_turn = self.get_recent_agent_responses()
                    if messages_for_this_turn and self.persistent_service:
                        metadata = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                        self.persistent_service.store_conversation_metadata(
                            self.current_conversation_id, metadata
                        )

                        self.persistent_service.append_conversation_messages(
                            self.current_conversation_id,
                            messages_for_this_turn,
                        )
                        self._notify(
                            "conversation_saved", {"id": self.current_conversation_id}
                        )
                except Exception as e:
                    error_message = f"Failed to save conversation turn to {self.current_conversation_id}: {str(e)}"
                    logger.error(f"ERROR: {error_message}")
                    self._notify("error", {"message": error_message})

            self.last_assisstant_response_idx = len(self.streamline_messages)
            # --- End of Persistence Logic ---

            if self.current_user_input and self.current_user_input_idx >= 0:
                self.conversation_manager.store_conversation_turn(
                    self.current_user_input, self.current_user_input_idx
                )
                if self.memory_service:
                    user_input = ""
                    user_message = self.current_user_input  # Get the user message
                    if (
                        isinstance(user_message["content"], list)
                        and len(user_message["content"]) > 0
                    ):
                        for content_item in user_message["content"]:
                            if content_item.get("type") == "text":
                                user_input += content_item.get("text", "")
                    elif isinstance(user_message["content"], str):
                        user_input = user_message["content"]

                    try:
                        self.memory_service.store_conversation(
                            user_input, assistant_response, self.agent.name
                        )
                    except Exception as e:
                        self._notify(
                            "error", f"Failed to store conversation in memory: {str(e)}"
                        )
                # Store the conversation turn reference for /jump command
                self.current_user_input = None
                self.current_user_input_idx = -1

            if self.agent_manager.defered_transfer:
                self.agent.history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""<Transfer_Post_Action_Reminder>{self.agent_manager.defered_transfer}. If action related to other agent, use `transfer` tool to chaining the work</Transfer_Post_Action_Reminder>""",
                            }
                        ],
                    }
                )
                self.agent_manager.defered_transfer = ""
                return await self.get_assistant_response()

            return assistant_response, input_tokens, output_tokens

        except GeneratorExit:
            return assistant_response, input_tokens, output_tokens
        except Exception as e:
            from openai import BadRequestError

            if isinstance(e, BadRequestError):
                if e.code == "model_max_prompt_tokens_exceeded":
                    from AgentCrew.modules.agents import LocalAgent
                    from AgentCrew.modules.llm.model_registry import ModelRegistry

                    if isinstance(self.agent, LocalAgent):
                        max_token = ModelRegistry.get_model_limit(
                            self.agent.get_model()
                        )
                        self.agent.input_tokens_usage = max_token
                        return await self.get_assistant_response()
            if self.current_user_input:
                self.conversation_manager.store_conversation_turn(
                    self.current_user_input, self.current_user_input_idx
                )
                self.current_user_input = None
                self.current_user_input_idx = -1
            if self.current_conversation_id and self.last_assisstant_response_idx >= 0:
                messages_for_this_turn = self.get_recent_agent_responses()
                if messages_for_this_turn and self.persistent_service:
                    metadata = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                    self.persistent_service.store_conversation_metadata(
                        self.current_conversation_id, metadata
                    )

                    self.persistent_service.append_conversation_messages(
                        self.current_conversation_id,
                        messages_for_this_turn,
                    )
                    self._notify(
                        "conversation_saved", {"id": self.current_conversation_id}
                    )
            self.last_assisstant_response_idx = len(self.streamline_messages)

            error_message = str(e)
            traceback_str = traceback.format_exc()
            logger.error(f"{error_message} \n {traceback_str}")
            self._notify(
                "error",
                {
                    "message": error_message,
                    "messages": self.agent.history,
                },
            )
            return None, 0, 0

    def get_recent_agent_responses(self) -> List:
        return self.streamline_messages[self.last_assisstant_response_idx :]

    # Delegate conversation management methods
    def list_conversations(self):
        """Lists available conversations from the persistence service."""
        return self.conversation_manager.list_conversations()

    def list_conversations_with_forks(self):
        """Lists available conversations with fork relationship information."""
        return self.conversation_manager.list_conversations_with_forks()

    def load_conversation(self, conversation_id: str):
        """Loads a specific conversation history and sets it as active."""
        # Reset approved tools for the loaded conversation
        self.tool_manager.reset_approved_tools()
        return self.conversation_manager.load_conversation(conversation_id)

    def delete_conversation_by_id(self, conversation_id: str) -> bool:
        """Deletes a conversation by its ID."""
        return self.conversation_manager.delete_conversation_by_id(conversation_id)

    def _is_voice_enabled(self) -> bool:
        """Check if voice is enabled in current agent settings."""
        try:
            # Check if voice service is available first
            if self.voice_service is None:
                return False

            if hasattr(self.agent, "voice_enabled"):
                return getattr(self.agent, "voice_enabled")

            return False
        except Exception as e:
            logger.warning(f"Failed to read voice_enabled setting: {e}")
            return False

    def _get_configured_voice_id(self) -> Optional[str]:
        """Get the voice ID from current agent settings or return default."""
        try:
            if hasattr(self.agent, "voice_id"):
                return getattr(self.agent, "voice_id", None)

            return None

        except Exception as e:
            logger.warning(f"Failed to read voice_id from agent config: {e}")
            return None
