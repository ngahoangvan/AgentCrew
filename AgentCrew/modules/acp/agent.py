import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from acp import Agent
from acp.schema import ToolKind
from loguru import logger

from AgentCrew.modules.acp.mcp import normalize_acp_mcp_servers
from AgentCrew.modules.acp.session_store import AcpSessionStore
from AgentCrew.modules.agents import AgentManager, LocalAgent
from AgentCrew.modules.agents.base import MessageType
from AgentCrew.modules.mcpclient import MCPSessionManager
from AgentCrew.modules.tools.parallel_executor import (
    execute_tools_in_parallel,
    is_sequential_tool,
)

ACP_MCP_STARTUP_TIMEOUT_SECONDS = 8.0
ACP_MCP_STARTUP_POLL_SECONDS = 0.1


@dataclass
class AcpSessionState:
    cwd: str
    agent_name: str
    history: list[dict[str, Any]] = field(default_factory=list)
    current_task: asyncio.Task | None = None
    cancelled: bool = False
    acp_mcp_server_configs: list[Any] = field(default_factory=list)
    acp_mcp_server_ids: list[str] = field(default_factory=list)
    title: str | None = None


class AgentCrewAcpAgent(Agent):
    def __init__(
        self,
        agent_manager: AgentManager,
        default_agent_name: str | None = None,
        session_store: AcpSessionStore | None = None,
    ):
        self.agent_manager: AgentManager = agent_manager
        self.default_agent_name: str | None = default_agent_name
        self.session_store: AcpSessionStore = session_store or AcpSessionStore()
        self._conn = None
        self._sessions: dict[str, AcpSessionState] = {}

    def on_connect(self, conn):
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities=None,
        client_info=None,
        **kwargs,
    ):
        from acp import PROTOCOL_VERSION
        from acp.schema import (
            AgentAuthCapabilities,
            AgentCapabilities,
            Implementation,
            McpCapabilities,
            PromptCapabilities,
            SessionCapabilities,
            SessionCloseCapabilities,
            SessionListCapabilities,
            SessionResumeCapabilities,
        )

        import AgentCrew

        return self._model(
            "InitializeResponse",
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                auth=AgentAuthCapabilities(),
                load_session=True,
                mcp_capabilities=McpCapabilities(http=False, sse=False),
                prompt_capabilities=PromptCapabilities(embedded_context=True),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            agent_info=Implementation(
                name="agentcrew",
                title="AgentCrew",
                version=getattr(AgentCrew, "__version__", "0.0.0"),
            ),
        )

    async def authenticate(self, method_id: str, **kwargs):
        return self._model("AuthenticateResponse")

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs,
    ):
        session_id = f"agentcrew-{uuid.uuid4().hex}"
        agent_name = self._resolve_agent_name(self.default_agent_name)
        state = AcpSessionState(
            cwd=os.path.abspath(os.path.expanduser(cwd)),
            agent_name=agent_name,
        )
        self._sessions[session_id] = state
        await self._setup_session_mcp_servers(session_id, state, mcp_servers)
        await self._persist_session(session_id, state)
        return self._model(
            "NewSessionResponse",
            session_id=session_id,
            modes=self._build_modes(agent_name),
            models=self._build_model_state(),
            config_options=self._build_config_options(),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs,
    ):
        stored = await self.session_store.load_session(session_id)
        agent_name = self._resolve_agent_name(
            stored.agent_name if stored else self.default_agent_name
        )
        loaded_history = [dict(message) for message in stored.history] if stored else []
        state = AcpSessionState(
            cwd=os.path.abspath(os.path.expanduser(cwd)),
            agent_name=agent_name,
            history=loaded_history,
            title=stored.title if stored else None,
        )
        self._sessions[session_id] = state
        await self._setup_session_mcp_servers(session_id, state, mcp_servers)
        await self._persist_session(session_id, state)
        await self._replay_history_to_client(session_id, loaded_history)
        return self._model(
            "LoadSessionResponse",
            modes=self._build_modes(agent_name),
            models=self._build_model_state(),
            config_options=self._build_config_options(),
        )

    async def close_session(self, session_id: str, **kwargs):
        state = self._sessions.pop(session_id, None)
        if state and state.current_task and not state.current_task.done():
            state.cancelled = True
            state.current_task.cancel()
        if state:
            await self._cleanup_session_mcp_servers(state, clear_configs=True)
        return self._model("CloseSessionResponse")

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs):
        if session_id not in self._sessions:
            self._sessions[session_id] = AcpSessionState(
                cwd=os.getcwd(),
                agent_name=self._resolve_agent_name(mode_id),
            )
        else:
            state = self._sessions[session_id]
            next_agent_name = self._resolve_agent_name(mode_id)
            if next_agent_name != state.agent_name and state.acp_mcp_server_configs:
                await self._cleanup_session_mcp_servers(state)
                state.agent_name = next_agent_name
                for config in state.acp_mcp_server_configs:
                    config.enabledForAgents = [next_agent_name]
                await self._start_session_mcp_configs(state)
            else:
                state.agent_name = next_agent_name
        await self._persist_session(session_id, self._sessions[session_id])
        await self._send_current_mode_update(session_id, self._sessions[session_id])
        return self._model("SetSessionModeResponse")

    async def list_sessions(
        self,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs,
    ):
        from acp.schema import SessionInfo

        normalized_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else None
        sessions_by_id = {}
        for stored in await self.session_store.list_sessions(cwd=normalized_cwd):
            sessions_by_id[stored.session_id] = SessionInfo(
                cwd=stored.cwd,
                session_id=stored.session_id,
                title=stored.title or stored.agent_name,
            )
        for session_id, state in self._sessions.items():
            if normalized_cwd and state.cwd != normalized_cwd:
                continue
            sessions_by_id[session_id] = SessionInfo(
                cwd=state.cwd,
                session_id=session_id,
                title=state.title or state.agent_name,
            )
        return self._model(
            "ListSessionsResponse", sessions=list(sessions_by_id.values())
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs,
    ):
        state = self._sessions.get(session_id)
        if state is None:
            stored = await self.session_store.load_session(session_id)
            agent_name = self._resolve_agent_name(
                stored.agent_name if stored else self.default_agent_name
            )
            state = AcpSessionState(
                cwd=os.path.abspath(os.path.expanduser(cwd)),
                agent_name=agent_name,
                history=[dict(message) for message in stored.history] if stored else [],
                title=stored.title if stored else None,
            )
            self._sessions[session_id] = state
        else:
            state.cwd = os.path.abspath(os.path.expanduser(cwd))
        await self._setup_session_mcp_servers(session_id, state, mcp_servers)
        await self._persist_session(session_id, state)
        return self._model(
            "ResumeSessionResponse",
            modes=self._build_modes(state.agent_name),
            models=self._build_model_state(),
            config_options=self._build_config_options(),
        )

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs,
    ):
        source = self._sessions.get(session_id)
        stored = None if source else await self.session_store.load_session(session_id)
        new_session_id = f"agentcrew-{uuid.uuid4().hex}"
        if source is not None:
            agent_name = source.agent_name
            history = [dict(message) for message in source.history]
            title = source.title
        elif stored is not None:
            agent_name = self._resolve_agent_name(stored.agent_name)
            history = [dict(message) for message in stored.history]
            title = stored.title
        else:
            agent_name = self._resolve_agent_name(self.default_agent_name)
            history = []
            title = None
        forked_state = AcpSessionState(
            cwd=os.path.abspath(os.path.expanduser(cwd)),
            agent_name=agent_name,
            history=history,
            title=title,
        )
        self._sessions[new_session_id] = forked_state
        await self._persist_session(new_session_id, forked_state)
        return self._model(
            "ForkSessionResponse",
            session_id=new_session_id,
            modes=self._build_modes(agent_name),
        )

    async def set_session_model(self, model_id: str, session_id: str, **kwargs):
        return self._model("SetSessionModelResponse")

    async def _persist_session(self, session_id: str, state: AcpSessionState):
        await self.session_store.save_session(
            session_id=session_id,
            cwd=state.cwd,
            agent_name=state.agent_name,
            history=state.history,
            title=state.title,
        )

    async def _replay_history_to_client(
        self, session_id: str, history: list[dict[str, Any]]
    ):
        if self._conn is None:
            return

        from acp import text_block, update_agent_message_text
        from acp.schema import UserMessageChunk

        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            if role == "user":
                await self._conn.session_update(
                    session_id,
                    UserMessageChunk(
                        content=text_block(content),
                        session_update="user_message_chunk",
                    ),
                )
            elif role in ("assistant", "thinking", "consolidated"):
                if content.strip():
                    await self._conn.session_update(
                        session_id,
                        update_agent_message_text(content),
                    )

    async def _setup_session_mcp_servers(
        self,
        session_id: str,
        state: AcpSessionState,
        mcp_servers: list[Any] | None,
    ):
        configs = normalize_acp_mcp_servers(session_id, state.agent_name, mcp_servers)
        if not configs:
            return
        await self._cleanup_session_mcp_servers(state, clear_configs=True)
        state.acp_mcp_server_configs = configs
        await self._start_session_mcp_configs(state)

    async def _start_session_mcp_configs(self, state: AcpSessionState):
        mcp_manager = MCPSessionManager.get_instance()
        if not mcp_manager.initialized:
            mcp_manager.initialize()

        service = mcp_manager.mcp_service
        active_configs: list[Any] = []
        active_ids: list[str] = []
        for config in list(state.acp_mcp_server_configs):
            combined_id = service._get_server_id_format(config.name, state.agent_name)
            try:
                service._run_async(
                    service.start_server_connection_management(config, state.agent_name)
                )
                ready = await self._wait_for_mcp_server_ready(
                    service, combined_id, state.agent_name
                )
            except Exception:
                logger.exception(
                    f"ACP MCP server '{config.name}' failed during startup"
                )
                ready = False

            if ready:
                active_configs.append(config)
                active_ids.append(combined_id)
                continue

            logger.warning(
                f"ACP MCP server '{config.name}' did not become ready; continuing without it"
            )
            await self._cleanup_single_mcp_server(
                service, config.name, state.agent_name, combined_id
            )

        state.acp_mcp_server_configs = active_configs
        state.acp_mcp_server_ids = active_ids

    async def _wait_for_mcp_server_ready(
        self,
        service: Any,
        combined_id: str,
        agent_name: str,
        timeout_seconds: float = ACP_MCP_STARTUP_TIMEOUT_SECONDS,
    ) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            agent = self.agent_manager.get_local_agent(agent_name)
            if agent:
                if combined_id not in agent.mcps_loading and (
                    service.connected_servers.get(combined_id)
                    or combined_id in service.tools_cache
                ):
                    return True
            elif (
                service.connected_servers.get(combined_id)
                or combined_id in service.tools_cache
            ):
                return True

            task = service._server_connection_tasks.get(combined_id)
            if task and task.done():
                if task.cancelled():
                    return False
                try:
                    exc = task.exception()
                except Exception:
                    return False
                if exc:
                    logger.warning(
                        f"ACP MCP server task failed for '{combined_id}': {exc}"
                    )
                    return False
                if agent:
                    return bool(
                        combined_id not in agent.mcps_loading
                        and (
                            service.connected_servers.get(combined_id)
                            or combined_id in service.tools_cache
                        )
                    )
                return bool(
                    service.connected_servers.get(combined_id)
                    or combined_id in service.tools_cache
                )

            await asyncio.sleep(ACP_MCP_STARTUP_POLL_SECONDS)

        return False

    async def _cleanup_single_mcp_server(
        self, service: Any, server_name: str, agent_name: str, combined_id: str
    ):
        try:
            service._run_async(service.deregister_server_tools(server_name, agent_name))
        except Exception:
            logger.exception("Error deregistering failed ACP MCP server tools")

        try:
            service._run_async(service.shutdown_single_server_connection(combined_id))
        except Exception:
            logger.exception("Error stopping failed ACP MCP server")

    async def _cleanup_session_mcp_servers(
        self, state: AcpSessionState, clear_configs: bool = False
    ):
        if not state.acp_mcp_server_ids and not state.acp_mcp_server_configs:
            return

        mcp_manager = MCPSessionManager.get_instance()
        service = mcp_manager.mcp_service
        for config in list(state.acp_mcp_server_configs):
            try:
                service._run_async(
                    service.deregister_server_tools(config.name, state.agent_name)
                )
            except Exception:
                logger.exception("Error deregistering ACP MCP server tools")

        for combined_id in list(state.acp_mcp_server_ids):
            try:
                service._run_async(
                    service.shutdown_single_server_connection(combined_id)
                )
            except Exception:
                logger.exception("Error shutting down ACP MCP server")

        state.acp_mcp_server_ids.clear()
        if clear_configs:
            state.acp_mcp_server_configs.clear()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs,
    ):
        return self._model("SetSessionConfigOptionResponse", config_options=[])

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs,
    ):
        state = self._sessions.get(session_id)
        if state is None:
            state = AcpSessionState(
                cwd=os.getcwd(),
                agent_name=self._resolve_agent_name(self.default_agent_name),
            )
            self._sessions[session_id] = state

        user_text = self._prompt_to_text(prompt)
        if user_text.strip():
            state.history.append({"role": "user", "content": user_text})
            if not state.title:
                state.title = user_text[:80].split("\n")[0].strip()
                await self._send_session_info_update(session_id, state)

        state.cancelled = False
        state.current_task = asyncio.current_task()
        try:
            await self._run_turn(session_id, state)
            return self._model(
                "PromptResponse",
                stop_reason="cancelled" if state.cancelled else "end_turn",
                user_message_id=message_id,
            )
        except asyncio.CancelledError:
            state.cancelled = True
            return self._model(
                "PromptResponse",
                stop_reason="cancelled",
                user_message_id=message_id,
            )
        except Exception as e:
            logger.exception("ACP prompt failed")
            await self._send_agent_message(session_id, f"AgentCrew ACP error: {e}")
            return self._model(
                "PromptResponse",
                stop_reason="refusal",
                user_message_id=message_id,
            )
        finally:
            state.current_task = None
            await self._persist_session(session_id, state)

    async def cancel(self, session_id: str, **kwargs):
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.cancelled = True
        if state.current_task and not state.current_task.done():
            state.current_task.cancel()

    async def ext_method(self, method: str, params: dict[str, Any]):
        return {"error": f"Unsupported extension method: {method}"}

    async def ext_notification(self, method: str, params: dict[str, Any]):
        logger.debug(f"Ignoring unsupported ACP extension notification: {method}")

    async def _run_turn(self, session_id: str, state: AcpSessionState):
        agent = self._get_agent(state.agent_name)
        current_response = ""
        thinking_content = ""
        thinking_signature = ""
        tool_uses: list[dict[str, Any]] = []
        token_usage = None

        def process_result(_tool_uses, _token_usage):
            nonlocal tool_uses, token_usage
            tool_uses = _tool_uses
            token_usage = _token_usage

        async for (
            response_message,
            chunk_text,
            thinking_chunk,
        ) in agent.process_messages(
            state.history,
            callback=process_result,
        ):
            if state.cancelled:
                return
            if response_message:
                current_response = response_message
            if chunk_text:
                await self._send_agent_message(session_id, chunk_text)
            if thinking_chunk:
                think_text_chunk, signature = thinking_chunk
                if think_text_chunk:
                    thinking_content += think_text_chunk
                    await self._send_thought_chunk(session_id, think_text_chunk)
                if signature:
                    thinking_signature += signature

        thinking_data = (
            (thinking_content, thinking_signature) if thinking_content else None
        )
        thinking_message = agent.format_message(
            MessageType.Thinking,
            {"thinking": thinking_data},
        )
        if thinking_message:
            state.history.append(thinking_message)

        assistant_message = agent.format_message(
            MessageType.Assistant,
            {"message": current_response, "tool_uses": tool_uses},
        )
        if assistant_message:
            state.history.append(assistant_message)

        if tool_uses:
            await self._execute_tools(session_id, state, agent, tool_uses)
            if not state.cancelled:
                await self._run_turn(session_id, state)

    async def _execute_tools(
        self,
        session_id: str,
        state: AcpSessionState,
        agent: LocalAgent,
        tool_uses: list[dict[str, Any]],
    ):
        parallel_buffer: list[dict[str, Any]] = []

        async def flush_parallel():
            nonlocal parallel_buffer
            if not parallel_buffer:
                return
            for tool_use in parallel_buffer:
                await self._send_tool_started(session_id, tool_use)
            results = await execute_tools_in_parallel(
                parallel_buffer, agent.execute_tool_call
            )
            for result in results:
                await self._append_tool_result(
                    session_id,
                    state,
                    agent,
                    result.tool_use,
                    result.result,
                    result.is_error,
                )
            parallel_buffer = []

        for tool_use in tool_uses:
            if state.cancelled:
                return
            if is_sequential_tool(tool_use["name"]):
                await flush_parallel()
                await self._send_tool_started(session_id, tool_use)
                try:
                    tool_result = await agent.execute_tool_call(
                        tool_use["name"],
                        tool_use.get("input", {}),
                    )
                    await self._append_tool_result(
                        session_id, state, agent, tool_use, tool_result
                    )
                except Exception as e:
                    await self._append_tool_result(
                        session_id, state, agent, tool_use, str(e), True
                    )
            else:
                parallel_buffer.append(tool_use)

        await flush_parallel()

    async def _append_tool_result(
        self,
        session_id: str,
        state: AcpSessionState,
        agent: LocalAgent,
        tool_use: dict[str, Any],
        tool_result: Any,
        is_error: bool = False,
    ):
        result_message = agent.format_message(
            MessageType.ToolResult,
            {"tool_use": tool_use, "tool_result": tool_result, "is_error": is_error},
        )
        if result_message:
            state.history.append(result_message)
        await self._send_tool_completed(session_id, tool_use, tool_result, is_error)

    async def _send_agent_message(self, session_id: str, text: str):
        from acp import update_agent_message_text

        if self._conn is not None:
            await self._conn.session_update(session_id, update_agent_message_text(text))

    async def _send_thought_chunk(self, session_id: str, text: str):
        from acp import text_block
        from acp.schema import AgentThoughtChunk

        if self._conn is not None and text.strip():
            await self._conn.session_update(
                session_id,
                AgentThoughtChunk(
                    session_update="agent_thought_chunk",
                    content=text_block(text),
                ),
            )

    async def _send_current_mode_update(self, session_id: str, state: AcpSessionState):
        from acp.schema import CurrentModeUpdate

        if self._conn is not None:
            await self._conn.session_update(
                session_id,
                CurrentModeUpdate(
                    current_mode_id=state.agent_name,
                    session_update="current_mode_update",
                ),
            )

    async def _send_session_info_update(self, session_id: str, state: AcpSessionState):
        from acp.schema import SessionInfoUpdate

        if self._conn is not None:
            await self._conn.session_update(
                session_id,
                SessionInfoUpdate(
                    title=state.title,
                    session_update="session_info_update",
                ),
            )

    async def _send_tool_started(self, session_id: str, tool_use: dict[str, Any]):
        from acp import start_tool_call

        if self._conn is not None:
            await self._conn.session_update(
                session_id,
                start_tool_call(
                    tool_use["id"],
                    self._tool_title(tool_use),
                    kind=self._tool_kind(tool_use.get("name", "")),
                    status="in_progress",
                    raw_input=tool_use.get("input", {}),
                ),
            )

    async def _send_tool_completed(
        self,
        session_id: str,
        tool_use: dict[str, Any],
        tool_result: Any,
        is_error: bool,
    ):
        from acp import text_block, tool_content
        from acp.schema import ToolCallProgress

        if self._conn is not None:
            await self._conn.session_update(
                session_id,
                ToolCallProgress(
                    tool_call_id=tool_use["id"],
                    session_update="tool_call_update",
                    status="failed" if is_error else "completed",
                    content=[tool_content(text_block(str(tool_result)))],
                    raw_output=tool_result,
                ),
            )

    def _get_agent(self, agent_name: str) -> LocalAgent:
        agent = self.agent_manager.get_local_agent(agent_name)
        if agent is None:
            raise ValueError(f"Local agent '{agent_name}' not found")
        return agent

    def _resolve_agent_name(self, agent_name: str | None) -> str:
        if agent_name and agent_name in self.agent_manager.agents:
            return agent_name
        current_agent = self.agent_manager.get_current_agent()
        if current_agent is not None:
            return current_agent.name
        for name, agent in self.agent_manager.agents.items():
            if isinstance(agent, LocalAgent):
                return name
        raise ValueError("No local agents are available for ACP")

    def _build_model_state(self) -> Any | None:
        try:
            from AgentCrew.modules.llm.model_registry import ModelRegistry
            from acp.schema import ModelInfo, SessionModelState

            registry = ModelRegistry.get_instance()
            current_model = registry.get_current_model()
            if not current_model:
                return None

            available: list[Any] = []
            seen: set[str] = set()
            for model in registry.get_models_by_provider(current_model.provider):
                if model.id in seen:
                    continue
                seen.add(model.id)
                available.append(
                    ModelInfo(
                        model_id=model.id,
                        name=model.name,
                        description=model.description,
                    )
                )
            if not available:
                return None
            return SessionModelState(
                available_models=available,
                current_model_id=current_model.id,
            )
        except Exception:
            logger.exception("Failed to build ACP model state")
            return None

    def _build_config_options(self) -> list[Any]:
        return []

    def _build_modes(self, current_agent_name: str):
        from acp.schema import SessionMode, SessionModeState

        modes = [
            SessionMode(
                id=name,
                name=name,
                description=getattr(agent, "description", None),
            )
            for name, agent in self.agent_manager.agents.items()
            if isinstance(agent, LocalAgent)
        ]
        if not modes:
            return None
        return SessionModeState(
            available_modes=modes, current_mode_id=current_agent_name
        )

    def _prompt_to_text(self, prompt: list[Any]) -> str:
        chunks: list[str] = []
        for block in prompt:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                chunks.append(getattr(block, "text", ""))
            elif block_type == "resource_link":
                chunks.append(self._resource_link_to_text(block))
            elif block_type == "resource":
                chunks.append(self._embedded_resource_to_text(block))
            else:
                chunks.append(f"[Unsupported ACP content block: {block_type}]")
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _resource_link_to_text(self, block: Any) -> str:
        uri = getattr(block, "uri", "")
        name = getattr(block, "name", "resource")
        return f"[ACP resource link: {name}]({uri})"

    def _embedded_resource_to_text(self, block: Any) -> str:
        resource = getattr(block, "resource", None)
        if resource is None:
            return ""
        text = getattr(resource, "text", None)
        uri = getattr(resource, "uri", "embedded-resource")
        if text is not None:
            return f"[ACP embedded resource: {uri}]\n{text}"
        return f"[ACP embedded binary resource: {uri}]"

    def _tool_title(self, tool_use: dict[str, Any]) -> str:
        return f"{tool_use.get('name', 'tool')}"

    def _tool_kind(self, tool_name: str) -> ToolKind:
        if "read" in tool_name or "get_file" in tool_name:
            return "read"
        if "write" in tool_name or "edit" in tool_name:
            return "edit"
        if "search" in tool_name or "grep" in tool_name or "find" in tool_name:
            return "search"
        if "command" in tool_name or "run" in tool_name:
            return "execute"
        if "web" in tool_name or "fetch" in tool_name:
            return "fetch"
        return "other"

    def _model(self, name: str, **kwargs):
        from acp import schema

        cls = getattr(schema, name)
        normalized = {}
        for key, value in kwargs.items():
            if "_" in key:
                parts = key.split("_")
                key = parts[0] + "".join(part.capitalize() for part in parts[1:])
            normalized[key] = value
        return cls(**normalized)


async def run_acp_agent(
    agent_manager: AgentManager, default_agent_name: str | None = None
):
    from acp import run_agent

    await run_agent(AgentCrewAcpAgent(agent_manager, default_agent_name))
