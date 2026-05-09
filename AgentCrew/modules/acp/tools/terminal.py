from __future__ import annotations

import os
import uuid
from typing import Any, Callable

from loguru import logger

from .context import _current_acp_session


def get_acp_run_command_tool_definition() -> dict[str, Any]:
    import sys

    is_windows = sys.platform == "win32"
    shell = "PowerShell" if is_windows else "Bash"
    ex = (
        "dir, python script.py, git status"
        if is_windows
        else "ls -la, python script.py, git status, ps aux"
    )

    return {
        "type": "function",
        "function": {
            "name": "acp_run_command",
            "description": (
                f"Execute commands via the ACP client's terminal ({shell}). "
                f"Commands run in the client's shell environment. Returns output and exit code. "
                f"Do not use `&` as it already a sub-process. Ex: {ex}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": f"Command to execute. Do not use `&`. Ex: {ex}",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds (default: 5, max: 60). Returns result when command completes or timeout.",
                        "minimum": 5,
                        "maximum": 60,
                        "default": 5,
                    },
                    "working_dir": {
                        "type": "string",
                        "description": f"Working directory. Current: {os.getcwd()}. Use './' for current dir.",
                    },
                    "env_vars": {
                        "type": "object",
                        "description": "Environment variables dict. Cannot override PATH, HOME, USER.",
                    },
                },
                "required": ["command", "working_dir"],
            },
        },
    }


def get_acp_run_command_tool_handler() -> Callable:
    async def handle_acp_run_command(**params) -> str:
        command = params.get("command", "")
        timeout = params.get("timeout", 5)
        working_dir = params.get("working_dir", os.getcwd())
        env_vars = params.get("env_vars")

        if not command:
            return "Error: command is required"

        ctx = _current_acp_session.get()
        if ctx is not None and ctx.conn is not None:
            try:
                if not _validate_command_safety(command, working_dir, env_vars):
                    return "Error: command blocked by security policy"

                cmd_parts = command.split(maxsplit=1)
                cmd_name = cmd_parts[0]
                cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                response = await ctx.conn.create_terminal(
                    command=cmd_name,
                    session_id=ctx.session_id,
                    args=cmd_args if cmd_args else None,
                    cwd=working_dir,
                    env=_build_env_list(env_vars) if env_vars else None,
                )
                terminal_id = response.terminal_id

                cmd_id = f"cmd_{uuid.uuid4().hex[:12]}"
                ctx.active_terminals[cmd_id] = terminal_id

                wait_response = await ctx.conn.wait_for_terminal_exit(
                    session_id=ctx.session_id,
                    terminal_id=terminal_id,
                )

                output_response = await ctx.conn.terminal_output(
                    session_id=ctx.session_id,
                    terminal_id=terminal_id,
                )

                await ctx.conn.release_terminal(
                    session_id=ctx.session_id,
                    terminal_id=terminal_id,
                )
                ctx.active_terminals.pop(cmd_id, None)

                exit_code = (
                    wait_response.exit_code
                    if wait_response.exit_code is not None
                    else 0
                )
                out = output_response.output
                return (
                    f"Command completed (exit code: {exit_code})\n"
                    f"Command ID: {cmd_id}\n"
                    f"stdout/stderr:\n{out}"
                )
            except Exception as exc:
                logger.warning(
                    f"ACP terminal failed for '{command}', falling back to local: {exc}"
                )

        return await _local_run_command(command, timeout, working_dir, env_vars)

    return handle_acp_run_command


def get_acp_check_command_status_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "acp_check_command_status",
            "description": (
                "Check status and output of a running ACP terminal command. "
                "Returns output, status (running/completed), elapsed time, exit code if completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command_id": {
                        "type": "string",
                        "description": "Command ID from acp_run_command (format: 'cmd_xxxxxxxxxxxx').",
                    },
                },
                "required": ["command_id"],
            },
        },
    }


def get_acp_check_command_status_tool_handler() -> Callable:
    async def handle_acp_check_command_status(**params) -> dict[str, Any] | str:
        command_id = params.get("command_id", "")

        if not command_id:
            return "Error: command_id is required"

        ctx = _current_acp_session.get()
        if ctx is not None and ctx.conn is not None:
            terminal_id = ctx.active_terminals.get(command_id)
            if terminal_id:
                try:
                    response = await ctx.conn.terminal_output(
                        session_id=ctx.session_id,
                        terminal_id=terminal_id,
                    )
                    status = "completed" if response.exit_status else "running"
                    out = response.output
                    return f"Status: {status}\nOutput:\n{out}"
                except Exception as exc:
                    logger.warning(f"ACP terminal_output failed: {exc}")

        return await _local_check_command_status(command_id)

    return handle_acp_check_command_status


def get_acp_terminate_command_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "acp_terminate_command",
            "description": (
                "Terminate a running ACP terminal command by command ID. "
                "Kills the terminal process and releases resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command_id": {
                        "type": "string",
                        "description": "Command ID from acp_run_command (format: 'cmd_xxxxxxxxxxxx').",
                    },
                },
                "required": ["command_id"],
            },
        },
    }


def get_acp_terminate_command_tool_handler() -> Callable:
    async def handle_acp_terminate_command(**params) -> str | dict[str, Any]:
        command_id = params.get("command_id", "")

        if not command_id:
            return "Error: command_id is required"

        ctx = _current_acp_session.get()
        if ctx is not None and ctx.conn is not None:
            terminal_id = ctx.active_terminals.pop(command_id, None)
            if terminal_id:
                try:
                    await ctx.conn.kill_terminal(
                        session_id=ctx.session_id,
                        terminal_id=terminal_id,
                    )
                    await ctx.conn.release_terminal(
                        session_id=ctx.session_id,
                        terminal_id=terminal_id,
                    )
                    return f"Command {command_id} terminated."
                except Exception as exc:
                    logger.warning(f"ACP terminal kill failed: {exc}")

        return await _local_terminate_command(command_id)

    return handle_acp_terminate_command


def _validate_command_safety(
    command: str, working_dir: str, env_vars: dict | None
) -> bool:
    import re

    from AgentCrew.modules.command_execution.constants import (
        BLOCKED_PATTERNS,
        PROHIBITED_WORKING_PATHS,
        PROTECTED_ENV_VARS,
    )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            logger.warning(f"ACP terminal: blocked pattern '{pattern}' in command")
            return False

    import sys

    platform_key = (
        "win32"
        if sys.platform == "win32"
        else ("darwin" if sys.platform == "darwin" else "linux")
    )
    for prohibited in PROHIBITED_WORKING_PATHS.get(platform_key, []):
        if os.path.abspath(working_dir).startswith(prohibited):
            logger.warning(
                f"ACP terminal: prohibited working directory '{working_dir}'"
            )
            return False

    if env_vars:
        for protected in PROTECTED_ENV_VARS:
            if protected.upper() in (k.upper() for k in env_vars):
                logger.warning(
                    f"ACP terminal: attempt to override protected env var '{protected}'"
                )
                return False

    return True


def _build_env_list(env_vars: dict[str, str]) -> list:
    from acp.schema import EnvVariable

    return [EnvVariable(name=k, value=v) for k, v in env_vars.items()]


async def _local_run_command(
    command: str, timeout: int, working_dir: str, env_vars: dict | None
) -> str:
    from AgentCrew.modules.command_execution.service import CommandExecutionService

    service = CommandExecutionService.get_instance()
    result = service.execute_command(
        command=command,
        timeout=timeout,
        working_dir=working_dir,
        env_vars=env_vars,
    )
    cmd_id = result.get("command_id", "unknown")
    out = result.get("stdout", "") or result.get("output", "")
    stderr = result.get("stderr", "")
    exit_code = result.get("exit_code", result.get("returncode", 0))
    parts = [f"Command completed (exit code: {exit_code})"]
    parts.append(f"Command ID: {cmd_id}")
    parts.append(f"stdout/stderr:\n{out}{stderr}")
    return "\n".join(parts)


async def _local_check_command_status(command_id: str) -> dict[str, Any]:
    from AgentCrew.modules.command_execution.service import CommandExecutionService

    service = CommandExecutionService.get_instance()
    return service.get_command_status(command_id)


async def _local_terminate_command(command_id: str) -> dict[str, Any]:
    from AgentCrew.modules.command_execution.service import CommandExecutionService

    service = CommandExecutionService.get_instance()
    return service.terminate_command(command_id)


def register(context: Any = None, agent: Any = None):
    from AgentCrew.modules.tools.registration import register_tool

    register_tool(
        get_acp_run_command_tool_definition,
        get_acp_run_command_tool_handler,
        context,
        agent,
    )
    register_tool(
        get_acp_check_command_status_tool_definition,
        get_acp_check_command_status_tool_handler,
        context,
        agent,
    )
    register_tool(
        get_acp_terminate_command_tool_definition,
        get_acp_terminate_command_tool_handler,
        context,
        agent,
    )
