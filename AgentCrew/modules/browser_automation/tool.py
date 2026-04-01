from __future__ import annotations

from typing import TYPE_CHECKING
import difflib
import time

if TYPE_CHECKING:
    from .service import BrowserAutomationService
    from typing import Dict, Any, Callable, Union, List


def get_browser_navigate_tool_definition(provider="claude") -> Dict[str, Any]:
    """Get tool definition for browser navigation."""
    tool_description = "Navigate to a URL in the browser when you need a live rendered page for interaction, workflow testing, authenticated or session-specific access, JavaScript-rendered content, or inspection of page behavior. Check result before proceeding with other actions."
    tool_arguments = {
        "url": {
            "type": "string",
            "description": "Valid HTTP/HTTPS URL to navigate to (e.g., 'https://example.com').",
        },
        "profile": {
            "type": "string",
            "description": "Chrome user profile directory name (default: 'Default'). Allows agent to choose which Chrome user profile to use.",
            "default": "Default",
        },
    }
    tool_required = ["url"]

    if provider == "claude":
        return {
            "name": "open_browser_url",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # OpenAI-compatible provider format
        return {
            "type": "function",
            "function": {
                "name": "open_browser_url",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_mouse_action_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "Perform a mouse action on a live browser element using its UUID. "
        "Use this for interaction and workflow testing on the rendered page: `action=click` clicks the target element, and `action=scroll_to` scrolls that element into view. "
        "Always get the `element_uuid` from `get_browser_content` or `get_browser_elements_by_text` first. "
        'Example: `{"action": "click", "element_uuid": "..."}` or `{"action": "scroll_to", "element_uuid": "..."}`.'
    )
    tool_arguments = {
        "action": {
            "type": "string",
            "enum": ["click", "scroll_to"],
            "description": "Mouse action to perform on the element.",
        },
        "element_uuid": {
            "type": "string",
            "description": "UUID of the target element from `get_browser_content` or `get_browser_elements_by_text`.",
        },
    }
    tool_required = ["action", "element_uuid"]

    if provider == "claude":
        return {
            "name": "perform_browser_mouse_action",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "perform_browser_mouse_action",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_get_content_tool_definition(provider="claude") -> Dict[str, Any]:
    """Get tool definition for browser content extraction."""
    tool_description = (
        "Extract live rendered page content as markdown with tables of clickable, input, and scrollable elements. Use this after opening a page when you need interactive element discovery, workflow testing, authenticated or session-specific content, or inspection of rendered output. UUIDs reset on each call."
        "get_browser_content tool's result is UNIQUE in whole conversation. Remember to summarize important information before calling again."
    )
    tool_arguments = {}
    tool_required = []

    if provider == "claude":
        return {
            "name": "get_browser_content",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:  # OpenAI-compatible provider format
        return {
            "type": "function",
            "function": {
                "name": "get_browser_content",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_get_content_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    """Get the handler function for the browser content extraction tool."""

    async def handle_browser_get_content(**params) -> Union[List[Dict[str, Any]], str]:
        result = browser_service.get_page_content()
        browser_service._last_page_content = result.get("content", "")
        context_image = browser_service.capture_screenshot(
            format="jpeg",
            quality=70,
        )

        if result["success"]:
            tool_result = [
                {
                    "type": "text",
                    "text": f"[UNIQUE]{result.get('content', 'Cannot get page content. Please try again.')}[/UNIQUE]",
                },
            ]
            if context_image.get("success", False):
                tool_result.append(context_image.get("screenshot", {}))
            return tool_result
        else:
            raise RuntimeError(f"Content extraction failed: {result['error']}")

    return handle_browser_get_content


def get_browser_navigate_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    """Get the handler function for the browser navigate tool."""

    async def handle_browser_navigate(**params) -> str:
        url = params.get("url")
        profile = params.get("profile", "Default")

        if not url:
            return "Error: No URL provided for navigation."

        result = browser_service.navigate(url, profile=profile)

        if result.get("success", True):
            profile_info = (
                f"\nProfile: {result.get('profile', profile)}"
                if result.get("profile")
                else ""
            )
            return f"{result.get('message', 'Success')}. Call `get_browser_content` tool to read the url content.\nCurrent URL: {result.get('current_url', 'Unknown')}{profile_info}"
        else:
            raise RuntimeError(f"Navigation failed: {result['error']}")

    return handle_browser_navigate


def get_browser_mouse_action_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    async def handle_browser_mouse_action(**params) -> str:
        action = params.get("action")
        element_uuid = params.get("element_uuid")

        if not action:
            raise RuntimeError("No action provided for browser mouse action.")
        if not element_uuid:
            raise RuntimeError("No element_uuid provided for browser mouse action.")

        if action == "click":
            result = browser_service.click_element(element_uuid)
            if result.get("success", True):
                diff_summary = _get_content_delta_changes(browser_service)
                return (
                    f"{result.get('message', 'Success')}. Call `get_browser_content` tool to get the updated content.\n"
                    f"Action: {action}\nUUID: {element_uuid}\n"
                    f"ClickedElement: {result.get('elementInfo', {}).get('text', 'Unknown')}.\n"
                    f"Content delta changes:\n{diff_summary}"
                )
            raise RuntimeError(
                f"Mouse action failed: {result['error']}\nAction: {action}\nUUID: {element_uuid}.\nCall `get_browser_content` tool to get the updated UUID"
            )

        if action == "scroll_to":
            result = browser_service.scroll_to_element(element_uuid)
            if result.get("success", True):
                return (
                    f"{result.get('message', 'Success')}.\n"
                    f"Action: {action}\nUUID: {element_uuid}\n"
                    "Call `get_browser_content` tool to get the updated content."
                )
            raise RuntimeError(
                f"Mouse action failed: {result['error']}\nAction: {action}\nUUID: {element_uuid}"
            )

        raise RuntimeError("Invalid mouse action. Supported actions: click, scroll_to.")

    return handle_browser_mouse_action


def get_browser_keyboard_action_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "Perform a keyboard action in the browser for live-page interaction and workflow testing. "
        "Use `action=input_text` to type text into a specific input element, and use `action=send_key` to send a keyboard key or shortcut to the browser. "
        "Use the `value` field for both actions: for `input_text`, `value` is the text to type; for `send_key`, `value` is the key name such as `enter`, `escape`, `a`, or `f5`. "
        "`element_uuid` is required only for `input_text`. `modifiers` are only used with `send_key`. "
        'Examples: `{"action": "input_text", "element_uuid": "...", "value": "hello"}` and `{"action": "send_key", "value": "a", "modifiers": ["ctrl"]}`.'
    )
    tool_arguments = {
        "action": {
            "type": "string",
            "enum": ["input_text", "send_key"],
            "description": "Keyboard action to perform.",
        },
        "element_uuid": {
            "type": "string",
            "description": "UUID of the target input element. Required for `input_text`.",
        },
        "value": {
            "type": "string",
            "description": "For `input_text`, the text to type. For `send_key`, the key name to send such as `enter`, `escape`, `a`, or `f5`.",
        },
        "modifiers": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["ctrl", "alt", "shift", "meta"],
            },
            "description": "Optional modifier keys used only with `send_key`.",
            "default": [],
        },
    }
    tool_required = ["action"]

    if provider == "claude":
        return {
            "name": "perform_browser_keyboard_action",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "perform_browser_keyboard_action",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_keyboard_action_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    async def handle_browser_keyboard_action(**params) -> str:
        action = params.get("action")

        if not action:
            raise RuntimeError("No action provided for browser keyboard action.")

        if action == "input_text":
            element_uuid = params.get("element_uuid")
            value = params.get("value")

            if not element_uuid:
                raise RuntimeError(
                    "No element_uuid provided for input_text keyboard action."
                )
            if value is None:
                raise RuntimeError("No value provided for input_text keyboard action.")

            result = browser_service.input_data(element_uuid, str(value))
            if result.get("success", True):
                diff_summary = _get_content_delta_changes(browser_service)
                return (
                    f"{result.get('message', 'Success')}\n"
                    f"Action: {action}\nUUID: {element_uuid}\nValue: {value}\n"
                    f"Content delta changes:\n{diff_summary}"
                )
            raise RuntimeError(
                f"Keyboard action failed: {result['error']}\n"
                f"Action: {action}\nUUID: {element_uuid}\nValue: {value}.\n"
                "Call `get_browser_content` tool to get updated UUID."
            )

        if action == "send_key":
            value = params.get("value")
            modifiers = params.get("modifiers", [])

            if not value:
                raise RuntimeError("No value provided for send_key keyboard action.")

            result = browser_service.dispatch_key_event(value, modifiers)
            if result.get("success", False):
                key_info = f"Key: {result.get('key')} (Code: {result.get('key_code')})"
                modifiers_info = (
                    f"Modifiers: {result.get('modifiers')}"
                    if result.get("modifiers")
                    else ""
                )
                diff_summary = _get_content_delta_changes(browser_service)
                success_msg = (
                    f"{result.get('message', 'Success')}. Action: {action}. {key_info}\n"
                    f"Content delta changes:\n{diff_summary}"
                )
                if modifiers_info:
                    success_msg += f". {modifiers_info}"
                return success_msg
            raise RuntimeError(
                f"Keyboard action failed: {result.get('error', 'Unknown error')}\n"
                f"Action: {action}\nValue: {value}"
            )

        raise RuntimeError(
            "Invalid keyboard action. Supported actions: input_text, send_key."
        )

    return handle_browser_keyboard_action


def get_browser_get_elements_by_text_tool_definition(
    provider="claude",
) -> Dict[str, Any]:
    """Get tool definition for browser elements by text search."""
    tool_description = "Find div elements containing specific text in the currently rendered browser page. Use this to locate live page elements for interaction, navigation, or testing after loading the page. Returns UUID table for use with other browser tools."
    tool_arguments = {
        "text": {
            "type": "string",
            "description": "Text to search for in div elements (case-insensitive).",
        }
    }
    tool_required = ["text"]

    if provider == "claude":
        return {
            "name": "get_browser_elements_by_text",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "get_browser_elements_by_text",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_get_elements_by_text_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    """Get handler function for browser get elements by text tool."""

    async def handle_browser_get_elements_by_text(**params) -> str:
        text = params.get("text")
        if not text:
            return "Error: No text provided for element search."

        result = browser_service.get_elements_by_text(text)

        if result.get("success", False):
            elements_found = result.get("elements_found", 0)
            if elements_found == 0:
                return f"No elements found containing text: '{text}'"

            content = result.get("content", "")
            return (
                f"Found {elements_found} elements containing text: '{text}'\n" + content
            )
        else:
            raise RuntimeError(
                f"Search failed: {result.get('error', 'Unknown error')}\nSearch text: '{text}'"
            )

    return handle_browser_get_elements_by_text


def get_browser_refresh_tool_definition(provider="claude") -> Dict[str, Any]:
    """Get tool definition for browser page refresh."""
    tool_description = (
        "Refresh/reload the current browser page to re-check live rendered state during interaction or testing. Equivalent to pressing F5 or Ctrl+R."
    )
    tool_arguments = {}
    tool_required = []

    if provider == "claude":
        return {
            "name": "refresh_browser_content",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "refresh_browser_content",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_refresh_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    """Get the handler function for the browser refresh tool."""

    async def handle_browser_refresh(**params) -> str:
        result = browser_service.refresh()

        if result.get("success", False):
            return f"{result.get('message', 'Page refreshed')}. Current URL: {result.get('current_url', 'Unknown')}"
        else:
            raise RuntimeError(
                f"Refresh failed: {result.get('error', 'Unknown error')}"
            )

    return handle_browser_refresh


def _get_content_delta_changes(browser_service: BrowserAutomationService):
    time.sleep(1)  # wait for page to stabilize
    current_content = browser_service.get_page_content()
    differ = difflib.Differ()
    _last_page_content_lines = browser_service._last_page_content.splitlines()
    try:
        cutoff_idx = _last_page_content_lines.index("## Clickable Elements")
    except ValueError:
        cutoff_idx = len(_last_page_content_lines)
    diffs = list(
        differ.compare(
            _last_page_content_lines[:cutoff_idx],
            current_content.get("content", "").splitlines(),
        )
    )
    diff_summary = "\n".join([d.lstrip("+- ") for d in diffs if d.startswith("+ ")])
    return diff_summary


def get_browser_execute_script_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "Execute JavaScript in the current browser page context using CDP. "
        "Use this for runtime debugging, rendered DOM inspection, metadata or style extraction, or custom page queries when built-in browser tools are insufficient. "
        "The script body is wrapped in an async IIFE — use `return` to yield a value."
    )
    tool_arguments = {
        "script": {
            "type": "string",
            "description": "JavaScript code to execute. Use `return <expr>` to return a value.",
        },
        "await_promise": {
            "type": "boolean",
            "description": "Whether to await the result if it is a Promise. Defaults to true.",
            "default": True,
        },
    }
    tool_required = ["script"]

    if provider == "claude":
        return {
            "name": "execute_browser_script",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "execute_browser_script",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_execute_script_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    async def handle_execute_browser_script(**params) -> str:
        script = params.get("script", "")
        await_promise = params.get("await_promise", True)

        if not script or not script.strip():
            raise RuntimeError("No script provided for execution.")

        result = browser_service.execute_script(script, await_promise)

        if result.get("success", False):
            result_type = result.get("resultType", "unknown")
            result_value = result.get("result")
            import json

            if isinstance(result_value, (dict, list)):
                formatted = json.dumps(result_value, indent=2, ensure_ascii=False)
            else:
                formatted = (
                    str(result_value) if result_value is not None else "undefined"
                )
            return f"Script executed successfully.\nResult type: {result_type}\nResult:\n{formatted}"
        else:
            error = result.get("error", "Unknown error")
            stack = result.get("stack", "")
            msg = f"Script execution failed: {error}"
            if stack:
                msg += f"\nStack:\n{stack}"
            raise RuntimeError(msg)

    return handle_execute_browser_script


def get_browser_view_console_log_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "View browser console/runtime log entries captured through CDP. "
        "Use this for testing and diagnosing live page behavior, including console.log/warn/error output, JavaScript exceptions, and network errors."
    )
    tool_arguments = {
        "limit": {
            "type": "integer",
            "description": "Maximum number of log entries to return. Defaults to 50.",
            "default": 50,
        },
        "levels": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["error", "warn", "info", "log", "debug"],
            },
            "description": "Filter by log levels. If omitted, all levels are returned.",
        },
        "since_last_read": {
            "type": "boolean",
            "description": "If true, return only entries since last read. Defaults to true.",
            "default": True,
        },
    }
    tool_required: list = []

    if provider == "claude":
        return {
            "name": "view_browser_console_log",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "view_browser_console_log",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_view_console_log_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    async def handle_view_browser_console_log(**params) -> str:
        limit = params.get("limit", 50)
        levels = params.get("levels", None)
        since_last_read = params.get("since_last_read", True)

        result = browser_service.get_console_logs(
            limit=limit,
            levels=levels,
            since_last_read=since_last_read,
        )

        if not result.get("success"):
            raise RuntimeError(result.get("error", "Failed to retrieve console logs"))

        logs = result.get("logs", [])
        count = result.get("count", 0)

        if count == 0:
            mode = "since last read" if since_last_read else "in buffer"
            return f"No browser console log entries {mode}."

        lines = [f"Showing {count} browser console log entries:"]
        for entry in logs:
            ts = entry.get("timestamp", "")
            level = entry.get("level", "LOG")
            source = entry.get("source", "")
            url = entry.get("url", "")
            line_num = entry.get("lineNumber")
            text = entry.get("text", "")

            location = url
            if location and line_num is not None:
                location = f"{url}:{line_num}"

            parts = [f"[{ts}]", level, source]
            if location:
                parts.append(location)
            parts.append(text)
            lines.append(" ".join(parts))

        return "\n".join(lines)

    return handle_view_browser_console_log


def get_browser_element_xpath_tool_definition(provider="claude") -> Dict[str, Any]:
    tool_description = (
        "Get the XPath value for a browser element by its UUID. "
        "Use this for DOM metadata, selector inspection, or debugging on the live rendered page. UUIDs are obtained from get_browser_content or get_browser_elements_by_text."
    )
    tool_arguments = {
        "element_uuid": {
            "type": "string",
            "description": "UUID identifier of the element.",
        }
    }
    tool_required = ["element_uuid"]

    if provider == "claude":
        return {
            "name": "get_browser_element_xpath",
            "description": tool_description,
            "input_schema": {
                "type": "object",
                "properties": tool_arguments,
                "required": tool_required,
            },
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "get_browser_element_xpath",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": tool_arguments,
                    "required": tool_required,
                },
            },
        }


def get_browser_element_xpath_tool_handler(
    browser_service: BrowserAutomationService,
) -> Callable:
    async def handle_get_browser_element_xpath(**params) -> str:
        element_uuid = params.get("element_uuid", "")
        if not element_uuid:
            raise RuntimeError("No element_uuid provided.")

        xpath = browser_service.uuid_to_xpath_mapping.get(element_uuid)
        if not xpath:
            raise RuntimeError(
                f"Element UUID '{element_uuid}' not found. "
                "Please use get_browser_content to get current element UUIDs."
            )

        return f"UUID: {element_uuid}\nXPath: {xpath}"

    return handle_get_browser_element_xpath


def register(service_instance=None, agent=None):
    """Register browser automation tools with the central registry or directly with an agent."""
    from AgentCrew.modules.tools.registration import register_tool

    register_tool(
        get_browser_navigate_tool_definition,
        get_browser_navigate_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_mouse_action_tool_definition,
        get_browser_mouse_action_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_get_content_tool_definition,
        get_browser_get_content_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_keyboard_action_tool_definition,
        get_browser_keyboard_action_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_get_elements_by_text_tool_definition,
        get_browser_get_elements_by_text_tool_handler,
        service_instance,
        agent,
    )
    # register_tool(
    #     get_browser_capture_screenshot_tool_definition,
    #     get_browser_capture_screenshot_tool_handler,
    #     service_instance,
    #     agent,
    # )
    register_tool(
        get_browser_refresh_tool_definition,
        get_browser_refresh_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_execute_script_tool_definition,
        get_browser_execute_script_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_view_console_log_tool_definition,
        get_browser_view_console_log_tool_handler,
        service_instance,
        agent,
    )
    register_tool(
        get_browser_element_xpath_tool_definition,
        get_browser_element_xpath_tool_handler,
        service_instance,
        agent,
    )
