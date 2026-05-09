from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AcpSessionContext:
    conn: Any
    session_id: str
    client_capabilities: Any = None
    active_terminals: dict[str, str] = field(default_factory=dict)


_current_acp_session: contextvars.ContextVar[AcpSessionContext | None] = (
    contextvars.ContextVar("current_acp_session", default=None)
)
