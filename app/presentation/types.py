from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SidebarState:
    active_page: str
    selected_notebook_path: str | None
