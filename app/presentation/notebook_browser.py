from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def discover_notebooks(notebooks_dir: str = "notebooks") -> list[Path]:
    root = Path(notebooks_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.ipynb"))


def summarize_notebook(notebook_path: Path) -> dict[str, int | str | None]:
    summary: dict[str, int | str | None] = {
        "total_cells": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "code_cells_with_outputs": 0,
        "first_heading": None,
        "error": None,
    }

    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        summary["error"] = f"Failed to read notebook: {error}"
        return summary

    return summarize_payload(payload)


def summarize_payload(payload: dict[str, Any]) -> dict[str, int | str | None]:
    summary: dict[str, int | str | None] = {
        "total_cells": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "code_cells_with_outputs": 0,
        "first_heading": None,
        "error": None,
    }

    cells = payload.get("cells", [])
    summary["total_cells"] = len(cells)

    for cell in cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])

        if cell_type == "code":
            summary["code_cells"] = int(summary["code_cells"]) + 1
            if cell.get("outputs"):
                summary["code_cells_with_outputs"] = int(summary["code_cells_with_outputs"]) + 1
        elif cell_type == "markdown":
            summary["markdown_cells"] = int(summary["markdown_cells"]) + 1
            if summary["first_heading"] is None:
                lines = source if isinstance(source, list) else [str(source)]
                for line in lines:
                    stripped = str(line).strip()
                    if stripped.startswith("#"):
                        summary["first_heading"] = stripped
                        break

    return summary


def read_notebook_payload(notebook_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return None, f"Failed to read notebook: {error}"

    if not isinstance(payload, dict):
        return None, "Notebook payload is not a valid JSON object."

    return payload, None


def _extract_headings(payload: dict[str, Any], max_items: int = 8) -> list[str]:
    headings: list[str] = []
    cells = payload.get("cells", [])

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue

        source = cell.get("source", [])
        lines = source if isinstance(source, list) else [str(source)]
        for line in lines:
            stripped = str(line).strip()
            if stripped.startswith("#"):
                headings.append(stripped)
                break

        if len(headings) >= max_items:
            break

    return headings


def _extract_heading_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    heading_rows: list[dict[str, Any]] = []
    cells = payload.get("cells", [])

    for cell_index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "markdown":
            continue

        source = cell.get("source", [])
        lines = source if isinstance(source, list) else [str(source)]
        for line in lines:
            stripped = str(line).strip()
            if not stripped.startswith("#"):
                continue

            level = 0
            for character in stripped:
                if character == "#":
                    level += 1
                else:
                    break

            heading_rows.append(
                {
                    "cell_number": cell_index,
                    "level": level,
                    "heading": stripped[level:].strip(),
                }
            )
            break

    return heading_rows


def _cell_source_preview(cell: dict[str, Any], max_len: int = 120) -> str:
    source = cell.get("source", [])
    lines = source if isinstance(source, list) else [str(source)]
    merged = " ".join(str(line).strip() for line in lines if str(line).strip())
    if not merged:
        return "(empty cell)"
    return merged[:max_len] + ("..." if len(merged) > max_len else "")


def _collect_output_kinds(outputs: list[Any]) -> str:
    kinds: set[str] = set()
    for output in outputs:
        output_type = str(output.get("output_type", "unknown"))
        kinds.add(output_type)

        data_obj = output.get("data", {})
        if isinstance(data_obj, dict):
            for mime_type in data_obj.keys():
                kinds.add(str(mime_type))

    return ", ".join(sorted(kinds)) if kinds else "-"


def _extract_cell_structure(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = payload.get("cells", [])

    for cell_index, cell in enumerate(cells, start=1):
        cell_type = str(cell.get("cell_type", "unknown"))
        outputs = cell.get("outputs", []) if cell_type == "code" else []
        execution_count = cell.get("execution_count") if cell_type == "code" else None

        rows.append(
            {
                "cell_number": cell_index,
                "type": cell_type,
                "execution_count": execution_count,
                "has_outputs": bool(outputs),
                "output_kinds": _collect_output_kinds(outputs),
                "source_preview": _cell_source_preview(cell),
            }
        )

    return rows


def _extract_saved_outputs(payload: dict[str, Any], max_cells: int = 6) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    cells = payload.get("cells", [])

    for cell_index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        output_types: list[str] = []
        text_preview = ""

        for output in outputs:
            output_type = str(output.get("output_type", "unknown"))
            output_types.append(output_type)

            if not text_preview:
                text_data = output.get("text")
                if isinstance(text_data, list):
                    text_preview = "".join(str(item) for item in text_data).strip()
                elif isinstance(text_data, str):
                    text_preview = text_data.strip()

                if not text_preview:
                    data_obj = output.get("data", {})
                    if isinstance(data_obj, dict):
                        if "text/plain" in data_obj:
                            plain = data_obj["text/plain"]
                            if isinstance(plain, list):
                                text_preview = "".join(str(item) for item in plain).strip()
                            elif isinstance(plain, str):
                                text_preview = plain.strip()

        previews.append(
            {
                "cell_number": cell_index,
                "output_types": sorted(set(output_types)),
                "text_preview": text_preview[:300] if text_preview else "(non-text output)",
            }
        )

        if len(previews) >= max_cells:
            break

    return previews


def _normalize_source(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return str(source or "")


def _normalize_mime_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value or "")


def _try_decode_base64_image(data: Any) -> bytes | None:
    raw = _normalize_mime_text(data).strip()
    if not raw:
        return None
    try:
        return base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        return None


def _render_output_block(output: dict[str, Any]) -> None:
    output_type = str(output.get("output_type", "unknown"))

    if output_type == "stream":
        text = _normalize_mime_text(output.get("text", ""))
        if text.strip():
            st.code(text, language="text")
        return

    if output_type == "error":
        traceback_lines = output.get("traceback", [])
        traceback_text = _normalize_mime_text(traceback_lines)
        if traceback_text.strip():
            st.error("Error output")
            st.code(traceback_text, language="text")
        return

    data_obj = output.get("data", {})
    if not isinstance(data_obj, dict):
        st.code(str(output), language="text")
        return

    if "image/png" in data_obj:
        image_bytes = _try_decode_base64_image(data_obj["image/png"])
        if image_bytes:
            st.image(image_bytes, use_container_width=True)

    if "image/jpeg" in data_obj:
        image_bytes = _try_decode_base64_image(data_obj["image/jpeg"])
        if image_bytes:
            st.image(image_bytes, use_container_width=True)

    if "text/html" in data_obj:
        html_text = _normalize_mime_text(data_obj["text/html"]).strip()
        if html_text:
            st.markdown(html_text, unsafe_allow_html=True)

    if "application/json" in data_obj:
        json_obj = data_obj["application/json"]
        st.json(json_obj)

    if "text/markdown" in data_obj:
        markdown_text = _normalize_mime_text(data_obj["text/markdown"]).strip()
        if markdown_text:
            st.markdown(markdown_text)

    if "text/plain" in data_obj:
        plain_text = _normalize_mime_text(data_obj["text/plain"]).strip()
        if plain_text:
            st.code(plain_text, language="text")


def _render_jupyter_style(payload: dict[str, Any]) -> None:
    cells = payload.get("cells", [])

    for cell_index, cell in enumerate(cells, start=1):
        cell_type = str(cell.get("cell_type", "unknown"))
        source_text = _normalize_source(cell.get("source", []))

        if cell_type == "markdown":
            st.markdown(source_text if source_text.strip() else "")
            st.caption(f"Cell {cell_index} - markdown")
            continue

        if cell_type == "code":
            execution_count = cell.get("execution_count")
            prompt = execution_count if execution_count is not None else " "
            st.markdown(f"**In [{prompt}]**")
            st.code(source_text, language="python")

            outputs = cell.get("outputs", [])
            for output in outputs:
                _render_output_block(output)

            st.caption(f"Cell {cell_index} - code")
            continue

        st.markdown(f"**Cell {cell_index} - {cell_type}**")
        st.code(source_text, language="text")


def _render_structured(payload: dict[str, Any], notebook_path: Path) -> None:
    summary = summarize_payload(payload)
    st.caption(
        "Cells: "
        f"{summary['total_cells']} total | "
        f"{summary['markdown_cells']} markdown | "
        f"{summary['code_cells']} code | "
        f"{summary['code_cells_with_outputs']} with saved outputs"
    )

    heading_rows = _extract_heading_records(payload)
    if heading_rows:
        with st.expander("Section map", expanded=True):
            st.dataframe(pd.DataFrame(heading_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No markdown headings found.")

    cell_rows = _extract_cell_structure(payload)
    with st.expander("Full cell structure", expanded=True):
        st.dataframe(pd.DataFrame(cell_rows), use_container_width=True, hide_index=True)

    if cell_rows:
        selected_cell_number = st.number_input(
            "Inspect cell number",
            min_value=1,
            max_value=len(cell_rows),
            value=1,
            step=1,
            key=f"inspect_cell_{notebook_path}",
        )
        selected_index = int(selected_cell_number) - 1
        selected_cell = payload.get("cells", [])[selected_index]
        selected_cell_type = selected_cell.get("cell_type", "unknown")
        selected_source = selected_cell.get("source", [])

        st.markdown(f"**Cell {selected_cell_number} ({selected_cell_type})**")
        st.code(
            "".join(selected_source) if isinstance(selected_source, list) else str(selected_source),
            language="python" if selected_cell_type == "code" else "markdown",
        )

    output_previews = _extract_saved_outputs(payload)
    if output_previews:
        with st.expander("Saved outputs preview", expanded=True):
            for preview in output_previews:
                st.markdown(
                    f"**Cell {preview['cell_number']}** | output types: {', '.join(preview['output_types'])}"
                )
                st.code(preview["text_preview"], language="text")
    else:
        st.info("No saved code-cell outputs found in this notebook.")


def _build_notebook_labels(notebook_paths: list[Path]) -> dict[str, Path]:
    notebook_map: dict[str, Path] = {}
    collisions: dict[str, int] = {}

    for path in notebook_paths:
        base_name = path.stem.replace("_", " ").replace("-", " ").strip().title() or "Notebook"
        collisions[base_name] = collisions.get(base_name, 0) + 1

        if collisions[base_name] == 1:
            label = base_name
        else:
            label = f"{base_name} ({collisions[base_name]})"

        notebook_map[label] = path

    return notebook_map


def _on_notebook_selected() -> None:
    st.session_state["active_page"] = "notebook_preview"


def render_sidebar_notebook_browser(notebooks_dir: str = "notebooks") -> None:
    if "show_notebook_dropdown" not in st.session_state:
        st.session_state["show_notebook_dropdown"] = False

    notebook_paths = discover_notebooks(notebooks_dir)
    if not notebook_paths:
        st.info("No notebooks found under notebooks/.")
        st.session_state["selected_notebook_path"] = None
        return

    notebook_map = _build_notebook_labels(notebook_paths)
    labels = list(notebook_map.keys())

    selected_path_raw = st.session_state.get("selected_notebook_path")
    if not selected_path_raw:
        st.session_state["selected_notebook_path"] = str(notebook_map[labels[0]])
        selected_path_raw = st.session_state["selected_notebook_path"]

    chevron = "▾" if st.session_state["show_notebook_dropdown"] else "▸"
    if st.button(f"{chevron} Notebooks", use_container_width=True, key="browse_notebooks_toggle"):
        st.session_state["show_notebook_dropdown"] = not st.session_state["show_notebook_dropdown"]

    if st.session_state["show_notebook_dropdown"]:
        for idx, label in enumerate(labels):
            path = notebook_map[label]

            if st.button(
                label,
                use_container_width=False,
                key=f"notebook_option_{idx}",
                type="secondary",
            ):
                st.session_state["selected_notebook_path"] = str(path)
                st.session_state["show_notebook_dropdown"] = False
                _on_notebook_selected()


def render_notebook_preview() -> None:
    selected_path_raw = st.session_state.get("selected_notebook_path")
    if not selected_path_raw:
        st.info("Select a notebook from the sidebar to preview it.")
        return

    notebook_path = Path(selected_path_raw)
    payload, error = read_notebook_payload(notebook_path)

    if error:
        st.warning(error)
        return

    assert payload is not None
    _render_jupyter_style(payload)
