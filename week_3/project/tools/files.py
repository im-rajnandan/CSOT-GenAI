"""Sandboxed file tools used by Research Desk."""

from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MAX_CHARS = 12_000
DEFAULT_LIST_LIMIT = 200
PRIVATE_PATHS = {".agent", ".env", ".git"}


def _bounded_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _max_chars() -> int:
    return _bounded_int(os.getenv("MAX_FILE_CHARS"), DEFAULT_MAX_CHARS)


def _workspace_root(workspace_root: str | os.PathLike[str] | None = None) -> Path:
    configured = workspace_root if workspace_root is not None else os.getenv("WORKSPACE_ROOT", ".")
    return Path(configured).expanduser().resolve()


def resolve_path(
    path: str,
    workspace_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve *path* and require it to remain inside the workspace."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string")

    root = _workspace_root(workspace_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escapes workspace") from exc
    if relative.parts and relative.parts[0] in PRIVATE_PATHS:
        raise ValueError("Path is private")
    return candidate


def _relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    return value or "."


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_file(
    path: str,
    start_line: int = 1,
    read_lines: int = 200,
    *,
    workspace_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read a numbered window of a UTF-8 text file."""
    try:
        start_line = int(start_line)
        read_lines = int(read_lines)
        if start_line < 1:
            raise ValueError("start_line must be at least 1")
        if read_lines < 1:
            raise ValueError("read_lines must be at least 1")

        root = _workspace_root(workspace_root)
        full_path = resolve_path(path, root)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not full_path.is_file():
            raise ValueError(f"Not a file: {path}")

        text = full_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines and start_line > total_lines:
            raise ValueError(f"start_line {start_line} is beyond the file's {total_lines} lines")

        selected = lines[start_line - 1 : start_line - 1 + read_lines]
        max_chars = _max_chars()
        rendered: list[str] = []
        char_count = 0
        char_truncated = False

        for offset, line in enumerate(selected):
            line_number = start_line + offset
            numbered = f"{line_number:>6}| {line}"
            addition = numbered if not rendered else f"\n{numbered}"
            if char_count + len(addition) > max_chars:
                remaining = max_chars - char_count
                if remaining > 0:
                    rendered.append(addition[:remaining] if rendered else numbered[:remaining])
                char_truncated = True
                break
            rendered.append(addition if rendered else numbered)
            char_count += len(addition)

        content = "".join(rendered)
        included_lines = len(rendered)
        if char_truncated and included_lines:
            end_line = start_line + included_lines - 1
        elif selected:
            end_line = start_line + len(selected) - 1
        else:
            end_line = 0
        has_more = char_truncated or (end_line > 0 and end_line < total_lines)

        return {
            "content": content,
            "path": _relative(full_path, root),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "has_more": has_more,
            "truncated": char_truncated,
        }
    except (OSError, UnicodeError, ValueError) as exc:
        return {"error": str(exc)}


def write_file(
    path: str,
    content: str,
    *,
    workspace_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create or replace a UTF-8 file inside the workspace."""
    try:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        root = _workspace_root(workspace_root)
        full_path = resolve_path(path, root)
        if full_path.exists() and not full_path.is_file():
            raise ValueError(f"Not a file: {path}")
        _atomic_write(full_path, content)
        return {
            "content": f"Wrote {_relative(full_path, root)}",
            "path": _relative(full_path, root),
            "characters": len(content),
        }
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}


def edit_file(
    path: str,
    operation: str,
    start_line: int,
    end_line: int | None = None,
    content: str | None = None,
    *,
    workspace_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Replace, delete, or append lines and return a unified diff preview."""
    try:
        operation = str(operation or "").lower().strip()
        if operation not in {"replace", "delete", "append"}:
            raise ValueError("operation must be replace, delete, or append")

        start_line = int(start_line)
        root = _workspace_root(workspace_root)
        full_path = resolve_path(path, root)
        if not full_path.exists() or not full_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        old_text = full_path.read_text(encoding="utf-8")
        old_lines = old_text.splitlines()
        had_trailing_newline = old_text.endswith(("\n", "\r"))
        new_lines = list(old_lines)

        if operation in {"replace", "delete"}:
            if end_line is None:
                raise ValueError(f"end_line is required for {operation}")
            end_line = int(end_line)
            if start_line < 1 or end_line < start_line or end_line > len(old_lines):
                raise ValueError("Invalid inclusive line range")
            replacement = []
            if operation == "replace":
                if content is None:
                    raise ValueError("content is required for replace")
                replacement = content.splitlines()
            new_lines[start_line - 1 : end_line] = replacement
        else:
            if start_line < 0 or start_line > len(old_lines):
                raise ValueError("append start_line must be between 0 and the file length")
            if content is None:
                raise ValueError("content is required for append")
            new_lines[start_line:start_line] = content.splitlines()

        trailing_newline = had_trailing_newline
        if not old_lines and content:
            trailing_newline = content.endswith(("\n", "\r"))
        new_text = "\n".join(new_lines)
        if trailing_newline and new_lines:
            new_text += "\n"

        relative_path = _relative(full_path, root)
        diff = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
        max_chars = _max_chars()
        diff_truncated = len(diff) > max_chars
        if diff_truncated:
            diff = diff[:max_chars] + "\n[diff truncated]"

        _atomic_write(full_path, new_text)
        return {
            "content": f"Edited {relative_path}",
            "path": relative_path,
            "operation": operation,
            "line_count": len(new_lines),
            "diff": diff,
            "truncated": diff_truncated,
        }
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return {"error": str(exc)}


def list_files(
    path: str = ".",
    pattern: str = "*",
    *,
    workspace_root: str | os.PathLike[str] | None = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> dict[str, Any]:
    """List sorted workspace entries matching a glob pattern."""
    try:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("pattern must be a non-empty glob string")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError("Glob pattern escapes workspace")
        limit = _bounded_int(limit, DEFAULT_LIST_LIMIT)

        root = _workspace_root(workspace_root)
        base = resolve_path(path, root)
        if not base.exists() or not base.is_dir():
            raise ValueError(f"Not a directory: {path}")

        entries: list[dict[str, str]] = []
        for candidate in base.glob(pattern):
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            relative_parts = Path(relative).parts
            if relative_parts and relative_parts[0] in PRIVATE_PATHS:
                continue
            entries.append({"path": relative, "type": "directory" if resolved.is_dir() else "file"})

        entries.sort(key=lambda item: item["path"])
        has_more = len(entries) > limit
        return {
            "content": entries[:limit],
            "path": _relative(base, root),
            "pattern": pattern,
            "has_more": has_more,
        }
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}
