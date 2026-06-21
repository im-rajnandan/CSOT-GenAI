"""Hugging Face Papers search and reading tools."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import unquote, urlparse

import requests


HF_BASE_URL = "https://huggingface.co"
DEFAULT_MAX_CHARS = 12_000
MODERN_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
LEGACY_ARXIV_ID = re.compile(r"^[A-Za-z][A-Za-z0-9.-]*/\d{7}$")


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _max_chars() -> int:
    return _bounded_int(os.getenv("MAX_PAPER_CHARS"), DEFAULT_MAX_CHARS, 1, 200_000)


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "ResearchDesk/1.0", "Accept": "application/json"}
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_arxiv_id(value: str) -> str:
    """Return a canonical, versionless arXiv ID from an ID or common URL."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("arxiv_id must be a non-empty string")

    candidate = unquote(value.strip())
    candidate = re.sub(r"^arxiv:\s*", "", candidate, flags=re.IGNORECASE)

    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        hostname = (parsed.hostname or "").lower()
        if hostname not in {"arxiv.org", "www.arxiv.org", "huggingface.co", "www.huggingface.co"}:
            raise ValueError("Only arXiv or Hugging Face paper URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
        if hostname.endswith("arxiv.org"):
            if parts and parts[0] in {"abs", "pdf", "html"}:
                parts = parts[1:]
        elif parts and parts[0] == "papers":
            parts = parts[1:]
        candidate = "/".join(parts)

    candidate = candidate.split("?", 1)[0].split("#", 1)[0]
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip("/")

    if not (MODERN_ARXIV_ID.fullmatch(candidate) or LEGACY_ARXIV_ID.fullmatch(candidate)):
        raise ValueError(f"Invalid arXiv ID: {value}")
    return candidate


def paper_search(query: str, limit: int = 5) -> dict[str, Any]:
    """Search the Hugging Face Papers index and return compact results."""
    query = str(query or "").strip()
    if not query:
        return {"error": "empty paper search query"}
    limit = _bounded_int(limit, 5, 1, 10)

    try:
        response = requests.get(
            f"{HF_BASE_URL}/api/papers/search",
            params={"q": query, "limit": limit},
            headers=_headers(),
            timeout=15,
        )
        if response.status_code == 429:
            return {"error": "Hugging Face paper search rate limit reached"}
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"error": f"paper_search request failed: {exc}"}
    except ValueError as exc:
        return {"error": f"paper_search returned invalid JSON: {exc}"}

    if isinstance(payload, dict):
        items = payload.get("papers") or payload.get("results") or []
    elif isinstance(payload, list):
        items = payload
    else:
        return {"error": "paper_search returned an unexpected response shape"}

    papers = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
        arxiv_id = str(paper.get("id") or paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        abstract = str(paper.get("summary") or paper.get("abstract") or "").strip()
        if len(abstract) > 1_200:
            abstract = abstract[:1_200] + "\n[abstract truncated]"
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": str(paper.get("title") or "Untitled").strip(),
                "abstract": abstract,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )

    return {"content": papers, "query": query, "count": len(papers)}


def read_paper(arxiv_id: str) -> dict[str, Any]:
    """Read paper metadata and markdown, falling back to the abstract."""
    try:
        canonical_id = normalize_arxiv_id(arxiv_id)
    except ValueError as exc:
        return {"error": str(exc)}

    arxiv_url = f"https://arxiv.org/abs/{canonical_id}"
    headers = _headers()
    try:
        metadata_response = requests.get(
            f"{HF_BASE_URL}/api/papers/{canonical_id}",
            headers=headers,
            timeout=15,
        )
        if metadata_response.status_code == 404:
            return {
                "error": "Paper is not indexed by Hugging Face Papers",
                "arxiv_id": canonical_id,
                "fallback_url": arxiv_url,
            }
        if metadata_response.status_code == 429:
            return {"error": "Hugging Face paper read rate limit reached", "fallback_url": arxiv_url}
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
    except requests.RequestException as exc:
        return {"error": f"read_paper metadata request failed: {exc}", "fallback_url": arxiv_url}
    except ValueError as exc:
        return {"error": f"read_paper metadata was invalid JSON: {exc}", "fallback_url": arxiv_url}

    title = str(metadata.get("title") or "Untitled").strip()
    abstract = str(metadata.get("summary") or metadata.get("abstract") or "").strip()
    content = ""
    source = "abstract"

    markdown_headers = dict(headers)
    markdown_headers["Accept"] = "text/markdown,text/plain;q=0.9,*/*;q=0.1"
    try:
        markdown_response = requests.get(
            f"{HF_BASE_URL}/papers/{canonical_id}.md",
            headers=markdown_headers,
            timeout=30,
        )
        if markdown_response.status_code == 200 and markdown_response.text.strip():
            content = markdown_response.text.strip()
            source = "markdown"
    except requests.RequestException:
        pass

    if not content:
        content = abstract
    if not content:
        return {
            "error": "Paper metadata was found, but neither markdown nor an abstract was available",
            "arxiv_id": canonical_id,
            "fallback_url": arxiv_url,
        }

    max_chars = _max_chars()
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + "\n\n[paper content truncated]"

    return {
        "content": content,
        "title": title,
        "arxiv_id": canonical_id,
        "abstract": abstract,
        "url": arxiv_url,
        "source": source,
        "truncated": truncated,
    }
