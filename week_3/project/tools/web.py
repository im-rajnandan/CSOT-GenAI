"""Web search and readable-page fetching tools."""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura
from markdownify import markdownify as html_to_markdown


DEFAULT_MAX_CHARS = 8_000


def _bounded_int(value: Any, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    return min(maximum, parsed) if maximum is not None else parsed


def _max_chars() -> int:
    return _bounded_int(os.getenv("MAX_WEB_CHARS"), DEFAULT_MAX_CHARS, 1, 200_000)


def html_to_readable_text(html: str) -> tuple[str, str]:
    """Extract article text, retaining Markdown as a structural fallback."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if text:
        return text.strip(), "trafilatura"

    markdown = html_to_markdown(
        html,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "nav", "footer"],
    )
    lines: list[str] = []
    previous_blank = False
    for line in markdown.splitlines():
        clean = line.rstrip()
        if not clean:
            if not previous_blank:
                lines.append("")
            previous_blank = True
        else:
            lines.append(clean)
            previous_blank = False
    return "\n".join(lines).strip(), "markdownify"


def _url_error(url: str) -> str | None:
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return "Invalid URL"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "Please provide a full http/https URL"

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in {"localhost", "0.0.0.0"} or hostname.endswith((".localhost", ".local")):
        return "Local and private URLs are blocked"
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return None
    if not address.is_global:
        return "Local and private URLs are blocked"
    return None


def web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search Serper and return compact organic results."""
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return {"error": "SERPER_API_KEY missing in .env"}
    query = str(query or "").strip()
    if not query:
        return {"error": "empty search query"}
    num_results = _bounded_int(num_results, 5, 1, 10)

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"error": f"web_search request failed: {exc}"}
    except ValueError as exc:
        return {"error": f"web_search returned invalid JSON: {exc}"}

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in payload.get("organic", [])[:num_results]
        if isinstance(item, dict)
    ]
    return {"content": results, "query": query, "count": len(results)}


def web_fetch(url: str) -> dict[str, Any]:
    """Fetch a page and return bounded, readable content."""
    url = str(url or "").strip()
    validation_error = _url_error(url)
    if validation_error:
        return {"error": validation_error}

    parsed = urlparse(url)
    headers = {"User-Agent": "ResearchDesk/1.0"}
    llms_url = f"{parsed.scheme}://{parsed.netloc}/llms.txt"
    llms_text = ""

    if parsed.path.rstrip("/") != "/llms.txt":
        try:
            llms_response = requests.get(llms_url, headers=headers, timeout=5)
            if llms_response.status_code == 200:
                llms_text = llms_response.text.strip()
        except requests.RequestException:
            pass

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        final_url_error = _url_error(response.url)
        if final_url_error:
            return {"error": "Redirected to a blocked URL"}
        if parsed.path.rstrip("/") == "/llms.txt":
            text, extractor = response.text.strip(), "llms.txt"
        else:
            text, extractor = html_to_readable_text(response.text)
    except requests.RequestException as exc:
        if llms_text:
            max_chars = _max_chars()
            truncated = len(llms_text) > max_chars
            content = llms_text[:max_chars]
            if truncated:
                content += "\n\n[content truncated]"
            return {
                "content": content,
                "url": llms_url,
                "extractor": "llms.txt",
                "truncated": truncated,
            }
        return {"error": f"web_fetch request failed: {exc}"}

    if not text:
        if llms_text:
            text, extractor = llms_text, "llms.txt"
        else:
            return {"error": "Page opened, but readable text was not found"}

    max_chars = _max_chars()
    if llms_text and extractor != "llms.txt":
        guide_limit = min(2_000, max(500, max_chars // 4))
        guide = llms_text[:guide_limit]
        if len(llms_text) > guide_limit:
            guide += "\n[site guide truncated]"
        text = f"[Site guide: {llms_url}]\n{guide}\n\n---\n\n[Requested page: {response.url}]\n{text}"

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[content truncated]"
    result = {
        "content": text,
        "url": response.url,
        "extractor": extractor,
        "truncated": truncated,
    }
    if llms_text:
        result["llms_txt"] = llms_url
    return result
