from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_snippet(value: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", value or "")).split())


@dataclass(frozen=True)
class ExternalCandidate:
    provider: str
    source_id: str
    title: str
    snippet: str
    url: str
    retrieved_at: float

    def compact(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "case_id": self.source_id,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "candidate_only": True,
        }


@dataclass(frozen=True)
class ExternalCaseEvidence:
    provider: str
    source_id: str
    title: str
    url: str
    extract: str
    retrieved_at: float

    def packet(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "case_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "evidence": self.extract,
            "evidence_scope": "bounded_source_extract",
            "normalization_status": "not_yet_casecard",
            "notice": (
                "This is source evidence, not a validated analogy. The LLM must decide whether "
                "the relational mechanism fits before recording or endorsing it."
            ),
        }


class WikipediaAdapter:
    """Small external candidate/evidence adapter over the MediaWiki Action API.

    Responsibilities are intentionally narrow:
    - search discovers pages that may be worth inspecting;
    - read fetches a bounded plain-text extract;
    - it does not infer analogy structure or decide whether a page is a good match.
    """

    endpoint = "https://en.wikipedia.org/w/api.php"
    article_base = "https://en.wikipedia.org/?curid="

    def __init__(self, *, timeout_seconds: float = 8.0, user_agent: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent or os.environ.get(
            "ECHO_WIKIMEDIA_USER_AGENT",
            "EchoAgentHarness/0.18 (read-only Wikimedia client)",
        )

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("wikipedia_query_required")
        if not 1 <= limit <= 8:
            raise ValueError("wikipedia_limit_must_be_1_to_8")

        payload = await asyncio.to_thread(
            self._request_json,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": "0",
                "srlimit": str(limit),
                "format": "json",
                "formatversion": "2",
            },
        )
        rows = payload.get("query", {}).get("search", [])
        now = time.time()
        out: list[dict[str, Any]] = []
        for row in rows:
            pageid = row.get("pageid")
            title = row.get("title")
            if pageid is None or not title:
                continue
            candidate = ExternalCandidate(
                provider="wikipedia",
                source_id=str(pageid),
                title=str(title),
                snippet=_clean_snippet(str(row.get("snippet", ""))),
                url=f"{self.article_base}{pageid}",
                retrieved_at=now,
            )
            out.append(candidate.compact())
        return out

    async def read(self, source_id: str, *, max_chars: int = 1200) -> dict[str, Any]:
        source_id = str(source_id).strip()
        if not source_id.isdigit():
            raise ValueError(f"invalid_wikipedia_page_id: {source_id}")
        if not 200 <= max_chars <= 1200:
            raise ValueError("wikipedia_max_chars_must_be_200_to_1200")

        payload = await asyncio.to_thread(
            self._request_json,
            {
                "action": "query",
                "prop": "extracts",
                "pageids": source_id,
                "explaintext": "1",
                "exchars": str(max_chars),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise ValueError(f"wikipedia_page_not_found: {source_id}")
        page = pages[0]
        title = str(page.get("title", "")).strip()
        extract = " ".join(str(page.get("extract", "")).split())
        if not title or not extract:
            raise ValueError(f"wikipedia_page_has_no_readable_extract: {source_id}")

        evidence = ExternalCaseEvidence(
            provider="wikipedia",
            source_id=source_id,
            title=title,
            url=f"{self.article_base}{source_id}",
            extract=extract,
            retrieved_at=time.time(),
        )
        return evidence.packet()

    def _request_json(self, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.endpoint}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ValueError(f"external_search_unavailable: wikipedia ({type(exc).__name__})") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("external_search_invalid_response: wikipedia") from exc
        if "error" in payload:
            code = payload.get("error", {}).get("code", "unknown")
            raise ValueError(f"external_search_error: wikipedia:{code}")
        return payload
