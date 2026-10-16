"""arXiv search and feed parsing."""

import hashlib
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from agentic_paper_retrieval import __version__
from agentic_paper_retrieval.models import Paper, SearchResult
from agentic_paper_retrieval.sources.base import SourceError

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "open": "http://a9.com/-/spec/opensearch/1.1/",
}
STOP_WORDS = frozenset(
    "a an the and or of in on for to with from by is are was were be been "
    "how what which why can could do does about find show me papers research".split()
)
_REQUEST_LOCK = threading.Lock()
_last_request: float | None = None


def keyword_query(query: str) -> str:
    """Join keywords and quoted phrases into an arXiv AND query."""
    terms = [
        phrase or word
        for phrase, word in re.findall(r'"([^\"]+)"|([\w][\w.+#-]*)', query)
        if phrase or word.casefold() not in STOP_WORDS
    ]
    terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
    if not terms:
        raise ValueError("The query must contain searchable words or a quoted phrase.")
    return " AND ".join(f'all:"{term}"' for term in terms)


def _text(element: ET.Element, path: str) -> str:
    child = element.find(path, NS)
    return " ".join(" ".join(child.itertext()).split()) if child is not None else ""


def parse_feed(data: bytes, query: str, source_query: str) -> SearchResult:
    """Read an arXiv Atom feed into search results."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SourceError("arXiv returned malformed XML.") from exc
    if root.tag != f"{{{NS['atom']}}}feed":
        raise SourceError("arXiv returned an unexpected response instead of an Atom feed.")
    papers: dict[str, Paper] = {}
    for entry in root.findall("atom:entry", NS):
        entry_url = _text(entry, "atom:id")
        if "/api/errors" in entry_url:
            raise SourceError(f"arXiv rejected the query: {_text(entry, 'atom:summary')}")
        parsed = urllib.parse.urlsplit(entry_url)
        title = _text(entry, "atom:title")
        if parsed.hostname != "arxiv.org" or not parsed.path.startswith("/abs/") or not title:
            raise SourceError("arXiv returned a paper without a valid identifier or title.")
        source_id = parsed.path.removeprefix("/abs/")
        if not source_id:
            raise SourceError("arXiv returned an empty paper identifier.")
        paper_id = "arxiv:" + re.sub(r"v\d+$", "", source_id)
        paper = Paper(
            id=paper_id,
            source="arxiv",
            source_id=source_id,
            title=title,
            authors=tuple(_text(author, "atom:name") for author in entry.findall("atom:author", NS)),
            abstract=_text(entry, "atom:summary"),
            published=_text(entry, "atom:published"),
            url=f"https://arxiv.org/abs/{source_id}",
            pdf_url=f"https://arxiv.org/pdf/{source_id}",
            doi=_text(entry, "arxiv:doi") or None,
            categories=tuple(item.attrib["term"] for item in entry.findall("atom:category", NS) if "term" in item.attrib),
        )
        papers.setdefault(paper.id, paper)
    try:
        total = int(_text(root, "open:totalResults"))
        if total < 0:
            raise ValueError
    except ValueError as exc:
        raise SourceError("arXiv returned an invalid result count.") from exc
    return SearchResult(query, source_query, "arxiv", total, tuple(papers.values()))


class ArxivSource:
    def __init__(self, cache_dir: Path | None = Path("data/cache/arxiv"), timeout: float = 30):
        self.cache_dir = cache_dir
        self.timeout = timeout

    def search(self, query: str, limit: int = 10, *, raw_query: bool = False) -> SearchResult:
        query = query.strip()
        if not query:
            raise ValueError("The search query cannot be empty.")
        if not 1 <= limit <= 50:
            raise ValueError("The result limit must be between 1 and 50.")
        source_query = query if raw_query else keyword_query(query)
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": source_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".xml") if self.cache_dir else None
        if cache:
            try:
                if time.time() - cache.stat().st_mtime < 86400:
                    return parse_feed(cache.read_bytes(), query, source_query)
            except (OSError, SourceError):
                pass  # An unusable local cache should not prevent a fresh request.
        data = self._fetch(url)
        result = parse_feed(data, query, source_query)
        if cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except OSError:
                pass  # Search remains usable when the cache directory is read-only.
        return result

    def _fetch(self, url: str) -> bytes:
        global _last_request
        request = urllib.request.Request(url, headers={
            "User-Agent": f"agentic-paper-retrieval/{__version__} (https://github.com/LorenzoVenturi/agentic-paper-retrieval)",
            "Accept": "application/atom+xml",
        })
        with _REQUEST_LOCK:
            if _last_request is not None:
                time.sleep(max(0, 3 - (time.monotonic() - _last_request)))
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read(4_000_001)
                if len(data) > 4_000_000:
                    raise SourceError("arXiv response exceeded the supported size.")
                return data
            except urllib.error.HTTPError as exc:
                raise SourceError(f"arXiv returned HTTP {exc.code}; try again later or refine the query.") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise SourceError(f"Could not reach arXiv: {exc}") from exc
            finally:
                _last_request = time.monotonic()
