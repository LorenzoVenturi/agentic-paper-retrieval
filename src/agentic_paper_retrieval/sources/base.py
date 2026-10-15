"""The interface used by the search agent."""

from typing import Protocol

from agentic_paper_retrieval.models import SearchResult


class SourceError(RuntimeError):
    """A paper source could not complete a request."""


class PaperSource(Protocol):
    def search(self, query: str, limit: int = 10) -> SearchResult: ...
