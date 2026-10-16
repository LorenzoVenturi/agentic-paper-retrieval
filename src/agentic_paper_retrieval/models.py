"""Paper and search-result records."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Paper:
    id: str
    source: str
    source_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: str
    url: str
    pdf_url: str | None = None
    doi: str | None = None
    categories: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    query: str
    source_query: str
    source: str
    total_results: int
    papers: tuple[Paper, ...]

    def to_dict(self) -> dict:
        return asdict(self)
