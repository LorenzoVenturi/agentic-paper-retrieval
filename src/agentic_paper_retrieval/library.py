"""An explicitly saved paper collection, persisted and searched with Chroma."""

import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from agentic_paper_retrieval.models import Paper, SearchResult
from agentic_paper_retrieval.sources.base import SourceError


class LibraryError(SourceError):
    """The saved library or its embedding model could not be used."""


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class PaperLibrary(Protocol):
    def search(self, query: str, limit: int = 5) -> SearchResult: ...

    def save(self, papers: tuple[Paper, ...], question: str) -> dict: ...

    def list_papers(self, limit: int = 50, offset: int = 0) -> dict: ...

    def remove(self, paper_id: str) -> None: ...


class OllamaEmbedder:
    def __init__(self, model="embeddinggemma", base_url="http://localhost:11434", timeout=180):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("The Ollama URL must be an HTTP(S) server URL without a path or query.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Choose an Ollama embedding model.")
        self.model, self.base_url, self.timeout = model.strip(), base_url.rstrip("/"), timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = urllib.request.Request(
            self.base_url + "/api/embed",
            data=json.dumps({"model": self.model, "input": texts, "truncate": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise LibraryError("Ollama returned an oversized embedding response.")
            payload = json.loads(raw)
            vectors = payload["embeddings"]
            if payload.get("error"):
                raise LibraryError(str(payload["error"]))
            _validate_vectors(vectors, len(texts))
            return vectors
        except urllib.error.HTTPError as exc:
            raise LibraryError(
                f"Could not generate embeddings with {self.model!r} (HTTP {exc.code}). "
                f"Install it with ollama pull {self.model}; check the model and input length."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LibraryError(f"Could not reach Ollama for library embeddings: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LibraryError("Ollama returned invalid embeddings.") from exc


def _validate_vectors(vectors, count):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Unexpected embedding count")
    dimension = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Empty embedding")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise ValueError("Invalid embedding values")
        if not any(vector) or (dimension is not None and len(vector) != dimension):
            raise ValueError("Invalid embedding dimensions")
        dimension = len(vector)


def _paper(metadata: dict) -> Paper:
    record = json.loads(metadata["paper_json"])
    return Paper(**{**record, "authors": tuple(record["authors"]), "categories": tuple(record.get("categories", ()))})


@dataclass
class ChromaPaperLibrary:
    path: Path
    embedder: Embedder

    def __post_init__(self):
        self.path = Path(self.path)
        self._lock = threading.RLock()
        self._collection = None
        self._client = None

    def _open(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="saved-papers", embedding_function=None,
                metadata={"schema_version": 1, "embedding_model": self.embedder.model},
                configuration={"hnsw": {"space": "cosine"}},
            )
        return self._collection

    def _check_model(self, collection):
        if collection.metadata.get("embedding_model") != self.embedder.model:
            raise LibraryError(
                f"This library uses embedding model {collection.metadata.get('embedding_model')!r}. "
                "Use that model or choose a different library directory."
            )

    def save(self, papers: tuple[Paper, ...], question: str) -> dict:
        unique = {paper.id: paper for paper in papers}
        if not unique or len(unique) > 100:
            raise ValueError("Choose between 1 and 100 papers to save.")
        with self._lock:
            try:
                collection = self._open()
                self._check_model(collection)
                existing = set(collection.get(ids=list(unique), include=[])["ids"])
                new = [paper for paper in unique.values() if paper.id not in existing]
                documents = [f"Title: {paper.title}\n\nAbstract: {paper.abstract}" for paper in new]
                vectors = []
                for start in range(0, len(documents), 16):
                    vectors.extend(self.embedder.embed(documents[start:start + 16]))
                if new:
                    _validate_vectors(vectors, len(new))
                    saved_at = datetime.now(timezone.utc).isoformat()
                    collection.upsert(
                        ids=[paper.id for paper in new], documents=documents, embeddings=vectors,
                        metadatas=[{
                            "paper_json": json.dumps(paper.to_dict(), ensure_ascii=False),
                            "title": paper.title, "source": paper.source,
                            "published": paper.published, "url": paper.url,
                            "saved_at": saved_at, "saved_from_question": question,
                        } for paper in new],
                    )
                return {"saved_paper_ids": list(unique), "added_count": len(new), "total": collection.count()}
            except LibraryError:
                raise
            except Exception as exc:
                raise LibraryError(f"Could not save papers to the library: {exc}") from exc

    def search(self, query: str, limit: int = 5) -> SearchResult:
        if not query.strip() or not 1 <= limit <= 20:
            raise ValueError("Provide a library query and a limit from 1 to 20.")
        with self._lock:
            try:
                collection = self._open()
                total = collection.count()
                if not total:
                    return SearchResult(query, query, "library", 0, ())
                self._check_model(collection)
                result = collection.query(
                    query_embeddings=self.embedder.embed([query]),
                    n_results=min(limit, total), include=["metadatas"],
                )
                papers = tuple(_paper(metadata) for metadata in result["metadatas"][0])
                return SearchResult(query, query, "library", total, papers)
            except LibraryError:
                raise
            except Exception as exc:
                raise LibraryError(f"Could not search the saved library: {exc}") from exc

    def list_papers(self, limit: int = 50, offset: int = 0) -> dict:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Use a library page size from 1 to 100 and a nonnegative offset.")
        with self._lock:
            try:
                collection = self._open()
                result = collection.get(limit=limit, offset=offset, include=["metadatas"])
                return {"total": collection.count(), "papers": [
                    {**_paper(metadata).to_dict(), "saved_at": metadata["saved_at"],
                     "saved_from_question": metadata["saved_from_question"]}
                    for metadata in result["metadatas"]
                ]}
            except Exception as exc:
                raise LibraryError(f"Could not read the saved library: {exc}") from exc

    def remove(self, paper_id: str) -> None:
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("Choose a saved paper to remove.")
        with self._lock:
            try:
                self._open().delete(ids=[paper_id])
            except Exception as exc:
                raise LibraryError(f"Could not remove the saved paper: {exc}") from exc
