import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from agentic_paper_retrieval.library import ChromaPaperLibrary, LibraryError, OllamaEmbedder
from agentic_paper_retrieval.models import Paper


class FixtureEmbedder:
    """Deterministic vectors for testing storage and matching, not embedding quality."""
    model = "fixture-embeddings"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return [[1., 0., 0.] if any(word in text.lower() for word in ("retrieval", "grounding", "rag"))
                else [0., 1., 0.] for text in texts]


def fixture_paper(number="one", title="Retrieval research"):
    return Paper(f"arxiv:{number}", "arxiv", number, title, ("Example Author",),
                 "A fictional abstract", "2025-01-01", f"https://arxiv.org/abs/{number}",
                 categories=("cs.CL",))


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "library"
        self.embedder = FixtureEmbedder()
        self.library = ChromaPaperLibrary(self.path, self.embedder)

    def test_empty_library_needs_no_embedding_model(self):
        self.assertEqual(self.library.search("new question").papers, ())
        self.assertEqual(self.embedder.calls, [])

    def test_save_is_explicit_idempotent_and_keeps_provenance_out_of_embedding(self):
        paper = fixture_paper()
        self.assertEqual(self.library.save((paper, paper), "Original question")["added_count"], 1)
        self.assertEqual(self.library.save((paper,), "Different question")["added_count"], 0)
        self.assertEqual(len(self.embedder.calls), 1)
        self.assertNotIn("Original question", self.embedder.calls[0])
        saved = self.library.list_papers()
        self.assertEqual(saved["total"], 1)
        self.assertEqual(saved["papers"][0]["saved_from_question"], "Original question")
        self.assertEqual(tuple(saved["papers"][0]["authors"]), paper.authors)

    def test_semantic_candidates_survive_a_new_process_and_can_be_removed(self):
        paper = fixture_paper()
        self.library.save((paper, fixture_paper("two", "Quantum circuits")), "Old question")
        code = """
import json, sys
from pathlib import Path
from agentic_paper_retrieval.library import ChromaPaperLibrary
class Embedder:
    model = 'fixture-embeddings'
    def embed(self, texts): return [[1., 0., 0.] for text in texts]
library = ChromaPaperLibrary(Path(sys.argv[1]), Embedder())
print(json.dumps(library.search('Grounding language models in external evidence', limit=1).to_dict()))
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["papers"][0]["id"], paper.id)
        self.library.remove(paper.id)
        self.assertEqual(self.library.list_papers()["total"], 1)

    def test_embedding_failure_does_not_save_partial_records(self):
        with patch.object(self.embedder, "embed", side_effect=LibraryError("offline")):
            with self.assertRaisesRegex(LibraryError, "offline"):
                self.library.save((fixture_paper(),), "q")
        self.assertEqual(self.library.list_papers()["total"], 0)

    def test_embedding_model_changes_are_rejected_before_comparing_vectors(self):
        self.library.save((fixture_paper(),), "q")
        changed = FixtureEmbedder()
        changed.model = "another-model"
        library = ChromaPaperLibrary(self.path, changed)
        with self.assertRaisesRegex(LibraryError, "uses embedding model"):
            library.search("q")
        self.assertEqual(changed.calls, [])
        self.assertEqual(library.list_papers()["total"], 1)

    def test_library_pagination(self):
        self.library.save(tuple(fixture_paper(str(i)) for i in range(3)), "q")
        first = self.library.list_papers(limit=2)
        second = self.library.list_papers(limit=2, offset=2)
        self.assertEqual(first["total"], 3)
        self.assertEqual(len(second["papers"]), 1)
        self.assertFalse({p["id"] for p in first["papers"]} & {p["id"] for p in second["papers"]})


class EmbeddingTests(unittest.TestCase):
    def test_embeddings_use_local_batch_endpoint_and_reject_silent_truncation(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"embeddings":[[1,0],[0,1]]}')) as send:
            self.assertEqual(len(OllamaEmbedder().embed(["first", "second"])), 2)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/embed")
        self.assertFalse(json.loads(request.data)["truncate"])

    def test_invalid_vectors_and_missing_model_are_actionable(self):
        for vectors in ([], [[0, 0]], [[True, 1]], [[float("nan"), 1]], [[1, 2], [1]]):
            with self.subTest(vectors=vectors), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps({"embeddings": vectors}).encode())):
                with self.assertRaisesRegex(LibraryError, "invalid embeddings"):
                    OllamaEmbedder().embed(["question"])
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 404, "missing", {}, None)):
            with self.assertRaisesRegex(LibraryError, "ollama pull embeddinggemma"):
                OllamaEmbedder().embed(["question"])
