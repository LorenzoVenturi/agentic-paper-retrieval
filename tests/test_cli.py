import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_paper_retrieval.__main__ import main
from agentic_paper_retrieval.sources.arxiv import parse_feed
from agentic_paper_retrieval.sources.base import SourceError
from test_planning import refinement_data

FEED = (Path(__file__).parent / "fixtures/arxiv.xml").read_bytes()


class CliTests(unittest.TestCase):
    def test_research_command_runs_planning_search_and_selection(self):
        decisions = iter([
            {**refinement_data(), "search_queries": ["retrieval evidence"]},
            {"action": "finish", "query": "", "reason": "Found a relevant candidate", "selected_paper_ids": ["arxiv:2501.99999"]},
        ])

        def send(request, **kwargs):
            if request.full_url.startswith("http://localhost:11434/"):
                decision = next(decisions)
                return io.BytesIO(json.dumps({"message": {"content": json.dumps(decision)}}).encode())
            self.assertTrue(request.full_url.startswith("https://export.arxiv.org/api/query?"))
            return io.BytesIO(FEED)

        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
             patch("urllib.request.urlopen", side_effect=send) as requests, \
             patch("agentic_paper_retrieval.sources.arxiv._last_request", None), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["research", "Find retrieval methods", "--model", "test-model", "--auto", "--max-rounds", "1", "--json", "--cache-dir", directory])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(requests.call_count, 3)
        self.assertEqual(payload["stop_reason"], "round_limit")
        self.assertEqual(payload["selected_paper_ids"], ["arxiv:2501.99999"])
        self.assertEqual(payload["steps"][0]["executed_query"], "retrieval evidence")
        self.assertEqual(len(payload["papers"]), 2)
        self.assertEqual(payload["question"], "Find retrieval methods")
        self.assertEqual(payload["refinement"]["search_queries"], ["retrieval evidence"])
        self.assertEqual(payload["effective_question"], refinement_data()["refined_question"])

    def test_refine_previews_queries_without_accessing_paper_source(self):
        stdout = io.StringIO()
        with patch("agentic_paper_retrieval.__main__.ArxivSource") as source, \
             patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", return_value=io.BytesIO(json.dumps({"message": {"content": json.dumps(refinement_data())}}).encode())), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["refine", "How can RAG reduce hallucinations?", "--model", "local-model", "--json"]), 0)
        source.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["question"], "How can RAG reduce hallucinations?")
        self.assertEqual(len(payload["refinement"]["search_queries"]), 2)

    def test_interactive_question_edit_replans_before_search_and_keeps_json_clean(self):
        first = {**refinement_data(), "search_queries": ["old query"]}
        second = {**refinement_data(), "search_queries": ["evaluation evidence"]}
        replies = iter([
            first, second,
            {"action": "finish", "query": "", "reason": "Done", "selected_paper_ids": []},
        ])

        def send(request, **kwargs):
            if request.full_url.startswith("http://localhost:11434/"):
                return io.BytesIO(json.dumps({"message": {"content": json.dumps(next(replies))}}).encode())
            self.assertIn("evaluation", request.full_url)
            return io.BytesIO(FEED)

        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
             patch("urllib.request.urlopen", side_effect=send) as requests, \
             patch("agentic_paper_retrieval.sources.arxiv._last_request", None), \
             patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", side_effect=["Focus on hallucination evaluation benchmarks", ""]), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["research", "original question", "--model", "test-model", "--max-rounds", "1", "--json", "--cache-dir", directory])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(requests.call_count, 4)
        self.assertEqual(payload["question"], "original question")
        self.assertEqual(payload["effective_question"], "Focus on hallucination evaluation benchmarks")
        self.assertEqual(payload["initial_refinement"]["search_queries"], ["old query"])
        self.assertEqual(payload["steps"][0]["executed_query"], "evaluation evidence")
        self.assertIn("Refined question:", stderr.getvalue())
        self.assertIn("Proposed search:", stderr.getvalue())

    def test_search_json_and_export_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested/papers.json"
            stdout = io.StringIO()
            with patch("agentic_paper_retrieval.__main__.ArxivSource") as source, contextlib.redirect_stdout(stdout):
                source.return_value.search.return_value = parse_feed(FEED, "retrieval", "all:retrieval")
                self.assertEqual(main(["search", "retrieval", "--json", "--output", str(target)]), 0)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(target.read_text()))
            self.assertEqual(len(json.loads(stdout.getvalue())["papers"]), 2)

    def test_source_failure_sets_exit_code_and_keeps_stdout_clean(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("agentic_paper_retrieval.__main__.ArxivSource") as source, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            source.return_value.search.side_effect = SourceError("offline")
            self.assertEqual(main(["search", "retrieval", "--json"]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("offline", stderr.getvalue())

    def test_input_errors_do_not_call_source(self):
        with patch("agentic_paper_retrieval.__main__.ArxivSource") as source, contextlib.redirect_stderr(io.StringIO()):
            for argv in (["search", "q", "--limit", "0"], ["research", "q", "--model", "model", "--limit", "21", "--auto"]):
                with self.assertRaises(SystemExit) as exit_info:
                    main(argv)
                self.assertEqual(exit_info.exception.code, 2)
            source.assert_not_called()
