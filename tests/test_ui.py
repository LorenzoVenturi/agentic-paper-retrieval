import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

from agentic_paper_retrieval.planning import Decision, QueryRefinement
from agentic_paper_retrieval.sources.arxiv import parse_feed
from agentic_paper_retrieval.ui import UIServer, UIRuntime

FEED = (Path(__file__).parent / "fixtures/arxiv.xml").read_bytes()


class UITests(unittest.TestCase):
    def setUp(self):
        self.planner, self.source = Mock(), Mock()

        def refine(question):
            query = "benchmark evidence" if "benchmark" in question.lower() else "retrieval evidence"
            return QueryRefinement(question + " (clarified)", ("retrieval",), (query,), (), (), "Find evidence.")

        self.planner.refine.side_effect = refine
        self.planner.decide.return_value = Decision("finish", "", "Relevant candidates", ("arxiv:2501.99999",))
        self.source.search.return_value = parse_feed(FEED, "retrieval", "all:retrieval")
        self.runtime = UIRuntime(planner_factory=lambda _: self.planner, source_factory=lambda: self.source)
        self.runtime.configuration = lambda: {"models": ["test-model"], "default_model": "test-model", "ollama_available": True}
        self.server = UIServer(("127.0.0.1", 0), self.runtime)
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.runtime.close()
        self.thread.join(timeout=3)

    def request(self, path, data=None, headers=None):
        request_headers = {"Origin": self.origin, **(headers or {})}
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.origin + path, data=body, headers=request_headers)
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            content = json.loads(raw) if response.headers.get_content_type() == "application/json" else raw
            return response.status, content, response.headers

    def create(self):
        status, data, _ = self.request("/api/sessions", {
            "question": "Original research question", "model": "test-model", "max_rounds": 1, "limit": 3,
        })
        self.assertEqual(status, 202)
        return data["id"]

    def wait_for(self, session_id, status):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            _, snapshot, _ = self.request(f"/api/sessions/{session_id}")
            if snapshot["status"] == status:
                return snapshot
            if snapshot["status"] == "error":
                self.fail(snapshot["error"])
            threading.Event().wait(0.01)
        self.fail(f"Session did not reach {status}: {snapshot}")

    def test_assets_and_configuration_are_served_without_exposing_project_files(self):
        for path, expected in (("/", b"Paper Retriever"), ("/app.js", b"sendReview"), ("/styles.css", b"--paper")):
            status, body, headers = self.request(path)
            self.assertEqual(status, 200)
            self.assertIn(expected, body)
            self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(self.request("/api/config")[1]["models"], ["test-model"])
        for path in ("/.env", "/../pyproject.toml", "/src/agentic_paper_retrieval/ui.py"):
            self.assertEqual(self.request(path)[0], 404)

    def test_question_edit_search_approval_and_export_use_real_graph_state(self):
        session_id = self.create()
        first = self.wait_for(session_id, "awaiting_query")
        self.source.search.assert_not_called()
        first_review = first["pending"]["id"]
        self.assertEqual(self.request(f"/api/sessions/{session_id}/respond", {
            "pending_id": first_review, "query": "Focus on evaluation benchmarks",
        })[0], 202)
        search = self.wait_for(session_id, "awaiting_search")
        self.assertEqual(search["refinement"]["search_queries"], ["benchmark evidence"])
        self.source.search.assert_not_called()
        self.assertEqual(self.request(f"/api/sessions/{session_id}/respond", {
            "pending_id": first_review, "query": "stale approval",
        })[0], 409)
        self.assertEqual(self.request(f"/api/sessions/{session_id}/respond", {
            "pending_id": search["pending"]["id"], "query": "my approved keywords",
        })[0], 202)
        final = self.wait_for(session_id, "complete")
        self.source.search.assert_called_once_with("my approved keywords", limit=3)
        self.assertEqual(self.planner.refine.call_count, 2)
        self.assertEqual(final["result"]["question"], "Original research question")
        self.assertEqual(final["result"]["effective_question"], "Focus on evaluation benchmarks")
        self.assertEqual(final["result"]["steps"][0]["executed_query"], "my approved keywords")
        self.assertEqual(len(final["result"]["papers"]), 2)
        code, export, headers = self.request(f"/api/sessions/{session_id}/export")
        self.assertEqual(code, 200)
        self.assertEqual(export, final["result"])
        self.assertIn("attachment;", headers["Content-Disposition"])

    def test_stop_at_review_does_not_search(self):
        session_id = self.create()
        self.wait_for(session_id, "awaiting_query")
        self.assertEqual(self.request(f"/api/sessions/{session_id}/cancel", {})[0], 202)
        final = self.wait_for(session_id, "stopped")
        self.assertEqual(final["result"]["stop_reason"], "user_stopped")
        self.source.search.assert_not_called()
        self.planner.decide.assert_not_called()

    def test_stop_during_model_inference_prevents_later_search(self):
        started, release = threading.Event(), threading.Event()
        original_refine = self.planner.refine.side_effect

        def blocked_refine(question):
            started.set()
            release.wait(timeout=2)
            return original_refine(question)

        self.planner.refine.side_effect = blocked_refine
        try:
            session_id = self.create()
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(self.request(f"/api/sessions/{session_id}/cancel", {})[0], 202)
        finally:
            release.set()
        final = self.wait_for(session_id, "stopped")
        self.assertEqual(final["result"]["stop_reason"], "user_stopped")
        self.source.search.assert_not_called()

    def test_bad_input_and_cross_origin_actions_are_rejected(self):
        self.assertEqual(self.request("/api/sessions", {"question": "", "model": "test"})[0], 400)
        self.assertEqual(self.request("/api/sessions", {"question": "q", "model": "test", "max_rounds": True})[0], 400)
        self.assertEqual(self.request("/api/sessions", {"question": "q", "model": "test", "limit": 21})[0], 400)
        self.assertEqual(self.request("/api/sessions", {}, {"Origin": "https://another-site.example"})[0], 403)
        self.assertEqual(self.request("/", headers={"Host": "another-site.example"})[0], 403)
        self.planner.refine.assert_not_called()
        self.source.search.assert_not_called()

    def test_provider_failure_is_visible_to_browser(self):
        self.planner.refine.side_effect = RuntimeError("Local model unavailable")
        with self.assertLogs("agentic_paper_retrieval.ui", level="ERROR"):
            session_id = self.create()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                _, final, _ = self.request(f"/api/sessions/{session_id}")
                if final["status"] == "error":
                    break
                threading.Event().wait(0.01)
        self.assertEqual(final["status"], "error")
        self.assertEqual(final["error"], "Local model unavailable")
        self.source.search.assert_not_called()
