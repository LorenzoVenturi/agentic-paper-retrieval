import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from agentic_paper_retrieval.agent import PlannerError
from agentic_paper_retrieval.ollama import OllamaPlanner, parse_decision
from test_planning import refinement_data


def response(decision):
    return io.BytesIO(json.dumps({"message": {"content": json.dumps(decision)}}).encode())


class OllamaTests(unittest.TestCase):
    def test_refiner_uses_original_question_and_a_dedicated_schema(self):
        data = refinement_data()
        with patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", return_value=response(data)) as send:
            plan = OllamaPlanner("local-model").refine("How can RAG reduce hallucinations?")
            payload = json.loads(send.call_args.args[0].data)
            context = json.loads(payload["messages"][1]["content"])
            self.assertEqual(context["original_question"], "How can RAG reduce hallucinations?")
            self.assertIn("search_queries", payload["format"]["required"])
            self.assertEqual(plan.refined_question, data["refined_question"])

    def test_request_uses_structured_nonstreaming_local_chat(self):
        decision = {"action": "search", "query": "medical retrieval", "reason": "Find evidence", "selected_paper_ids": []}
        with patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", return_value=response(decision)) as send:
            result = OllamaPlanner("local-model").decide("question", (), (), allow_search=True)
            request = send.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["format"]["properties"]["action"]["enum"], ["search"])
            self.assertEqual(result.query, "medical retrieval")

    def test_invalid_model_decisions_are_rejected(self):
        valid = {"action": "search", "query": "q", "reason": "r", "selected_paper_ids": []}
        invalid = [None, [], {}, {**valid, "action": "execute"}, {**valid, "query": ""}, {**valid, "reason": None}, {**valid, "selected_paper_ids": "fake"}, {**valid, "selected_paper_ids": [1]}, {**valid, "extra": True}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(PlannerError):
                parse_decision(value)

    def test_model_cannot_ignore_budget_or_invent_paper_ids(self):
        for decision in (
            {"action": "search", "query": "q", "reason": "r", "selected_paper_ids": []},
            {"action": "finish", "query": "", "reason": "r", "selected_paper_ids": ["invented"]},
        ):
            with patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", return_value=response(decision)):
                with self.assertRaises(PlannerError):
                    OllamaPlanner("model").decide("q", (), (), allow_search=False)

    def test_unavailable_ollama_and_malformed_json_report_clear_errors(self):
        with patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(PlannerError, "Start Ollama"):
                OllamaPlanner("model").decide("q", (), (), allow_search=True)
        for raw in (b"bad JSON", b"[]", b'{"message":{"content":"bad JSON"}}'):
            with patch("agentic_paper_retrieval.ollama.urllib.request.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaisesRegex(PlannerError, "valid JSON"):
                    OllamaPlanner("model").decide("q", (), (), allow_search=True)
