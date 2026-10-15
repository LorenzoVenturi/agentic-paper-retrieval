import unittest
from dataclasses import asdict

from agentic_paper_retrieval.planning import PlannerError, parse_refinement


def refinement_data():
    return {
        "refined_question": "How does retrieval-augmented generation affect hallucinations?",
        "concepts": ["retrieval-augmented generation", "hallucinations"],
        "search_queries": ['"retrieval augmented generation" hallucinations', "RAG factuality"],
        "assumptions": [],
        "clarification_questions": [],
        "rationale": "Expand the abbreviation and search alternative terminology.",
    }


class PlanningTests(unittest.TestCase):
    def test_refinement_preserves_alternatives_and_round_trips(self):
        data = refinement_data()
        result = parse_refinement(data)
        self.assertEqual(result.search_queries, tuple(data["search_queries"]))
        self.assertEqual(parse_refinement(asdict(result)), result)

    def test_whitespace_and_case_duplicates_do_not_create_extra_queries(self):
        data = refinement_data()
        data["search_queries"] = ["RAG factuality", " RAG   factuality ", "rag factuality"]
        self.assertEqual(parse_refinement(data).search_queries, ("RAG factuality",))

    def test_invalid_refinements_fail_before_any_search(self):
        valid = refinement_data()
        invalid = [
            None, [], {}, {**valid, "extra": True},
            {**valid, "refined_question": " "}, {**valid, "refined_question": 3},
            {**valid, "concepts": []}, {**valid, "search_queries": []},
            {**valid, "search_queries": ["a", "b", "c", "d"]},
            {**valid, "search_queries": [" "]}, {**valid, "search_queries": [True]},
            {**valid, "assumptions": None}, {**valid, "clarification_questions": ["q"] * 4},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(PlannerError):
                parse_refinement(value)
