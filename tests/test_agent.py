import unittest
from unittest.mock import Mock

from agentic_paper_retrieval.agent import Decision, PlannerError, ResearchAgent
from agentic_paper_retrieval.models import Paper, SearchResult


def paper(number):
    return Paper(f"arxiv:{number}", "arxiv", number, f"Synthetic paper {number}", (), "Abstract", "2025-01-01", f"https://arxiv.org/abs/{number}")


def result(*papers):
    return SearchResult("query", "all:query", "arxiv", len(papers), tuple(papers))


class AgentTests(unittest.TestCase):
    def test_refines_from_results_deduplicates_and_finishes(self):
        first, second = paper("one"), paper("two")
        source, planner = Mock(), Mock()
        source.search.side_effect = [result(first), result(first, second)]
        planner.decide.side_effect = [
            Decision("search", "initial keywords", "Find candidate methods."),
            Decision("search", "evaluation evidence", "Find evaluations missing from the first results."),
            Decision("finish", "", "The second paper evaluates the method.", (second.id, first.id)),
        ]
        research = ResearchAgent(source, planner).run("A question")
        self.assertEqual(len(research.steps), 2)
        self.assertEqual(research.papers, (first, second))
        self.assertEqual(research.selected_paper_ids, (second.id, first.id))
        self.assertEqual(research.stop_reason, "model_finished")
        # The next decision sees the actual papers from the previous search.
        self.assertEqual(planner.decide.call_args_list[1].args[2], (first,))

    def test_user_can_edit_or_stop_before_any_network_request(self):
        source, planner = Mock(), Mock()
        planner.decide.return_value = Decision("search", "model query", "Reason")
        research = ResearchAgent(source, planner).run("question", approve=lambda _: None)
        source.search.assert_not_called()
        self.assertEqual(research.stop_reason, "user_stopped")
        source.search.return_value = result(paper("one"))
        planner.decide.side_effect = [Decision("search", "model query", "Reason"), Decision("finish", "", "Done")]
        research = ResearchAgent(source, planner).run("question", approve=lambda _: "my edited query")
        source.search.assert_called_once_with("my edited query", limit=5)
        self.assertEqual(research.steps[0].proposed_query, "model query")
        self.assertEqual(research.steps[0].executed_query, "my edited query")

    def test_budget_forces_a_final_selection_without_another_search(self):
        source, planner = Mock(), Mock()
        source.search.return_value = result(paper("one"))
        planner.decide.side_effect = [Decision("search", "keywords", "Reason"), Decision("finish", "", "Budget reached", ("arxiv:one",))]
        research = ResearchAgent(source, planner).run("question", max_rounds=1)
        self.assertEqual(research.stop_reason, "round_limit")
        source.search.assert_called_once()
        self.assertFalse(planner.decide.call_args.kwargs["allow_search"])

    def test_repeated_query_does_not_repeat_a_source_call(self):
        source, planner = Mock(), Mock()
        source.search.return_value = result()
        planner.decide.side_effect = [Decision("search", "Some Query", "Reason"), Decision("search", "some   query", "Again")]
        research = ResearchAgent(source, planner).run("question")
        self.assertEqual(research.stop_reason, "repeated_query")
        source.search.assert_called_once()

    def test_invented_ids_and_actions_are_rejected(self):
        for final in (Decision("finish", "", "Done", ("invented",)), Decision("search", "again", "Reason"), Decision("shell", "rm", "Reason")):
            with self.subTest(final=final):
                source, planner = Mock(), Mock()
                source.search.return_value = result(paper("one"))
                planner.decide.side_effect = [Decision("search", "keywords", "Reason"), final]
                with self.assertRaises(PlannerError):
                    ResearchAgent(source, planner).run("question", max_rounds=1)
                source.search.assert_called_once()

    def test_invalid_limits_and_empty_question_fail_before_planning(self):
        source, planner = Mock(), Mock()
        for question, rounds, limit in (("", 1, 5), ("q", 0, 5), ("q", 6, 5), ("q", 1, 21)):
            with self.assertRaises(ValueError):
                ResearchAgent(source, planner).run(question, max_rounds=rounds, limit=limit)
        planner.decide.assert_not_called()
