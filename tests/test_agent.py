import unittest
import threading
from unittest.mock import Mock

from agentic_paper_retrieval.agent import ResearchAgent
from agentic_paper_retrieval.planning import Decision, PlannerError, QueryRefinement
from agentic_paper_retrieval.models import Paper, SearchResult
from agentic_paper_retrieval.library import LibraryError
from agentic_paper_retrieval.sources.base import SourceError


def paper(number):
    return Paper(f"arxiv:{number}", "arxiv", number, f"Synthetic paper {number}", (), "Abstract", "2025-01-01", f"https://arxiv.org/abs/{number}")


def result(*papers):
    return SearchResult("query", "all:query", "arxiv", len(papers), tuple(papers))


def plan(query="initial keywords", question="A clearer research question"):
    return QueryRefinement(question, ("retrieval",), (query,), (), (), "Find candidate methods.")


class AgentTests(unittest.TestCase):
    def test_parallel_retrievers_merge_before_evaluation_and_preserve_both_origins(self):
        source, library, planner = Mock(), Mock(), Mock()
        barrier = threading.Barrier(2)
        first, second = paper("saved"), paper("new")

        def remote_search(*args, **kwargs):
            barrier.wait(timeout=2)
            return result(first, second)

        def local_search(*args, **kwargs):
            barrier.wait(timeout=2)
            return result(first)

        source.search.side_effect, library.search.side_effect = remote_search, local_search
        planner.refine.return_value = plan()
        planner.decide.return_value = Decision("finish", "", "Both candidates are relevant", (first.id, second.id))
        research = ResearchAgent(source, planner, library).run("question", max_rounds=1)
        self.assertEqual(research.papers, (first, second))
        self.assertEqual(research.paper_origins[first.id], ["library", "arxiv"])
        self.assertEqual(research.paper_origins[second.id], ["arxiv"])
        self.assertEqual(research.steps[0].library_paper_ids, (first.id,))
        planner.decide.assert_called_once()
        library.save.assert_not_called()

    def test_one_failed_source_keeps_the_other_sources_results(self):
        for failing_source in ("library", "arxiv"):
            with self.subTest(failing_source=failing_source):
                source, library, planner = Mock(), Mock(), Mock()
                source.search.return_value = library.search.return_value = result(paper("one"))
                (library if failing_source == "library" else source).search.side_effect = LibraryError("source offline")
                planner.refine.return_value = plan()
                planner.decide.return_value = Decision("finish", "", "Partial coverage", ("arxiv:one",))
                research = ResearchAgent(source, planner, library).run("q", max_rounds=1)
                self.assertEqual(len(research.papers), 1)
                self.assertIn("source offline", research.steps[0].library_error + research.steps[0].source_error)

    def test_both_unavailable_sources_fail_without_claiming_empty_search_results(self):
        source, library, planner = Mock(), Mock(), Mock()
        source.search.side_effect = SourceError("arxiv offline")
        library.search.side_effect = LibraryError("library offline")
        planner.refine.return_value = plan()
        with self.assertRaisesRegex(SourceError, "arxiv offline.*library offline"):
            ResearchAgent(source, planner, library).run("q")
        planner.decide.assert_not_called()

    def test_refines_from_results_deduplicates_and_finishes(self):
        first, second = paper("one"), paper("two")
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan()
        source.search.side_effect = [result(first), result(first, second)]
        planner.decide.side_effect = [
            Decision("search", "evaluation evidence", "Find evaluations missing from the first results."),
            Decision("finish", "", "The second paper evaluates the method.", (second.id, first.id)),
        ]
        research = ResearchAgent(source, planner).run("A question")
        self.assertEqual(len(research.steps), 2)
        self.assertEqual(research.papers, (first, second))
        self.assertEqual(research.selected_paper_ids, (second.id, first.id))
        self.assertEqual(research.stop_reason, "model_finished")
        self.assertEqual(planner.decide.call_args_list[0].args[2], (first,))
        self.assertEqual(planner.decide.call_args_list[0].args[0], "A clearer research question")
        self.assertEqual(research.question, "A question")
        self.assertEqual(research.refinement.search_queries, ("initial keywords",))

    def test_user_can_edit_or_stop_before_any_network_request(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan("model query")
        research = ResearchAgent(source, planner).run("question", approve=lambda _: None)
        source.search.assert_not_called()
        planner.refine.assert_called_once_with("question")
        self.assertEqual(research.stop_reason, "user_stopped")
        source.search.return_value = result(paper("one"))
        planner.decide.return_value = Decision("finish", "", "Done")
        research = ResearchAgent(source, planner).run("question", approve=lambda _: "my edited query")
        source.search.assert_called_once_with("my edited query", limit=5)
        self.assertEqual(research.steps[0].proposed_query, "model query")
        self.assertEqual(research.steps[0].executed_query, "my edited query")

    def test_budget_forces_a_final_selection_without_another_search(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan("keywords")
        source.search.return_value = result(paper("one"))
        planner.decide.return_value = Decision("finish", "", "Budget reached", ("arxiv:one",))
        research = ResearchAgent(source, planner).run("question", max_rounds=1)
        self.assertEqual(research.stop_reason, "round_limit")
        source.search.assert_called_once()
        self.assertFalse(planner.decide.call_args.kwargs["allow_search"])

    def test_repeated_query_does_not_repeat_a_source_call(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan("Some Query")
        source.search.return_value = result()
        planner.decide.return_value = Decision("search", "some   query", "Again")
        research = ResearchAgent(source, planner).run("question")
        self.assertEqual(research.stop_reason, "repeated_query")
        source.search.assert_called_once()

    def test_invented_ids_and_actions_are_rejected(self):
        for final in (Decision("finish", "", "Done", ("invented",)), Decision("search", "again", "Reason"), Decision("shell", "rm", "Reason")):
            with self.subTest(final=final):
                source, planner = Mock(), Mock()
                planner.refine.return_value = plan("keywords")
                source.search.return_value = result(paper("one"))
                planner.decide.return_value = final
                with self.assertRaises(PlannerError):
                    ResearchAgent(source, planner).run("question", max_rounds=1)
                source.search.assert_called_once()

    def test_invalid_limits_and_empty_question_fail_before_planning(self):
        source, planner = Mock(), Mock()
        for question, rounds, limit in (("", 1, 5), ("q", 0, 5), ("q", 6, 5), ("q", 1, 21)):
            with self.assertRaises(ValueError):
                ResearchAgent(source, planner).run(question, max_rounds=rounds, limit=limit)
        planner.decide.assert_not_called()
        planner.refine.assert_not_called()

    def test_query_review_stop_prevents_retrieval(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan()
        research = ResearchAgent(source, planner).run("original", review_query=lambda _: None)
        self.assertEqual(research.query_review, "stopped")
        self.assertEqual(research.stop_reason, "user_stopped")
        planner.refine.assert_called_once_with("original")
        planner.decide.assert_not_called()
        source.search.assert_not_called()

    def test_question_edit_regenerates_keywords_and_preserves_original(self):
        source, planner = Mock(), Mock()
        original_plan, edited_plan = plan("old keywords"), plan("new evaluation terms", "A revised question")
        planner.refine.side_effect = [original_plan, edited_plan]
        planner.decide.return_value = Decision("finish", "", "Done", ("arxiv:one",))
        source.search.return_value = result(paper("one"))
        reviewer = Mock(return_value="Focus on evaluation benchmarks")
        approver = Mock(side_effect=lambda decision: decision.query)
        research = ResearchAgent(source, planner).run("original question", review_query=reviewer, approve=approver)
        self.assertEqual([item.args[0] for item in planner.refine.call_args_list], ["original question", "Focus on evaluation benchmarks"])
        source.search.assert_called_once_with("new evaluation terms", limit=5)
        self.assertEqual(research.question, "original question")
        self.assertEqual(research.effective_question, "Focus on evaluation benchmarks")
        self.assertEqual(research.initial_refinement, original_plan)
        self.assertEqual(research.refinement, edited_plan)
        self.assertEqual(research.query_review, "edited")
        reviewer.assert_called_once()
        approver.assert_called_once()

    def test_five_rounds_fit_graph_budget_and_approvals_do_not_replay_searches(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = plan("query zero")
        source.search.return_value = result(paper("one"))
        planner.decide.side_effect = [
            *(Decision("search", f"query {number}", "Find more evidence") for number in range(1, 5)),
            Decision("finish", "", "Budget reached", ("arxiv:one",)),
        ]
        research = ResearchAgent(source, planner).run(
            "question", max_rounds=5,
            review_query=lambda refinement: refinement.refined_question,
            approve=lambda decision: decision.query,
        )
        self.assertEqual(len(research.steps), 5)
        self.assertEqual(source.search.call_count, 5)
        self.assertEqual(planner.decide.call_count, 5)
        planner.refine.assert_called_once()
        self.assertEqual(research.stop_reason, "round_limit")

    def test_model_cannot_clear_open_clarifications_or_change_actual_stop_reason(self):
        source, planner = Mock(), Mock()
        planner.refine.return_value = QueryRefinement(
            "How does RAG help medical QA?", ("RAG",), ("medical RAG",), (),
            ("Which medical domain should be covered?",), "Expand the query.",
        )
        source.search.return_value = result(paper("one"))
        claim = "All clarifications were answered and coverage is comprehensive."
        planner.decide.return_value = Decision("finish", "", claim, ("arxiv:one",))
        research = ResearchAgent(source, planner).run("medical RAG", max_rounds=1)
        self.assertEqual(research.unresolved_clarifications, ("Which medical domain should be covered?",))
        self.assertIn("configured search limit", research.rationale)
        self.assertNotIn("clarifications were answered", research.rationale)
        self.assertEqual(research.model_assessment, claim)
