"""A bounded search loop with optional human approval at each search."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

from agentic_paper_retrieval.models import Paper
from agentic_paper_retrieval.sources.base import PaperSource


class PlannerError(RuntimeError):
    """A planner failed or proposed an invalid action."""


@dataclass(frozen=True)
class Decision:
    action: str
    query: str
    reason: str
    selected_paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchStep:
    proposed_query: str
    executed_query: str
    reason: str
    source_query: str
    total_results: int
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearchResult:
    question: str
    steps: tuple[SearchStep, ...]
    papers: tuple[Paper, ...]
    selected_paper_ids: tuple[str, ...]
    stop_reason: str
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class Planner(Protocol):
    def decide(
        self, question: str, steps: tuple[SearchStep, ...], papers: tuple[Paper, ...],
        *, allow_search: bool,
    ) -> Decision: ...


class ResearchAgent:
    def __init__(self, source: PaperSource, planner: Planner):
        self.source = source
        self.planner = planner

    def run(
        self, question: str, *, max_rounds: int = 3, limit: int = 5,
        approve: Callable[[Decision], str | None] | None = None,
        on_results: Callable[[SearchStep, tuple[Paper, ...]], None] | None = None,
    ) -> ResearchResult:
        question = question.strip()
        if not question:
            raise ValueError("The research question cannot be empty.")
        if not 1 <= max_rounds <= 5 or not 1 <= limit <= 20:
            raise ValueError("Use 1–5 search rounds and 1–20 results per round.")
        steps: list[SearchStep] = []
        catalog: dict[str, Paper] = {}
        seen_queries: set[str] = set()

        def finish(reason: str, rationale: str, selected: tuple[str, ...] = ()) -> ResearchResult:
            if any(paper_id not in catalog for paper_id in selected):
                raise PlannerError("The model selected a paper that was never retrieved.")
            return ResearchResult(question, tuple(steps), tuple(catalog.values()), tuple(dict.fromkeys(selected)), reason, rationale)

        # At most max_rounds searches and max_rounds + 1 model decisions.
        for turn in range(max_rounds + 1):
            allow_search = turn < max_rounds
            decision = self.planner.decide(question, tuple(steps), tuple(catalog.values()), allow_search=allow_search)
            if decision.action == "finish":
                if not steps:
                    raise PlannerError("The model tried to finish before searching for papers.")
                return finish("model_finished" if allow_search else "round_limit", decision.reason, decision.selected_paper_ids)
            if decision.action != "search" or not allow_search:
                raise PlannerError("The model proposed an action outside the search budget.")
            query = approve(decision) if approve else decision.query
            if query is None:
                return finish("user_stopped", "The researcher stopped the search.")
            query = " ".join(query.split())
            if not query:
                raise PlannerError("The proposed search query is empty.")
            key = query.casefold()
            if key in seen_queries:
                return finish("repeated_query", "Stopped because the proposed search was already executed.")
            result = self.source.search(query, limit=limit)
            seen_queries.add(key)
            for paper in result.papers:
                catalog.setdefault(paper.id, paper)
            step = SearchStep(decision.query, query, decision.reason, result.source_query, result.total_results, tuple(p.id for p in result.papers))
            steps.append(step)
            if on_results:
                on_results(step, result.papers)
        raise AssertionError("The bounded loop must finish or raise an error.")
