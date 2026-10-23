"""LangGraph research workflow."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentic_paper_retrieval.models import Paper
from agentic_paper_retrieval.library import PaperLibrary
from agentic_paper_retrieval.planning import (
    Decision, Planner, PlannerError, QueryRefinement, SearchStep, parse_refinement,
)
from agentic_paper_retrieval.sources.base import PaperSource, SourceError


@dataclass(frozen=True)
class ResearchResult:
    question: str
    initial_refinement: QueryRefinement
    refinement: QueryRefinement
    effective_question: str
    query_review: str
    steps: tuple[SearchStep, ...]
    papers: tuple[Paper, ...]
    selected_paper_ids: tuple[str, ...]
    stop_reason: str
    rationale: str
    model_assessment: str
    unresolved_clarifications: tuple[str, ...]
    paper_origins: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchState(TypedDict, total=False):
    question: str
    initial_refinement: dict
    refinement: dict
    effective_question: str
    query_review: str
    review_query: bool
    approve_searches: bool
    max_rounds: int
    limit: int
    decision: dict
    executed_query: str
    steps: list[dict]
    papers: list[dict]
    seen_queries: list[str]
    selected_paper_ids: list[str]
    stop_reason: str
    rationale: str
    model_assessment: str
    result: dict
    arxiv_result: dict
    library_result: dict
    paper_origins: dict[str, list[str]]


def _papers(records: list[dict]) -> tuple[Paper, ...]:
    return tuple(Paper(**{
        **record, "authors": tuple(record["authors"]),
        "categories": tuple(record.get("categories", ())),
    }) for record in records)


def _steps(records: list[dict]) -> tuple[SearchStep, ...]:
    return tuple(SearchStep(**{
        **record, "paper_ids": tuple(record["paper_ids"]),
        "library_paper_ids": tuple(record.get("library_paper_ids", ())),
    }) for record in records)


def _decision(record: dict) -> Decision:
    return Decision(**{**record, "selected_paper_ids": tuple(record["selected_paper_ids"])})


def _result(record: dict) -> ResearchResult:
    return ResearchResult(
        question=record["question"],
        initial_refinement=parse_refinement(record["initial_refinement"]),
        refinement=parse_refinement(record["refinement"]),
        effective_question=record["effective_question"], query_review=record["query_review"],
        steps=_steps(record["steps"]), papers=_papers(record["papers"]),
        selected_paper_ids=tuple(record["selected_paper_ids"]),
        stop_reason=record["stop_reason"], rationale=record["rationale"],
        model_assessment=record["model_assessment"],
        unresolved_clarifications=parse_refinement(record["refinement"]).clarification_questions,
        paper_origins=record.get("paper_origins", {}),
    )


class ResearchAgent:
    def __init__(self, source: PaperSource, planner: Planner, library: PaperLibrary | None = None):
        self.source = source
        self.planner = planner
        self.library = library

    @staticmethod
    def initial_state(
        question: str, *, max_rounds: int = 3, limit: int = 5,
        review_query: bool = True, approve_searches: bool = True,
    ) -> ResearchState:
        """Validate settings and create the initial research state."""
        question = question.strip()
        if not question:
            raise ValueError("The research question cannot be empty.")
        if not 1 <= max_rounds <= 5 or not 1 <= limit <= 20:
            raise ValueError("Use 1–5 search rounds and 1–20 results per round.")
        return {
            "question": question, "query_review": "auto",
            "review_query": review_query, "approve_searches": approve_searches,
            "max_rounds": max_rounds, "limit": limit, "steps": [], "papers": [],
            "seen_queries": [], "selected_paper_ids": [], "stop_reason": "",
            "rationale": "", "model_assessment": "",
            "paper_origins": {},
        }

    def build_graph(
        self, *, on_results: Callable[[SearchStep, tuple[Paper, ...]], None] | None = None,
    ):
        """Build the graph with an in-memory checkpointer."""
        def refine_query(state: ResearchState) -> dict:
            plan = self.planner.refine(state["question"])
            return {
                "initial_refinement": plan.to_dict(), "refinement": plan.to_dict(),
                "effective_question": plan.refined_question,
            }

        def review_query(state: ResearchState) -> dict:
            if not state["review_query"]:
                return {"query_review": "auto"}
            # Interrupt nodes replay on resume, so model calls belong elsewhere.
            response = interrupt({
                "kind": "query_review", "question": state["question"],
                "refinement": state["refinement"],
            })
            question = response["query"]
            if question is None:
                return {
                    "query_review": "stopped", "stop_reason": "user_stopped",
                    "rationale": "The researcher stopped during query review.",
                }
            if not isinstance(question, str) or not question.strip():
                raise PlannerError("The approved research question cannot be empty.")
            question = question.strip()
            return {
                "effective_question": question,
                "query_review": "approved" if question == state["refinement"]["refined_question"] else "edited",
            }

        def replan_query(state: ResearchState) -> dict:
            return {"refinement": self.planner.refine(state["effective_question"]).to_dict()}

        def prepare_search(state: ResearchState) -> dict:
            plan = parse_refinement(state["refinement"])
            return {"decision": asdict(Decision("search", plan.search_queries[0], plan.rationale))}

        def approve_search(state: ResearchState) -> dict:
            decision = _decision(state["decision"])
            if state["approve_searches"]:
                response = interrupt({"kind": "search_approval", "decision": asdict(decision)})
                query = response["query"]
            else:
                query = decision.query
            if query is None:
                return {"stop_reason": "user_stopped", "rationale": "The researcher stopped the search."}
            if not isinstance(query, str) or not query.strip():
                raise PlannerError("The approved search query cannot be empty.")
            query = " ".join(query.split())
            if query.casefold() in state["seen_queries"]:
                return {
                    "stop_reason": "repeated_query",
                    "rationale": "Stopped because the proposed search was already executed.",
                }
            return {"executed_query": query}

        def search_arxiv(state: ResearchState) -> dict:
            if len(state["steps"]) >= state["max_rounds"]:
                raise PlannerError("The search budget has been exhausted.")
            try:
                result = self.source.search(state["executed_query"], limit=state["limit"])
                return {"arxiv_result": {**result.to_dict(), "error": ""}}
            except SourceError as exc:
                return {"arxiv_result": {"papers": [], "source_query": "", "total_results": 0, "error": str(exc)}}

        def search_library(state: ResearchState) -> dict:
            # The question carries intent; approved search terms narrow later rounds.
            query = state["effective_question"] + "\nSearch focus: " + state["executed_query"]
            try:
                result = self.library.search(query, limit=state["limit"]) if self.library else None
                return {"library_result": {
                    "papers": [paper.to_dict() for paper in result.papers] if result else [],
                    "source_query": query if result else "", "error": "",
                }}
            except SourceError as exc:
                return {"library_result": {"papers": [], "source_query": query, "error": str(exc)}}

        def merge_results(state: ResearchState) -> dict:
            query, decision = state["executed_query"], _decision(state["decision"])
            remote, local = state["arxiv_result"], state["library_result"]
            if remote["error"] and not local["papers"] and not state["papers"]:
                raise SourceError(remote["error"] + ("; " + local["error"] if local["error"] else ""))
            catalog = {paper.id: paper for paper in _papers(state["papers"])}
            origins = {key: list(value) for key, value in state["paper_origins"].items()}
            found = {}
            for origin, records in (("library", local["papers"]), ("arxiv", remote["papers"])):
                for paper in _papers(records):
                    catalog[paper.id] = found[paper.id] = paper
                    sources = origins.setdefault(paper.id, [])
                    if origin not in sources:
                        sources.append(origin)
            step = SearchStep(
                decision.query, query, decision.reason, remote["source_query"],
                remote["total_results"], tuple(paper["id"] for paper in remote["papers"]),
                local["source_query"], tuple(paper["id"] for paper in local["papers"]),
                local["error"], remote["error"],
            )
            if on_results:
                on_results(step, tuple(found.values()))
            return {
                "steps": [*state["steps"], asdict(step)],
                "papers": [paper.to_dict() for paper in catalog.values()],
                "seen_queries": [*state["seen_queries"], query.casefold()],
                "paper_origins": origins,
            }

        def evaluate(state: ResearchState) -> dict:
            allow_search = len(state["steps"]) < state["max_rounds"]
            decision = self.planner.decide(
                state["effective_question"], _steps(state["steps"]), _papers(state["papers"]),
                allow_search=allow_search, refinement=parse_refinement(state["refinement"]),
            )
            if decision.action == "finish":
                count = len(state["steps"])
                noun = "search" if count == 1 else "searches"
                known_ids = {paper["id"] for paper in state["papers"]}
                if any(paper_id not in known_ids for paper_id in decision.selected_paper_ids):
                    raise PlannerError("The model selected a paper that was never retrieved.")
                return {
                    "decision": asdict(decision), "model_assessment": decision.reason,
                    "rationale": (
                        f"The model chose to finish after {count} {noun}."
                        if allow_search else
                        f"Stopped after {count} {noun} because the configured search limit was reached."
                    ),
                    "selected_paper_ids": list(dict.fromkeys(decision.selected_paper_ids)),
                    "stop_reason": "model_finished" if allow_search else "round_limit",
                }
            if decision.action != "search" or not allow_search:
                raise PlannerError("The model proposed an action outside the search budget.")
            return {"decision": asdict(decision), "model_assessment": decision.reason}

        def finish(state: ResearchState) -> dict:
            return {"result": _result(state).to_dict()}

        def after_review(state: ResearchState) -> str:
            if state["stop_reason"]:
                return "finish"
            return "replan_query" if state["query_review"] == "edited" else "prepare_search"

        def after_decision(state: ResearchState) -> str:
            return "finish" if state["stop_reason"] else "approve_search"

        builder = StateGraph(ResearchState)
        for name, node in (
            ("refine_query", refine_query), ("review_query", review_query),
            ("replan_query", replan_query), ("prepare_search", prepare_search),
            ("approve_search", approve_search), ("search_arxiv", search_arxiv),
            ("search_library", search_library), ("merge_results", merge_results),
            ("evaluate", evaluate), ("finish", finish),
        ):
            builder.add_node(name, node)


            
        builder.add_edge(START, "refine_query")
        builder.add_edge("refine_query", "review_query")
        builder.add_conditional_edges("review_query", after_review, ["finish", "replan_query", "prepare_search"])
        builder.add_edge("replan_query", "prepare_search")
        builder.add_edge("prepare_search", "approve_search")
        builder.add_conditional_edges(
            "approve_search", lambda state: ["finish"] if state["stop_reason"] else ["search_arxiv", "search_library"],
            ["finish", "search_arxiv", "search_library"],
        )
        builder.add_edge(["search_arxiv", "search_library"], "merge_results")
        builder.add_edge("merge_results", "evaluate")
        builder.add_conditional_edges("evaluate", after_decision, ["finish", "approve_search"])
        builder.add_edge("finish", END)
        return builder.compile(checkpointer=InMemorySaver())

    def run(
        self, question: str, *, max_rounds: int = 3, limit: int = 5,
        approve: Callable[[Decision], str | None] | None = None,
        review_query: Callable[[QueryRefinement], str | None] | None = None,
        on_results: Callable[[SearchStep, tuple[Paper, ...]], None] | None = None,
    ) -> ResearchResult:
        initial = self.initial_state(
            question, max_rounds=max_rounds, limit=limit,
            review_query=review_query is not None, approve_searches=approve is not None,
        )
        graph = self.build_graph(on_results=on_results)
        config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 8 * max_rounds + 15}
        state = graph.invoke(initial, config=config)
        # Resume human reviews; search iteration stays inside the graph.
        while state.get("__interrupt__"):
            event = state["__interrupt__"][0].value
            if event["kind"] == "query_review" and review_query:
                response = review_query(parse_refinement(state["refinement"]))
            elif event["kind"] == "search_approval" and approve:
                response = approve(_decision(state["decision"]))
            else:
                raise PlannerError("The graph requested an unsupported approval.")
            state = graph.invoke(Command(resume={"query": response}), config=config)
        return _result(state["result"])

