"""Query plans and search decisions."""

from dataclasses import asdict, dataclass
from typing import Protocol

from agentic_paper_retrieval.models import Paper


class PlannerError(RuntimeError):
    """A planner failed or proposed an invalid action."""


@dataclass(frozen=True)
class QueryRefinement:
    refined_question: str
    concepts: tuple[str, ...]
    search_queries: tuple[str, ...]
    assumptions: tuple[str, ...]
    clarification_questions: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


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
    library_query: str = ""
    library_paper_ids: tuple[str, ...] = ()
    library_error: str = ""
    source_error: str = ""


class Planner(Protocol):
    def refine(self, question: str) -> QueryRefinement: ...

    def decide(
        self, question: str, steps: tuple[SearchStep, ...], papers: tuple[Paper, ...],
        *, allow_search: bool, refinement: QueryRefinement,
    ) -> Decision: ...


REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_question": {"type": "string", "minLength": 1, "maxLength": 2000},
        "concepts": {
            "type": "array", "minItems": 1, "maxItems": 10,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "search_queries": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "assumptions": {
            "type": "array", "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "clarification_questions": {
            "type": "array", "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
    },
    "required": [
        "refined_question", "concepts", "search_queries", "assumptions",
        "clarification_questions", "rationale",
    ],
    "additionalProperties": False,
}


def parse_refinement(data: object) -> QueryRefinement:
    """Check a model's query plan before using it."""
    if not isinstance(data, dict) or set(data) != set(REFINEMENT_SCHEMA["required"]):
        raise PlannerError("The model returned an invalid query refinement object.")
    values = {}
    for key, schema in REFINEMENT_SCHEMA["properties"].items():
        value = data[key]
        if schema["type"] == "string":
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= schema["maxLength"]:
                raise PlannerError(f"The model returned an invalid {key}.")
            values[key] = value.strip()
            continue
        if not isinstance(value, (list, tuple)) or not schema.get("minItems", 0) <= len(value) <= schema["maxItems"]:
            raise PlannerError(f"The model returned an invalid {key} list.")
        if any(not isinstance(item, str) or not 1 <= len(item.strip()) <= schema["items"]["maxLength"] for item in value):
            raise PlannerError(f"The model returned invalid text in {key}.")
        unique: dict[str, str] = {}
        for item in value:
            normalized = " ".join(item.split())
            unique.setdefault(normalized.casefold(), normalized)
        values[key] = tuple(unique.values())
    return QueryRefinement(**values)
