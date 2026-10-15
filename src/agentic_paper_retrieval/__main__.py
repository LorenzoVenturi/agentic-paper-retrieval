"""Search papers directly or run an assisted Ollama search agent."""

import argparse
import json
import os
import sys
from pathlib import Path

from agentic_paper_retrieval import __version__
from agentic_paper_retrieval.agent import Decision, PlannerError, ResearchAgent, SearchStep
from agentic_paper_retrieval.models import Paper
from agentic_paper_retrieval.ollama import OllamaPlanner
from agentic_paper_retrieval.sources.arxiv import ArxivSource
from agentic_paper_retrieval.sources.base import SourceError


def _positive_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an integer from 1 to 50.") from exc
    if not 1 <= number <= 50:
        raise argparse.ArgumentTypeError("Use an integer from 1 to 50.")
    return number


def _approve(decision: Decision) -> str | None:
    print(f"\nProposed search: {decision.query}\nReason: {decision.reason}", file=sys.stderr)
    print("Enter to approve, type a replacement query, or /stop: ", end="", file=sys.stderr, flush=True)
    try:
        answer = input().strip()
    except EOFError:
        return None
    return None if answer == "/stop" else (answer or decision.query)


def _progress(step: SearchStep, papers: tuple[Paper, ...]) -> None:
    print(f"\nSearch: {step.executed_query}\nReturned {len(papers)} papers ({step.total_results} matches).", file=sys.stderr)
    for paper in papers[:5]:
        print(f"  {paper.title}", file=sys.stderr)


def _show_papers(papers: list[dict], selected: list[str] | tuple[str, ...] = ()) -> None:
    by_id = {paper["id"]: paper for paper in papers}
    ordered = [by_id[paper_id] for paper_id in selected] if selected else papers
    if not ordered:
        print("No papers found. Try fewer terms or a broader query.")
    for index, paper in enumerate(ordered, 1):
        print(f"\n{index}. {paper['title']}")
        print(f"   {', '.join(paper['authors'])} | {paper['published'][:10]}")
        print(f"   {paper['url']}")
        print(f"   {paper['abstract']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paper-retriever",
        description="Search arXiv papers directly or with a local Ollama research agent.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    search = commands.add_parser("search", help="Search arXiv without a language model")
    research = commands.add_parser("research", help="Plan and refine searches using local Ollama")
    for command in (search, research):
        command.add_argument("query", help="Research question or search keywords")
        command.add_argument("--limit", type=_positive_limit, default=5, help="Papers per search (default: 5)")
        command.add_argument("--json", action="store_true", help="Print JSON to stdout")
        command.add_argument("--output", type=Path, help="Also save JSON to this file")
        command.add_argument("--cache-dir", type=Path, default=Path("data/cache/arxiv"))
    search.add_argument("--raw-query", action="store_true", help="Use arXiv field/Boolean syntax verbatim")
    research.add_argument("--model", default=os.getenv("OLLAMA_MODEL"), help="Installed Ollama model (or OLLAMA_MODEL)")
    research.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    research.add_argument("--max-rounds", type=int, choices=range(1, 6), default=3)
    research.add_argument("--auto", action="store_true", help="Allow searches within the round limit without human prompts")
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "research":
        if not args.model:
            parser.error("Choose an installed Ollama model with --model or OLLAMA_MODEL.")
        if args.limit > 20:
            parser.error("Research mode supports at most 20 papers per round.")
        if not args.auto and not sys.stdin.isatty():
            parser.error("Interactive research needs a terminal. Use --auto for bounded searches without prompts.")
    try:
        source = ArxivSource(cache_dir=args.cache_dir)
        if args.command == "search":
            payload = source.search(args.query, args.limit, raw_query=args.raw_query).to_dict()
        else:
            planner = OllamaPlanner(args.model, args.ollama_url)
            print(f"Planning with {args.model}; up to {args.max_rounds} searches.", file=sys.stderr)
            payload = ResearchAgent(source, planner).run(
                args.query, max_rounds=args.max_rounds, limit=args.limit,
                approve=None if args.auto else _approve, on_results=_progress,
            ).to_dict()
            payload["model"] = args.model
            payload["ollama_url"] = args.ollama_url
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        if args.json:
            print(serialized, end="")
        else:
            if args.command == "research":
                print(f"Stopped: {payload['stop_reason']}\n{payload['rationale']}")
                print("\nSelected papers:" if payload["selected_paper_ids"] else "\nRetrieved candidates (no shortlist selected):")
            else:
                print(f"arXiv: {len(payload['papers'])} papers returned ({payload['total_results']} matches).")
            _show_papers(payload["papers"], payload.get("selected_paper_ids", ()))
            if args.output:
                print(f"\nSaved JSON to {args.output}")
        return 0
    except (SourceError, PlannerError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by researcher.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
