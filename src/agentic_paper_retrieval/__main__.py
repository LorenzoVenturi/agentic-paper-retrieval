"""Command-line entry points."""

import argparse
import json
import os
import sys
from pathlib import Path

from agentic_paper_retrieval import __version__
from agentic_paper_retrieval.agent import ResearchAgent
from agentic_paper_retrieval.models import Paper
from agentic_paper_retrieval.library import ChromaPaperLibrary, LibraryError, OllamaEmbedder
from agentic_paper_retrieval.ollama import OllamaPlanner
from agentic_paper_retrieval.planning import Decision, PlannerError, QueryRefinement, SearchStep
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


def _show_refinement(plan: dict, *, file=None) -> None:
    print(f"\nRefined question: {plan['refined_question']}", file=file)
    print("Concepts: " + "; ".join(plan["concepts"]), file=file)
    print("Suggested searches:", file=file)
    for query in plan["search_queries"]:
        print(f"  - {query}", file=file)
    for label, key in (("Assumptions to review", "assumptions"), ("Clarifications to consider", "clarification_questions")):
        if plan[key]:
            print(label + ":", file=file)
            for item in plan[key]:
                print(f"  - {item}", file=file)
    print(f"Reason: {plan['rationale']}", file=file)


def _review_query(plan: QueryRefinement) -> str | None:
    _show_refinement(plan.to_dict(), file=sys.stderr)
    print(
        "Enter to approve, type an updated research question with any clarification answers, or /stop: ",
        end="", file=sys.stderr, flush=True,
    )
    try:
        answer = input().strip()
    except EOFError:
        return None
    return None if answer == "/stop" else (answer or plan.refined_question)


def _progress(step: SearchStep, papers: tuple[Paper, ...]) -> None:
    print(f"\nSearch: {step.executed_query}\nReturned {len(papers)} papers ({step.total_results} matches).", file=sys.stderr)
    for paper in papers[:5]:
        print(f"  {paper.title}", file=sys.stderr)
    if step.library_paper_ids:
        print(f"  {len(step.library_paper_ids)} candidates came from your saved library.", file=sys.stderr)
    for error in (step.library_error, step.source_error):
        if error:
            print(f"  Source unavailable: {error}", file=sys.stderr)


def _save_research(payload, library, automatic=False):
    papers = payload["papers"]
    selected = set(payload["selected_paper_ids"])
    ids = list(payload["selected_paper_ids"])
    if not automatic:
        print("\nSave papers for future questions?", file=sys.stderr)
        for index, paper in enumerate(papers, 1):
            suggested = " [suggested]" if paper["id"] in selected else ""
            print(f"  {index}. {paper['title']}{suggested}", file=sys.stderr)
        print("Type y to save the shortlist, paper numbers such as 1,3, or Enter to skip: ", end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip()
        except EOFError:
            answer = ""
        if answer.casefold() == "y":
            pass
        elif answer in ("", "/skip", "/stop"):
            ids = []
        else:
            try:
                numbers = [int(value.strip()) for value in answer.split(",")]
                if any(not 1 <= value <= len(papers) for value in numbers):
                    raise ValueError
                ids = [papers[value - 1]["id"] for value in numbers]
            except ValueError:
                print("Invalid paper numbers; no papers were saved.", file=sys.stderr)
                ids = []
    if not ids:
        return {"status": "skipped", "saved_paper_ids": []}
    known = {paper["id"]: paper for paper in papers}
    chosen = tuple(Paper(**{**known[key], "authors": tuple(known[key]["authors"]),
                           "categories": tuple(known[key].get("categories", ()))}) for key in dict.fromkeys(ids))
    try:
        return {"status": "saved", **library.save(chosen, payload["effective_question"])}
    except LibraryError as exc:
        print(f"Research completed, but saving failed: {exc}", file=sys.stderr)
        return {"status": "error", "error": str(exc), "saved_paper_ids": []}


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
    refine = commands.add_parser("refine", help="Improve a research question and preview keyword searches")
    ui = commands.add_parser("ui", help="Open the local browser research interface")
    library_command = commands.add_parser("library", help="Browse, search, or remove saved papers")
    library_actions = library_command.add_subparsers(dest="library_action", required=True)
    library_list = library_actions.add_parser("list", help="List saved papers without running a model")
    library_search = library_actions.add_parser("search", help="Search the library semantically")
    library_search.add_argument("query")
    library_remove = library_actions.add_parser("remove", help="Remove a saved paper by its stable ID")
    library_remove.add_argument("paper_id")
    library_list.add_argument("--offset", type=int, default=0)
    for command in (ui, research, library_list, library_search, library_remove):
        command.add_argument("--library-dir", type=Path, default=Path("data/library"))
        command.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma"))
    for command in (library_list, library_search, library_remove):
        command.add_argument("--json", action="store_true")
        command.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    for command in (library_list, library_search):
        command.add_argument("--limit", type=_positive_limit, default=10)
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ui.add_argument("--cache-dir", type=Path, default=Path("data/cache/arxiv"))
    for command in (search, research, refine):
        command.add_argument("query", help="Research question or search keywords")
        command.add_argument("--json", action="store_true", help="Print JSON to stdout")
        command.add_argument("--output", type=Path, help="Also save JSON to this file")
    for command in (search, research):
        command.add_argument("--limit", type=_positive_limit, default=5, help="Papers per search (default: 5)")
        command.add_argument("--cache-dir", type=Path, default=Path("data/cache/arxiv"))
    search.add_argument("--raw-query", action="store_true", help="Use arXiv field/Boolean syntax verbatim")
    for command in (research, refine):
        command.add_argument("--model", default=os.getenv("OLLAMA_MODEL"), help="Installed Ollama model (or OLLAMA_MODEL)")
        command.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    research.add_argument("--max-rounds", type=int, choices=range(1, 6), default=3)
    research.add_argument("--auto", action="store_true", help="Allow searches within the round limit without human prompts")
    research.add_argument("--no-library", action="store_true", help="Search arXiv only")
    research.add_argument("--save-shortlist", action="store_true", help="Explicitly save the final shortlist, including in --auto mode")
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "ui":
        if not 1 <= args.port <= 65535:
            parser.error("Choose a port between 1 and 65535.")
        from agentic_paper_retrieval.ui import serve
        try:
            serve(args.port, args.ollama_url, args.cache_dir, args.library_dir, args.embedding_model)
        except OSError as exc:
            print(f"Could not start the local UI: {exc}. Try --port 8001.", file=sys.stderr)
            return 1
        return 0
    if args.command == "library":
        try:
            library = ChromaPaperLibrary(args.library_dir, OllamaEmbedder(args.embedding_model, args.ollama_url))
            if args.library_action == "list":
                payload = library.list_papers(args.limit, args.offset)
            elif args.library_action == "search":
                if args.limit > 20:
                    parser.error("Library search supports at most 20 candidates.")
                payload = library.search(args.query, args.limit).to_dict()
            else:
                library.remove(args.paper_id)
                payload = {"removed": args.paper_id}
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            elif args.library_action == "remove":
                print(f"Removed {args.paper_id} from your saved library.")
            else:
                _show_papers(payload["papers"])
            return 0
        except (LibraryError, ValueError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if args.command in ("research", "refine"):
        if not args.model:
            parser.error("Choose an installed Ollama model with --model or OLLAMA_MODEL.")
    if args.command == "research":
        if args.no_library and args.save_shortlist:
            parser.error("--save-shortlist requires the saved library; remove --no-library.")
        if args.limit > 20:
            parser.error("Research mode supports at most 20 papers per round.")
        if not args.auto and not sys.stdin.isatty():
            parser.error("Interactive research needs a terminal. Use --auto for bounded searches without prompts.")
    try:
        if args.command == "search":
            source = ArxivSource(cache_dir=args.cache_dir)
            payload = source.search(args.query, args.limit, raw_query=args.raw_query).to_dict()
        elif args.command == "refine":
            planner = OllamaPlanner(args.model, args.ollama_url)
            payload = {"question": args.query, "refinement": planner.refine(args.query).to_dict()}
        else:
            source = ArxivSource(cache_dir=args.cache_dir)
            planner = OllamaPlanner(args.model, args.ollama_url)
            library = None if args.no_library else ChromaPaperLibrary(
                args.library_dir, OllamaEmbedder(args.embedding_model, args.ollama_url),
            )
            print(f"Planning with {args.model}; up to {args.max_rounds} searches.", file=sys.stderr)
            payload = ResearchAgent(source, planner, library).run(
                args.query, max_rounds=args.max_rounds, limit=args.limit,
                approve=None if args.auto else _approve, on_results=_progress,
                review_query=None if args.auto else _review_query,
            ).to_dict()
            payload["model"] = args.model
            payload["ollama_url"] = args.ollama_url
            if library and payload["papers"] and (args.save_shortlist or not args.auto):
                payload["library_save"] = _save_research(payload, library, args.save_shortlist)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        if args.json:
            print(serialized, end="")
        else:
            if args.command == "refine":
                _show_refinement(payload["refinement"])
            elif args.command == "research":
                print(f"Research question: {payload['effective_question']}")
                print(f"Stopped: {payload['stop_reason']}\n{payload['rationale']}")
                if payload["unresolved_clarifications"]:
                    print("\nOpen clarifications:")
                    for question in payload["unresolved_clarifications"]:
                        print(f"  - {question}")
                print("\nSelected papers:" if payload["selected_paper_ids"] else "\nRetrieved candidates (no shortlist selected):")
            else:
                print(f"arXiv: {len(payload['papers'])} papers returned ({payload['total_results']} matches).")
            if args.command != "refine":
                _show_papers(payload["papers"], payload.get("selected_paper_ids", ()))
            if args.output:
                print(f"\nSaved JSON to {args.output}")
        return 1 if payload.get("library_save", {}).get("status") == "error" else 0
    except (SourceError, PlannerError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by researcher.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
