# Assisted Agentic RAG Paper Retriever

A Python project for finding research papers from a natural-language query, with human guidance during search and evidence-backed answers using retrieval-augmented generation (RAG).

**Status:** arXiv search and an assisted Ollama search agent are implemented. Full-text ingestion, embeddings, and RAG answer generation are the next milestones.

## Try paper search

Requires Python 3.11 or newer. Direct search needs internet access; no API key or language model is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paper-retriever search "retrieval augmented generation" --limit 5
```

Save normalized metadata, abstracts, source identifiers, and paper/PDF links:

```bash
paper-retriever search "retrieval augmented generation" --limit 10 --output outputs/papers.json
paper-retriever search "retrieval augmented generation" --json
```

Plain queries become keyword searches joined with `AND`; common question words are removed. Double-quoted phrases stay together. This is lexical search and results use arXiv's relevance order. For explicit fields, Boolean operators, categories, or date filters, use arXiv syntax:

```bash
paper-retriever search 'cat:cs.CL AND ti:"retrieval augmented"' --raw-query --limit 10
```

## Try the local search agent

Install [Ollama](https://docs.ollama.com/macos), start its server in a separate terminal, and download a model supporting structured JSON output. A starting model is [qwen3:4b](https://ollama.com/library/qwen3:4b), a roughly 2.5 GB download.

```bash
ollama serve
```

In your project terminal:

```bash
ollama pull qwen3:4b
source .venv/bin/activate
paper-retriever research "How does retrieval augmented generation reduce hallucinations?" --model qwen3:4b --max-rounds 3 --output outputs/research.json
```

The model proposes keywords and explains its choice. Press Enter to approve a search, type a replacement query to edit it, or enter `/stop`. After each search, the model reads returned titles and abstract excerpts, then proposes a refinement or finishes with a paper shortlist. Models do not execute arbitrary tools; the runner can only search the configured paper source.

Use `--auto` to allow searches within the same round limit without approval prompts:

```bash
paper-retriever research "RAG hallucination evaluation benchmarks" --model qwen3:4b --max-rounds 2 --auto --json --output outputs/research.json
```

`--model` can also be set through `OLLAMA_MODEL`. `--ollama-url` defaults to `http://localhost:11434` and can also be set through `OLLAMA_BASE_URL`. Environment variables are read from the shell; `.env` files are not loaded automatically.

## What the agent does

- Plans searches from the research question and refines them using retrieved evidence.
- Records proposed queries, researcher edits, actual source queries, and returned paper IDs.
- Combines papers across rounds, deduplicating arXiv versions by their stable identifier.
- Stops after at most 1–5 searches (3 by default), an earlier model decision, a repeated query, or a researcher stop.
- Allows at most one model decision per search plus a final selection decision when the budget is reached.
- Rejects selections containing paper IDs outside the evidence supplied to the model.

The JSON export contains all retrieved candidates, the ordered selected IDs, the action history, and the stopping reason. Progress and approval prompts go to stderr so `--json` stdout remains machine-readable.

## Current limits

This version discovers and shortlists papers from metadata. It does not download PDFs, build a vector index, or produce a research answer. It searches only arXiv, so coverage depends on that source. The planner sees up to the 20 most recently collected papers and 500 characters from each abstract; its relevance judgments need researcher review and evaluation. Research mode accepts up to 20 results per search; direct search accepts up to 50.

Successful API responses are cached for 24 hours under `data/cache/arxiv/`. Network requests are serialized and spaced at least three seconds apart within one process, following the [arXiv API guidance](https://info.arxiv.org/help/api/tou.html). Avoid concurrent retriever processes sharing the same upstream API allowance.

## Intended workflow

1. Enter a research question, scope, and optional filters.
2. Review the agent's proposed search queries and refine the plan.
3. Search paper sources and deduplicate candidate papers.
4. Rank results and select papers to keep or explore further.
5. Index abstracts and available full text for retrieval.
6. Generate an answer with citations to the retrieved evidence.

See [the architecture and milestones](docs/architecture.md) for the implementation plan.

## Project layout

```text
src/agentic_paper_retrieval/
  models.py                  Normalized metadata
  sources/                   Reusable paper-source interface and arXiv adapter
  ollama.py                  Local model planner with structured decisions
  agent.py                   Bounded assisted search orchestration
  __main__.py                CLI and JSON export
docs/architecture.md         Workflow, boundaries, and implementation milestones
tests/                      Offline behavioral tests and synthetic fixtures
data/                        Local papers, metadata, and indexes (Git-ignored)
.github/workflows/ci.yml      Package checks and behavioral tests
```

Keep provider credentials in environment variables or a local `.env` file. Credentials, downloaded papers, indexes, and generated results are excluded from Git. Commit small synthetic evaluation fixtures under `tests/fixtures/` when needed.

## Development checks

```bash
python -m pip install build
python -m build
python -m unittest discover -s tests -v
```

CI builds and installs the wheel in a fresh environment, then checks both entry points and runs the offline suite on Python 3.11 and 3.12. Tests cover metadata parsing, API failures, caching, human edits, deduplication, search limits, invented citations, Ollama's request contract, and JSON export. They require neither internet access nor a running model.

API references: [arXiv](https://info.arxiv.org/help/api/user-manual.html), [Ollama chat](https://docs.ollama.com/api/chat), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
