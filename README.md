# Assisted Agentic RAG Paper Retriever

A Python project for finding research papers from a natural-language query, with human guidance during search and evidence-backed answers using retrieval-augmented generation (RAG).

**Status:** project scaffold. Paper search, agent orchestration, indexing, and answer generation are planned; they are not implemented yet.

## Intended workflow

1. Enter a research question, scope, and optional filters.
2. Review the agent's proposed search queries and refine the plan.
3. Search paper sources and deduplicate candidate papers.
4. Rank results and select papers to keep or explore further.
5. Index abstracts and available full text for retrieval.
6. Generate an answer with citations to the retrieved evidence.

See [the architecture and milestones](docs/architecture.md) for the implementation plan.

## Local setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paper-retriever --version
```

The starter command currently provides help and version information. No API keys or external services are needed for these checks.

## Project layout

```text
src/agentic_paper_retrieval/   Python package and CLI entry point
docs/architecture.md          Workflow, boundaries, and implementation milestones
tests/                       Future behavioral and retrieval evaluation tests
data/                        Local papers, metadata, and indexes (Git-ignored)
.github/workflows/ci.yml      Build, installation, and CLI checks
```

Keep provider credentials in environment variables or a local `.env` file. Credentials, downloaded papers, indexes, and generated results are excluded from Git. Commit small synthetic evaluation fixtures under `tests/fixtures/` when needed.

## Development checks

```bash
python -m pip install build
python -m build
paper-retriever --help
python -m agentic_paper_retrieval --version
```

CI builds and installs the wheel in a fresh environment, then checks both entry points. Add behavioral tests as retrieval functionality is implemented.
