# Architecture and milestones

## Product goal

Help a researcher turn a natural-language query into a relevant, reviewable paper collection and an answer grounded in that collection. The researcher can refine the search plan and paper selection before synthesis.

## Proposed flow

```text
Research query + filters
          |
    Search planner
          |
   Human plan review <---- query refinement
          |
    Paper source adapters
          |
    Normalize + deduplicate + rank
          |
   Human paper selection
          |
    Abstract / permitted full-text ingestion
          |
    Chunking + embedding + retrieval
          |
    Answer synthesis + citations
```

## Component boundaries

| Component | Responsibility |
| --- | --- |
| Query planning | Turn the research question into search queries and explicit filters. |
| Source adapters | Search scholarly sources through a shared interface; preserve source identifiers and URLs. |
| Paper catalog | Normalize metadata and deduplicate by DOI, source ID, and title. |
| Ranking | Score relevance and present enough evidence for researcher selection. |
| Ingestion | Read abstracts and accessible full text while retaining paper and section provenance. |
| RAG | Retrieve relevant chunks and produce answers whose citations resolve to retrieved evidence. |
| Orchestration | Track workflow state, bounded retries, human decisions, and search iterations. |
| Evaluation | Measure retrieval quality, duplicate handling, and citation support on curated queries. |

Implementation modules will be added as these components are built. Model providers, paper APIs, embedding models, and index storage are deliberately undecided at this stage.

## First milestones

1. **Paper search:** implement one source adapter, a query command, normalized metadata, and JSON export.
2. **Assisted selection:** add search-plan review, filters, deduplication, and paper selection.
3. **Evidence retrieval:** ingest selected abstracts/full text, retain provenance, and retrieve relevant chunks.
4. **Grounded answers:** generate answers with resolvable citations and surface missing evidence.
5. **Bounded agent loop:** allow query refinement with explicit search limits and observable actions.
6. **Evaluation:** add reproducible query sets and assess retrieval relevance and citation support.

## Data handling

- Store local papers, metadata, and indexes under `data/`; keep them out of Git.
- Keep secrets in environment variables or an untracked `.env` file.
- Store small synthetic test fixtures under `tests/fixtures/`.
- Preserve the origin of every paper and retrieved passage so users can inspect the supporting evidence.
