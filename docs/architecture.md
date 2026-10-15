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

## Implemented first slice

- `models.py` defines normalized paper metadata and search results.
- `sources/base.py` defines the source interface; `sources/arxiv.py` implements keyword/raw search, stable arXiv IDs, version deduplication, a 24-hour cache, and request pacing.
- `ollama.py` calls a local Ollama chat server with a JSON schema. The model can propose a search or finish with retrieved paper IDs.
- `agent.py` enforces a bounded search loop, researcher approval/edits, repeated-query detection, and paper-ID validation. Source-supplied relevance order is retained; the model can order its final shortlist.
- The CLI exposes `search` and `research`, readable output, and JSON exports. Direct search works without Ollama.

The first model provider is local Ollama and the first paper API is arXiv. Embedding models and index storage remain undecided. Source metadata and abstract excerpts inform search refinement; this is not yet a RAG answer-generation pipeline.

## First milestones

1. **Implemented — paper search:** one source adapter, keyword/raw query commands, normalized metadata, and JSON export.
2. **Implemented — assisted agent loop:** Ollama planning/refinement, per-search human approval and edits, version deduplication, a model-selected shortlist, explicit search limits, and an action trace.
3. **Next — retrieval evaluation:** curate realistic questions, label relevant papers, and compare direct search with the agent's shortlist.
4. **Next — researcher paper selection:** let the researcher explicitly approve the final shortlist for ingestion.
5. **Next — evidence retrieval:** ingest selected abstracts/full text, retain provenance, and retrieve relevant chunks.
6. **Next — grounded answers:** generate answers with resolvable citations and evaluate citation support.

## Data handling

- Store local papers, metadata, and indexes under `data/`; keep them out of Git.
- Keep secrets in environment variables or an untracked `.env` file.
- Store small synthetic test fixtures under `tests/fixtures/`.
- Preserve the origin of every paper and retrieved passage so users can inspect the supporting evidence.
