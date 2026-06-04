# Armchair — Quick Context for Claude

A locally-runnable semantic similarity engine for 35 UIUC MechSE courses. Goal:
surface **non-obvious cross-domain connections** between course topics
(ME 340 spring-mass ↔ ECE 206 RLC circuit — same 2nd-order ODE, different domain).

**Authoritative docs (read these for anything substantive):**
- `README.md` — quick start + doc index
- `docs/ARCHITECTURE.md` — system design, file reference, scoring math, API, UI
- `docs/HISTORY.md` — what changed and why (Phase 12 = Postgres-only rewrite, Phase 13 = archive consolidation)
- `.archive/README.md` — quarantined paths (legacy local docker, Milvus, Oracle, deploy/scripts extras)

**One pipeline.** Local dev = `uvicorn` against the same Supabase as production.
Single env file at the repo root (`.env`, gitignored) is the source of truth for
DATABASE_URL, GEMINI_API_KEY, and Fly deploy keys. `deploy/deploy.sh` reads it.

## The one-liner you'll be asked about most

```
non_obvious_score = sem_score × category_jsd
final_score       = (ALPHA_LEX·lex + ALPHA_SEM·sem_calibrated + ALPHA_CAT·(1-cat_jsd)) / Σ
```

Defaults: `ALPHA_LEX=0.3, ALPHA_SEM=0.5, ALPHA_CAT=0.2`. All knobs live in `config.yml`.

Math scores; LLM only explains. Gemini key lives in browser `localStorage` and is
sent as `X-Api-Key` — server never stores it.

## Current architecture (Phase 12 — June 2026)

**One persistent store: Postgres (Supabase) with pgvector.** Neo4j has been
removed entirely; graph stats (`community`, `pagerank`) are columns on
`courses`, recomputed in-process by `scoring/graph_compute.py` via NetworkX
Louvain + PageRank over `similarity_cache` edges ≥ `MIN_SCORE`.

**Single config:** `config.yml` is the source of truth. `config.py` reads it
with precedence `env > yaml > default`.

**Embedding runtime is switchable:** PyTorch (default) or ONNX INT8 (opt-in
once you run `scripts/export_scincl_onnx.py` and flip `model.embedding.runtime`
in `config.yml`). Same model_id is stamped on every `chunk_embeddings` row.

**Pgvector is HNSW-indexed** (cosine, `m=16`, `ef_search=64`).

**Every API request is logged** to a `query_log` table by middleware in
`api/main.py`. Best-effort; never fails a request.

## Recovery / data restoration

The full LLM-enriched dataset (35 courses, 938 topics with categories +
cross-domain tags) is preserved in `data/courses_dump.json`. To rebuild a
fresh Supabase project end-to-end with zero Gemini calls:

```bash
python scripts/restore_from_dump.py
```

## Research context

- Supervisor: Eliot Bethke (bethke2@illinois.edu)
- Goal: non-obvious matches spanning departments (fluids ↔ mechanics, electronics ↔ materials)
- The team wants clean, explained, non-obvious matches they can evaluate without reading score math
