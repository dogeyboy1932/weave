---
title: ArmChair
emoji: 🪑
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

<!-- The YAML block above configures the Hugging Face Space (Docker SDK).
     It's ignored on GitHub (rendered as a metadata table) and required by HF.
     Deploy flow: deploy/HUGGINGFACE.md -->

# ArmChair

UIUC MechSE curriculum mapping. Finds **non-obvious cross-domain connections**
between course topics (ME 340 spring-mass ↔ ECE 206 RLC circuit — same
second-order ODE, different department).

**Live:** https://dogeyboy19-armchair.hf.space  (old Fly URL https://siip-armchair-akhil.fly.dev 301-redirects here)
**Supervisor:** Eliot Bethke (bethke2@illinois.edu)

---

## What this is

A research tool for mapping conceptual overlap across a curriculum. Standard
similarity engines surface obvious same-domain matches (ME 340 ↔ ME 310 — both
mechanical). SIIP is built to surface the **non-obvious** ones — courses whose
topics are mathematically/structurally similar but live in different
departments. The driving insight (from supervisor Eliot Bethke): *"semantic
similarity on topic:definition strings, weighted by dissimilarity of category
composition."*

**The one-line scoring story:**

```
non_obvious_score = sem_score × category_jsd
                     ↑              ↑
                     how similar    how different
                     the topics     the disciplines
                     read           the courses span
```

A pair scores high only when both factors are high. Same-department pairs
collapse to zero on the `category_jsd` term and stop crowding the results.
Full math (TF-IDF + Dirichlet smoothing + JSD + symmetrised SciNCL cosine)
lives in `docs/ARCHITECTURE.md`.

## What you can do with it

The live site (single SPA, no build step) gives three views over the 35-course
corpus:

- **Map** — D3 force-directed graph. Node size = PageRank, colour = Louvain
  community. Toggle Normal/Non-obvious to swap which signal drives edge weights.
- **Courses** — pick a course, see its neighbours ranked by hybrid *or*
  non-obvious score. Each card shows the 8-bin category distribution, driving
  terms, and a "Generate explanation" button (Gemini, on-demand, cached forever).
- **Search Topics** — keyword search over all topic texts. Expanded hits show
  semantically similar topics in *other* courses, surfaced via topic-to-topic ANN.
- **Add syllabus** (`/upload`) — drop in a PDF or TXT. Background pipeline does
  Gemini topic extraction + category labels + SciNCL embedding + scoring against
  every existing course + Louvain re-clustering, end-to-end in ~60 s.

## How it works (end-to-end, 60 seconds)

A course PDF lands at `/ingest/pdf`. The pipeline:

1. **LLM analysis** — one Gemini call returns description, learning objectives,
   topic names, one-line topic definitions, 8-bin category distributions, and
   cross-domain tags. No further LLM calls during ingest.
2. **Chunk + embed** — topics become `"Course Name [SEP] Topic: definition"`
   strings; SciNCL encodes each into a 768-dim vector stored in pgvector with
   an HNSW index.
3. **Persist** — course node + chunks + term counts + category distributions
   into Postgres. The embedding rows carry a `model_version` stamp.
4. **Score** — for every other course, compute `lex_score` (TF-IDF cosine),
   `sem_score` (symmetrised SciNCL ANN), `category_jsd` (JSD over the 8-bin
   averaged distributions), then `final_score` (weighted blend) and
   `non_obvious_score = sem · category_jsd`. All pairs cached in Postgres.
5. **Graph stats** — Louvain + PageRank run in-process via NetworkX over
   `similarity_cache` edges ≥ `MIN_SCORE` (0.55); `community` and `pagerank`
   columns on `courses` are updated.

**Principle: math scores, LLM explains.** Scores are deterministic. The LLM
only renders explanations *after* the math has ranked pairs — and only on user
demand (`/similarity/explain`). The Gemini key lives in the browser's
`localStorage` and is sent as `X-Api-Key`; the server never stores it.

---

## Quick start (cloud-backed local dev)

Local code talks to the same Supabase that production uses.

```bash
# Create .env at the repo root (gitignored) with:
#   DATABASE_URL=postgresql://postgres:<PASSWORD>@db.<REF>.supabase.co:5432/postgres
#   POSTGRES_SSLMODE=require
#   GEMINI_API_KEY=                # optional
#   FLY_APP_NAME=siip-armchair-akhil
#   FLY_API_TOKEN=                 # optional (deploy only)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080. Ship changes with `git push origin main` —
GitHub Actions deploys Fly automatically.

## Restoring data after a wipe / new Supabase project

The complete LLM-enriched dataset is preserved in `data/courses_dump.json`.
Bringing a fresh database fully back up takes one command:

```bash
python scripts/restore_from_dump.py
```

This recreates 35 courses, 938 topics with categories + tags, ~1,700
embeddings, all pairwise scores, the topic-to-topic graph, and Louvain +
PageRank — **with zero Gemini calls**. The only thing not restored is
on-demand connection explanations, which regenerate naturally as users click
"Generate explanation" on the live site.

## Optional: ONNX serving (strip PyTorch from production)

Once the ONNX bundle is exported and committed:

```bash
python scripts/export_scincl_onnx.py
# → models/scincl-int8.onnx + tokenizer + MANIFEST.json (parity-verified)
```

Flip `config.yml`:

```yaml
model:
  embedding:
    runtime:   onnx
    onnx_path: models/scincl-int8.onnx
    id:        scincl-v1-int8
```

Rebuild with `docker build --build-arg EMBEDDING_RUNTIME=onnx`. Image drops
from ~1.5 GB to ~150 MB and PyTorch leaves the production image entirely.

## One pipeline, not two

Local development is just `uvicorn` pointed at the same Supabase the live site
uses. There is **no parallel local-only pipeline** — no separate docker-compose
stack to keep in sync, no local-vs-prod config drift. The cloud-backed flow is
the only flow.

The original local docker-compose stack (Postgres + Milvus + Neo4j containers,
the Oracle self-host path) is preserved under `.archive/legacy-local-docker/`
for historical reference. See `.archive/README.md`.

---

## Docs

| File | Purpose |
|---|---|
| `docs/ARCHITECTURE.md` | System design, file reference, scoring math, API, UI |
| `docs/HISTORY.md`      | What changed in the June 2026 rewrite (and why) |
| `.archive/README.md`   | Quarantined paths (legacy local docker, Oracle, Milvus, deploy/scripts extras) |
| `CLAUDE.md`            | Pointers for the Claude Code assistant |

## Common commands

```bash
# Restore the full corpus from data/courses_dump.json (zero Gemini calls)
python scripts/restore_from_dump.py

# Re-score everything after weight changes (config.yml → scoring)
python scripts/build_graph.py

# Build topic-to-topic similarity (powers Topics tab)
python scripts/build_topic_graph.py

# Refresh the canonical courses_dump.json from the live DB
python scripts/dump_courses.py

# Export the SciNCL ONNX bundle (one-time on a dev box)
python scripts/export_scincl_onnx.py

# Wipe and start over (destructive)
python scripts/reset.py                # wipe
python scripts/reset.py --restore      # wipe + restore from dump
```

Run any script on production with:

```bash
flyctl ssh console --app siip-armchair-akhil \
  -C "sh -c 'cd /app && python scripts/<name>.py'"
```
