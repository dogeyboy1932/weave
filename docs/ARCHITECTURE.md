# SIIP Architecture

Single source of truth for system design, file layout, scoring math, APIs, and UI.
For the user-facing "what is this and what can I do with it" pitch see `README.md`.
For what changed in the June 2026 rewrite (Neo4j removed, ONNX-ready, config.yml),
see `docs/HISTORY.md`.

---

## 1. Context

**Research goal** (supervisor Eliot Bethke, UIUC MechSE): build an engine that
surfaces *non-obvious* cross-domain conceptual overlap across a curriculum.
Standard similarity engines bury those connections under same-domain noise
(every ME course is "similar" to every other ME course). The driving insight
from Bethke: *"semantic similarity on topic:definition strings, weighted by
dissimilarity of category composition."*

That weighting — penalising same-department matches — is what makes the engine
useful. ME 340 ↔ ME 310 is uninteresting (same mechanics curriculum); ME 340 ↔
ECE 206 is interesting (same second-order ODE, different department).

**The two key scores:**

```
final_score        = (ALPHA_LEX·lex + ALPHA_SEM·sem_calibrated + ALPHA_CAT·cat_sim) / Σ
non_obvious_score  = sem · category_jsd
```

`final_score` is the conventional hybrid (lexical + semantic + category agreement).
`non_obvious_score` is the headline signal: it stays high *only* when topics are
semantically close **and** the courses live in different categorical bins.
Same-department pairs collapse to ~0 on the `category_jsd` term.

**Design principle: math scores, LLM explains — never the reverse.** Scores
are deterministic and reproducible. The LLM (Gemini) is used in exactly two
places: (a) at ingest, one call extracts topics + 8-bin category distribution
per topic; (b) on user demand, one call renders an explanation for a pair the
math has already ranked. The Gemini key lives in the browser's localStorage
and ships as `X-Api-Key`; the server never stores it.

---

## 2. End-to-end walkthrough (a course's journey)

A PDF lands at `POST /ingest/pdf`. Background task, ~60 s:

```
PDF
 │
 ▼
[1] Gemini analyze_course()
    ├── description (2-3 sentences)
    ├── objectives[]
    └── topics[] each with:
        ├── name
        ├── one-sentence definition
        ├── 8-bin category distribution {Mechanics, Thermodynamics, Electrical,
        │   Fluids, Materials, Mathematics, Chemistry, Systems} summing to 1.0
        └── 5-15 cross-domain tags
 │
 ▼
[2] Chunker (scoring/chunker.py)
    ├── topic chunks    "Course Name [SEP] Topic: LLM definition"   (embedded)
    ├── definition      same shape, supplementary                    (embedded)
    ├── description     not embedded — term counts only
    └── objectives      not embedded — term counts only
    Also: term_counts {token: count} with domain stopwords removed
 │
 ▼
[3] SciNCL encoder (storage/vectors/encoder.py)
    768-dim float32, L2-normalised (so cosine = dot product).
    Runtime selected by config.yml: pytorch (default) or onnx (INT8 quantized).
    Either runtime stamps the same model_id onto chunk_embeddings.model_version.
 │
 ▼
[4] Persistence — Postgres ONLY (no Neo4j)
    ├── courses          (incl. community, pagerank columns)
    ├── chunks
    ├── term_counts
    ├── topic_categories (incl. cross-domain tags)
    └── pgvector chunk_embeddings (HNSW indexed, model_version stamped)
 │
 ▼
[5] Pairwise scoring (scoring/hybrid_scorer.score_all_pairs)
    For each other course B:
      lex_score      = cosine(tfidf_A, tfidf_B)
      sem_score      = symmetrised SciNCL cosine via ANN (sem_floor calibrated)
      category_jsd   = JSD(avg category dist A, avg category dist B)
      final_score    = (ALPHA_LEX·lex + ALPHA_SEM·sem_cal + ALPHA_CAT·(1-cat_jsd)) / Σ
      non_obvious    = sem · category_jsd
      driving_terms  = min(P(w|A), P(w|B)) · IDF(w)   top 8
    All cached in Postgres similarity_cache (every pair, no threshold).
 │
 ▼
[6] Topic-to-topic ANN backfill (build_for_courses)
    For each topic chunk: ANN search against chunks of OTHER courses.
    Pairs with raw cosine ≥ 0.70 → topic_similarity table (HNSW-served).
 │
 ▼
[7] Graph stats (scoring/graph_compute.recompute_graph_stats)
    NetworkX Louvain + PageRank over similarity_cache edges ≥ MIN_SCORE.
    Writes (community, pagerank) back into the courses table.
    No Neo4j. Same NetworkX algorithm that previously served as the Aura-Free
    fallback, so node colours and PageRank values are bit-identical for any
    given input.
 │
 ▼
[8] Cache invalidation
    /topics/search LRU cleared. query_log middleware records the ingest call.
```

Result: the new course shows up in every view (Map, Courses, Search Topics)
with neighbours, community colour, PageRank-sized node, driving terms, and
ranked non-obvious matches — without any further user action.

**Explanations are separate.** When a user clicks "Generate explanation" on a
pair, `GET /similarity/explain` makes a second Gemini call (using the user's
key) returning `{shared_math, why_surprising, analogy}`. Cached in
`similarity_cache.llm_explanation` forever — first viewer pays, everyone after
gets it free.

---

## 3. Tech stack

| Service / module | Role | Local port |
|---|---|---|
| FastAPI | REST API + serves the SPA | 8080 |
| PostgreSQL (Supabase) | **All persistent state** — courses, chunks, scores, categories, embeddings (pgvector), graph stats, query log | 5432 |
| pgvector | 768-dim SciNCL embeddings + HNSW cosine index | (in-DB) |
| NetworkX (in-process) | Louvain communities + PageRank, computed on every graph rebuild | n/a |
| SciNCL (`malteos/scincl`) | Sentence transformer; runs as PyTorch (dev/default) **or** ONNX INT8 (lightweight prod) | local |
| Gemini | Topic extraction, category labelling, explanations | remote |

Production and local dev share the same Supabase project. **Neo4j has been
removed entirely** — see § 13.

---

## 4. Data model

### PostgreSQL

```
courses(course_id PK, name, description, prereqs, credits, sequence, course_type,
        instructors, community INT, pagerank FLOAT)
   community / pagerank are written by scoring/graph_compute.py after every
   similarity rebuild — these used to live in Neo4j.

chunks(chunk_id PK, course_id FK, chunk_type, raw_text, keyphrases)
  chunk_type ∈ {topic, definition, description, objective}
  topic+definition chunks are embedded; others contribute term counts only
  topic chunk text format: "Course Name [SEP] Topic: LLM-generated definition"

chunk_embeddings(chunk_id PK, course_id, embedding vector(768), model_version)
  Indexed: HNSW (vector_cosine_ops, m=16, ef_search=64). pgvector ≥ 0.5.
  model_version is the EMBEDDING_MODEL_ID stamp at insert time so a mixed
  PyTorch/ONNX dataset is detectable. Configurable via config.yml.

term_counts(course_id, term, count)   -- composite PK; used for LM + TF-IDF

topic_categories(course_id, topic_text, categories JSONB, tags JSONB, labeled_at)
  categories: {"Mechanics": 0.6, "Mathematics": 0.3, ...}  -- sums to 1.0, 8 bins
  tags:       ["second-order-ode", "natural-frequency", ...]  -- cross-domain

topic_similarity(course_a, topic_a, course_b, topic_b, sem_score)
  pre-computed topic-to-topic ANN pairs above raw cosine 0.70

similarity_cache(course_a, course_b, final_score, lex_score, sem_score, jsd,
                 driving_terms, category_jsd, non_obvious_score, llm_explanation,
                 computed_at)
  stored with course_a < course_b; every pair, no threshold

topic_explanations(source_course, source_topic, target_course, target_topic,
                   explanation, signed_by, generated_at)
  cached topic-pair LLM explanations (phase-2 feature)

query_log(id BIGSERIAL, ts, endpoint, method, query_params JSONB, status_code,
          result_count, latency_ms, user_agent_hash)
  one row per API call. Cheap structured telemetry; offline eval data.
```

### Knowledge graph (in-process, not a separate database)

- Nodes: `courses` rows (id, name, community, pagerank).
- Edges: `similarity_cache` rows with `final_score ≥ MIN_SCORE` (0.55).
- Communities + PageRank: NetworkX Louvain run by
  `scoring/graph_compute.recompute_graph_stats()` after every score rebuild.
- Shortest path / neighbours: NetworkX over the same edge list (sub-second at
  hundreds of courses).

---

## 5. Scoring pipeline

Every course pair gets these scores, all stored in `similarity_cache`.

### 5.1 Lexical (`lex_score`) — TF-IDF cosine

```
lex_score = cosine(tfidf(A), tfidf(B))
```
Term counts from all chunks with domain stopwords removed. Sparse by design —
only shared terms contribute. Range [0, 1].

### 5.2 Semantic (`sem_score`) — SciNCL + ANN

```
directed(A→B) = mean over A-chunks of: max cosine(chunk_A, chunks_B)
sem_score     = 0.5·directed(A→B) + 0.5·directed(B→A)
```
Embeddings are L2-normalised at encode time, so cosine == dot product. Raw
cosine on engineering text typically sits in 0.65–0.90.

The `sem_floor` config knob (default 0.35) is the random-pair baseline used
during hybrid blending — see § 5.3.

### 5.3 Hybrid (`final_score`)

```
sem_cal   = max(0, (sem - sem_floor) / (1 - sem_floor))   # [0, 1]
cat_sim   = 1 - category_jsd                              # may be None
weighted  = ALPHA_LEX·lex + ALPHA_SEM·sem_cal + ALPHA_CAT·cat_sim
final     = weighted / (ALPHA_LEX + ALPHA_SEM + ALPHA_CAT)
```
Defaults: `ALPHA_LEX=0.3, ALPHA_SEM=0.5, ALPHA_CAT=0.2`. When `cat_jsd` is
missing the ALPHA_CAT weight is redistributed proportionally to lex and sem.
All knobs live in `config.yml` under `scoring:`.

### 5.4 Language-model JSD (`jsd`) — reference only

Stored for diagnostic use; not in the headline ranking math.

### 5.5 Category JSD (`category_jsd`)

8 bins: `Mechanics | Thermodynamics | Electrical | Fluids | Materials | Mathematics | Chemistry | Systems`

```
category_vec(course) = mean of topic category distributions for that course
category_jsd         = JSD(category_vec(A), category_vec(B))    # [0, 1]
```
Topic distributions come from the LLM during ingest.

### 5.6 Non-obvious (`non_obvious_score`) — the key signal

```
non_obvious_score = sem_score × category_jsd
```
- `sem=0.9, cat_jsd=0.9` → jackpot cross-domain match
- `sem=0.9, cat_jsd=0.1` → obvious same-department match

### 5.7 Driving terms (explainability)

```
driving_score(w) = min(P(w|A), P(w|B)) · log(N / (1 + df(w)))
```
Only terms present in both raw vocabularies count. Top-8 stored per pair.

---

## 6. LLM layer

**Principle: math narrows, LLM explains. LLM never scores.**

### Job 1 — full course analysis (one call per ingest)
`llm/topic_extractor.py → analyze_course()` using `gemini-2.5-flash-lite`.
Input: raw PDF/TXT. Single JSON response with `description`, `objectives[]`, and
`topics[]` (each with name, definition, 8-category distribution, cross-domain
tags). No further LLM calls during ingest.

### Job 2 — connection explanation (on-demand only)
`llm/llm_explainer.py` using `gemini-2.5-flash`. Returns `shared_math`,
`why_surprising`, `analogy`. Cached forever in `similarity_cache.llm_explanation`.

**API key flow:** The Gemini key lives only in the browser (`localStorage`) and is
sent as `X-Api-Key` on every LLM-touching request (ingest + explain). The server
never stores it. Missing key → HTTP 400 with the actual Gemini error.

---

## 7. Ingest

### New course — `POST /ingest/pdf` (accepts `.pdf` and `.txt`)

Background task (~60 s):
1. LLM `analyze_course()` — one call, returns everything
2. Chunk topics into `"Course [SEP] Topic: definition"` strings
3. SciNCL encode topic chunks → pgvector (PyTorch or ONNX runtime, configurable)
4. Upsert course + chunks + term counts + category distributions in Postgres
5. Score against all N-1 existing courses (lex + sem + category_jsd + non_obvious)
6. Topic-to-topic ANN backfill (`topic_similarity`)
7. `graph_compute.recompute_graph_stats()` — Louvain + PageRank, write to `courses`

UI polls `GET /ingest/status/{course_id}` → `{status, message}`. On error,
partial data is cleaned up via `pg_store.delete_course` + `vs.delete_course`.

### Append to existing course — `POST /ingest/append`

For lecture notes, slides, readings. Same pipeline but:
- No new course node created
- Chunk IDs are timestamped to avoid collision (`CS_521_app1746900000__topic__0`)
- Term counts **accumulate** (added, not replaced)
- New topics extracted and **upserted** into `topic_categories`
- All similarity scores for this course are recomputed from the full combined dataset

---

## 8. Topic-to-topic similarity

Powers the Topics view. `scripts/build_topic_graph.py` runs an ANN search for every
topic against all chunks in *other* courses, stores hits with `sem ≥ 0.70` in
`topic_similarity`. Pure pgvector — zero LLM cost. ~2–5 min for the full corpus.

```bash
python scripts/build_topic_graph.py                           # all courses
python scripts/build_topic_graph.py --course "ME 340"         # one course
python scripts/build_topic_graph.py --top-k 10 --min-score 0.70
```

---

## 9. API

### Courses
```
GET /courses
GET /courses/{course_id}
GET /courses/{course_id}/topics      → topics with 8-category distributions
```

### Similarity
```
GET /similarity?a=ME 340&b=ECE 205
GET /similarity/neighbors?course=ME 340&top=10&sort=hybrid|non_obvious
GET /similarity/non-obvious?top=20&min_sem=0.3
GET /similarity/categories?course=ME 340
GET /similarity/explain?a=ME 340&b=ECE 205     # cached, or generates with X-Api-Key
```

### Topics
```
GET /topics/search?q=vibration&top=40
```

### Graph (Postgres-backed, NetworkX in-process)
```
GET /graph/all?min_score=0.4          → {nodes, edges} for the D3 view
GET /graph/path?from=ME 200&to=ME 370 → shortest similarity path
GET /graph/neighbors?course=ME 340    → top neighbours with names
GET /graph/communities                → {community_id: [course_id, ...]}
```

### Ingest
```
POST /ingest/pdf       body: {file, course_id, course_name}
POST /ingest/append    body: {file, course_id}
GET  /ingest/status/{course_id}
```

### Meta
```
GET /health   → {"status":"ok"}
GET /docs     → Swagger UI
```

Every non-trivial request is recorded in `query_log` by a middleware in
`api/main.py`. The middleware is best-effort — a logging failure never causes
a request failure.

---

## 10. UI — single SPA

Vanilla JS + D3.js v7, no build step. All three views live in `ui/index.html`.

### Graph view
D3 force-directed graph. Node size = PageRank, colour = Louvain community.
Toggle Normal/Non-obvious mode swaps edge weights (final_score vs
non_obvious_score — non-obvious mode uses a distinct orange). Threshold slider
+ click-to-highlight.

### Explorer view
Pick a course → ranked neighbours (by hybrid or non-obvious). Each card shows
`final_score` + `non_obvious_score` badges, 8-bin category distribution bar,
driving terms, and a "Generate explanation" button (cached afterwards). Right
sidebar has full topic list with dominant-category dots.

### Topics view
Keyword search over `topic_categories`. Left: matching topics with
"Similar topics in other courses" expansion (from `topic_similarity`). Right:
course-level similarity matrix for the keyword set.

### Upload page (`/upload`)
Mode toggle: **New Course** vs **Append to Existing**. Drag-and-drop `.pdf`/`.txt`.
Polls `/ingest/status` every 3 s; surfaces real error messages.

### API key panel (⚙)
Password input, stored in `localStorage`. Sent as `X-Api-Key` on all LLM-touching
requests. Button border turns purple when set. Server never sees / persists it.

---

## 11. File layout

```
akhil_app/
├── README.md, CLAUDE.md            # entry + Claude context
├── config.py                       # thin loader; reads config.yml + env
├── config.yml                      # single source of truth for all knobs
├── Dockerfile.api                  # production image (pytorch OR onnx mode)
├── fly.toml                        # Fly app config (Postgres-only secrets)
├── requirements.txt                # local dev / scripts deps
├── requirements-api.txt            # production deps
│
├── api/                            # FastAPI app
│   ├── main.py                     # lifespan, query_log middleware, routers
│   └── routes/
│       ├── courses.py, similarity.py, topics.py, ingest.py
│       └── graph.py                # reads Postgres + NetworkX (no Neo4j)
│
├── scoring/                        # All math + chunker preprocessing
│   ├── chunker.py
│   ├── language_model.py
│   ├── jsd.py
│   ├── vector_similarity.py
│   ├── driving_terms.py
│   ├── category_scorer.py
│   ├── hybrid_scorer.py
│   └── graph_compute.py            # NetworkX Louvain + PageRank → Postgres
│
├── llm/                            # All Gemini calls
│   ├── topic_extractor.py          # analyze_course() — only ingest LLM call
│   ├── category_labeler.py
│   └── llm_explainer.py
│
├── storage/
│   ├── postgres/store.py           # courses (+ community, pagerank), chunks,
│   │                               # term_counts, similarity_cache,
│   │                               # topic_categories, query_log, helpers
│   └── vectors/                    # Embedding + vector store
│       ├── encoder.py              # pytorch | onnx dispatcher (config-driven)
│       ├── store.py                # thin re-export of pgvector_store
│       └── pgvector_store.py       # HNSW-indexed; model_version stamped
│
├── ui/                             # static UI served by FastAPI
│   ├── index.html
│   └── upload.html
│
├── scripts/                        # six load-bearing scripts
│   ├── restore_from_dump.py        # rebuild from data/courses_dump.json, zero Gemini
│   ├── build_graph.py              # all pairs + graph_compute
│   ├── build_topic_graph.py        # topic-to-topic ANN (imported by ingest + restore)
│   ├── dump_courses.py             # refresh the canonical courses_dump.json
│   ├── export_scincl_onnx.py       # SciNCL → ONNX INT8 + MANIFEST.json
│   └── reset.py                    # wipe (supports --restore)
│
├── models/                         # ONNX bundle (gitignored; created by export)
│   ├── scincl-int8.onnx
│   ├── tokenizer.json / etc.
│   └── MANIFEST.json               # {id, dim, sha256, parity cosines, exported_at}
│
├── data/                           # source JSON + canonical dump
│   ├── mechse_syllabi.json
│   ├── topic_definitions.json
│   ├── course_info.json, instructors.json
│   ├── courses_dump.json           # full LLM-enriched snapshot (35 courses)
│   └── edit/                       # per-course edit buffers
│
├── docs/
│   ├── ARCHITECTURE.md             # (you are here)
│   └── HISTORY.md                  # June 2026 rewrite notes
│
├── .env                            # single source of truth — DATABASE_URL,
│                                   # GEMINI_API_KEY, FLY_*, etc. (gitignored)
├── deploy/
│   └── deploy.sh                   # verify Supabase + push secrets + fly deploy
│                                   # + smoke /health; reads .env directly
│
├── .archive/                       # quarantined paths (see .archive/README.md)
│   ├── legacy-local-docker/        # original local docker stack + milvus_store.py
│   │                               # + Oracle self-host (ORACLE.md, oci/)
│   ├── deploy-extras/              # deploy/ items now redundant (README, examples,
│   │                               # install-flyctl, setup-ci, Neo4j creds backup)
│   └── scripts-extras/             # 9 one-time / niche scripts superseded by
│                                   # restore_from_dump.py
│
└── .model_cache/                   # SciNCL weights (PyTorch path); gitignored
```

---

## 12. Environment

All non-secret knobs live in `config.yml`. Environment variables override
individual values, so Fly secrets and per-environment `.env` overrides still
work. Precedence: env var > config.yml > built-in default.

```yaml
# config.yml (excerpt)
model:
  embedding:
    id:        scincl-v1-fp32
    hf_name:   malteos/scincl
    runtime:   pytorch          # or "onnx"
    onnx_path: models/scincl-int8.onnx
    dim:       768
scoring:
  alpha_lex: 0.3
  alpha_sem: 0.5
  alpha_cat: 0.2
  sem_floor: 0.35
  min_score: 0.55
retrieval:
  ann:
    index_type: hnsw
    hnsw_m: 16
    hnsw_ef_search: 64
```

Secrets stay env-only:

```bash
# Required
DATABASE_URL=postgres://...        # Supabase Direct connection
GEMINI_API_KEY=                    # optional; UI also accepts user-supplied keys

# Optional per-environment overrides
POSTGRES_SSLMODE=require
```

---

## 13. June 2026 rewrite — what changed (and why)

The May 2026 outage exposed a structural fragility: the production stack
depended on two free-tier managed services (Supabase Postgres + Neo4j Aura),
both of which auto-pause after ~1 week of inactivity and are reclaimed after
extended pauses. Both projects' DNS records went NXDOMAIN. The June rewrite
addresses both the recovery problem and the broader engineering principles
laid out by the team.

**1. Neo4j removed entirely.**
Graph stats now live in `courses.community` + `courses.pagerank`, computed
in-process by `scoring/graph_compute.py` using the same NetworkX Louvain +
PageRank that previously served as the Aura-Free fallback. One less service to
keep alive; one less failure mode. At 35 nodes / ~123 edges this is sub-second.

**2. `config.yml` as single source of truth.**
All scoring weights, model IDs, ANN parameters, and retrieval knobs live in
one YAML. `config.py` is now a thin loader that respects env > yaml > default.

**3. Pgvector HNSW index + model versioning.**
`chunk_embeddings` is now HNSW-indexed (cosine, `m=16`, `ef_search=64`) so
retrieval scales sub-linearly. Every row carries a `model_version` stamp that
matches `config.EMBEDDING_MODEL_ID`, so a mixed-model dataset (PyTorch vs ONNX
INT8) is detectable in SQL.

**4. ONNX-INT8 serving (opt-in).**
`scripts/export_scincl_onnx.py` exports SciNCL → INT8 ONNX with a parity
check against the FP32 PyTorch baseline (cosine ≥ 0.998 on a golden set).
The output bundle is committed with a `MANIFEST.json` recording id, dim,
sha256, and the worst-case cosine. `storage/vectors/encoder.py` dispatches
to either runtime based on `model.embedding.runtime`. The ONNX path drops
~700 MB of PyTorch from the Fly image (target: ~150 MB) and keeps the web
server free of any heavy model framework. Build with
`docker build --build-arg EMBEDDING_RUNTIME=onnx`.

**5. `query_log` table + middleware.**
Every API call records endpoint, params, status, latency, and a hashed UA.
Logging is best-effort — a logging failure never fails a request. The table
becomes free offline-evaluation data.

**6. `restore_from_dump.py`.**
The complete LLM-enriched dataset (35 courses, 938 topics with categories +
cross-domain tags) is preserved in `data/courses_dump.json`. The restore
script rebuilds Postgres + pgvector + similarity_cache + graph stats end-to-end
with **zero Gemini calls**. Only on-demand explanations remain non-restorable;
they regenerate naturally on first user click.

**7. Heartbeat workflow.**
`.github/workflows/heartbeat.yml` curls `/health` weekly so Supabase never
crosses its inactivity-pause threshold.

---

## 14. Design principles

**Math scores, LLM explains — never the reverse.** Scores are deterministic and
reproducible. The LLM only adds the human-readable layer after the math has ranked.

**Same model for index and query.** A single `encoder.encode()` function is used
at ingest *and* anywhere else embeddings are generated. The model_version stamp
on every row makes this auditable in SQL.

**Offline-heavy, online-light.** SciNCL only runs at ingest. Every user query
reads pre-computed scores from `similarity_cache`. ONNX INT8 brings the model
itself down to ~110 MB; serving stays well under one core.

**Config-driven everything.** Scoring weights, model paths, ANN parameters —
all in `config.yml`. Env vars override per environment.

**Don't brute-force vector search.** HNSW (cosine, m=16, ef_search=64) on
`chunk_embeddings` — exact FLAT was fine at 1.7k vectors but the principle is
right past ~100k.

**Log every query.** Endpoint + params + latency + result count. Free eval data.

**One LLM call per ingest.** Topic extraction, definitions, category labels,
description, objectives, and tags all come from one `analyze_course()`.

**No silent fallbacks.** Bad key, rate limit, invalid JSON → ingest stops,
partial data cleaned up, exact error surfaced to the UI.

**Cache aggressively.** Topic categories: permanent. Scores: recomputed only on
course change. LLM explanations: generated once, cached forever.

**Append-not-replace** for supplementary material. Term counts accumulate; a
course profile only ever gets richer.

---

## 15. Current state (June 2026)

| Metric | Count |
|---|---|
| Courses | 35 |
| Topics labelled | 938 (every topic has categories + tags) |
| Similarity pairs | 595 (35×34/2) |
| Embedding chunks | ~1,700 |
| Graph edges (final_score ≥ 0.55) | ~123 |
| Topic similarity pairs | ~6,700 |
| Persistent stores | Postgres + pgvector (Supabase) — **one** managed service |

---

## 16. Future work

**Roll the ONNX bundle live.** Run `scripts/export_scincl_onnx.py` on a dev
box, commit `models/scincl-int8.onnx` + `MANIFEST.json`, set `model.embedding.runtime: onnx`
in `config.yml`, rebuild with `--build-arg EMBEDDING_RUNTIME=onnx`, deploy.
Cuts image size by ~5× and removes PyTorch from prod.

**Object-storage versioning for model artifacts.** Once the ONNX bundle is in
production, move it out of the repo and into versioned object storage (S3,
Supabase Storage, GitHub Releases). The Docker build pulls a pinned hash;
rollback is "change the hash in config.yml and redeploy."

**Eval harness from `query_log`.** Replay weekly query logs against scoring
weights to A/B-test ALPHA_LEX/SEM/CAT changes offline.

**Topic-to-topic LLM explanations.** Phase 1 (topic_similarity table) is shipped.
Phase 2 adds on-demand LLM explanations per topic pair, cached forever in
`topic_explanations`.

**Score refinement via LLM feedback.** Use explanation quality ("actually not
related") to downweight/upweight raw `sem_score`, stored as a separate
`refined_score` column.
