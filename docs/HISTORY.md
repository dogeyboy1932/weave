# SIIP / akhil_app — Complete History

This is the consolidated, change-by-change record of every meaningful thing
that has happened on this project, from the original UIUC cPanel site through
today's Fly + Supabase + Aura production stack. It's written as a textbook /
narrative — not just a commit log — so it preserves the *why* behind each
pivot, the bugs that triggered them, and the design principles that emerged.

Sources used to reconstruct this:
- `~/.claude/history.jsonl` (global prompt log, 45 prompts)
- Six session transcripts under `~/.claude/projects/.../*.jsonl`
- Two git repos: `~/Desktop/Projects/UIUC Research/SIIP/` and `~/Desktop/Projects/UIUC Research/akhil_app/` (this repo)
- `docs/EARLIER_DEPLOY.md`, `docs/EARLIER_SESSION_HISTORY.md`, `docs/ARCHITECTURE.md`
- Project memory at `~/.claude/projects/.../memory/`

---

## Table of contents

- [Phase 0 — Origin: UIUC cPanel static site](#phase-0--origin-uiuc-cpanel-static-site)
- [Phase 1 — JSON → AWS migration](#phase-1--json--aws-migration)
- [Phase 2 — Upload pipeline + AI topic extraction](#phase-2--upload-pipeline--ai-topic-extraction)
- [Phase 3 — AWS production deploy + pivot announcement](#phase-3--aws-production-deploy--pivot-announcement)
- [Phase 4 — New repo: akhil_app with Postgres + Milvus + Neo4j](#phase-4--new-repo-akhil_app-with-postgres--milvus--neo4j)
- [Phase 5 — Unified UI + scoring overhaul](#phase-5--unified-ui--scoring-overhaul)
- [Phase 6 — Topics, categories, and the non-obvious pivot](#phase-6--topics-categories-and-the-non-obvious-pivot)
- [Phase 7 — Tags system + edit tooling + Netlify config](#phase-7--tags-system--edit-tooling--netlify-config)
- [Phase 8 — Containerization + Oracle Cloud attempt](#phase-8--containerization--oracle-cloud-attempt)
- [Phase 9 — $0/month production stack (Fly + Supabase + Aura + Netlify)](#phase-9--0month-production-stack-fly--supabase--aura--netlify)
- [Phase 10 — Consolidation + performance + reorg](#phase-10--consolidation--performance--reorg)
- [Phase 11 — Today: history reconstruction](#phase-11--today-history-reconstruction)
- [Cross-cutting design principles that emerged](#cross-cutting-design-principles-that-emerged)
- [Repo split — what lives where](#repo-split--what-lives-where)

---

## Phase 0 — Origin: UIUC cPanel static site

**The starting point.** A static HTML/CSS/JS site hosted on UIUC's free cPanel tier.

- Single monolithic `public_html/data.json` (~124 MB) loaded on every page visit.
- Stock template assets (fancybox, zoom icons, Bloom's-taxonomy PNGs under `img/`).
- Two main pages: `graph.html` (force-directed course graph) and `visualize.html` (similar-topic explorer).
- Per-course JSONs in `public_html/data/courses/*.json` (CHEM_102.json, CS_101.json, ECE_205.json, ME_*.json, MATH_*.json, PHYS_*.json, TAM_*.json, …).
- No backend, no database. The browser held everything.
- Hard cold-load (~5 s) on every visit; impossible to update without re-uploading the whole bundle.

That's the *before* picture in `docs/EARLIER_DEPLOY.md`.

---

## Phase 1 — JSON → AWS migration

**Date:** Apr 20, 2026 (~13:48–15:43)
**Repo:** `~/Desktop/Projects/UIUC Research/SIIP/`
**Commits:** `2d8af16` "Initial commit — SIIP AWS migration", `28682ed` "Remove large data files — data lives in DynamoDB"

This was the first thing we did together — "port from JSON to a database. Use AWS. Keep it simple."

### Architectural decisions
- **DynamoDB** for topic and connection storage (pay-per-request, free at this scale).
- **AWS Lambda** + **API Gateway HTTP API** for the read API.
- **S3 static hosting** for the frontend.
- Hard constraint set early: ***the UI must stay identical to the cPanel version.*** Only the data path was allowed to change.
- `.env`-driven configuration; setup instructions in `.md` files only.

### Deploy automation
`aws/deploy.py` (491 lines) — single script that creates the DynamoDB table
`siip-topics`, loads all 1,642 topics (~5–10 min), creates an IAM role for
Lambda, creates the Lambda function, creates the API Gateway, saves the API
URL back into `aws/.env`, creates the S3 bucket, uploads all static files,
and prints the final website URL.

Re-deploy modes:
- `--upload-site` for frontend-only changes
- `--infra-only` for `lambda_handler.py` changes
- `--load-data-only` to reload DynamoDB

### Files added
- `aws/deploy.py` — orchestrator
- `aws/dev_server.py` (167 lines) — local server that mirrors prod behavior
- `aws/lambda_handler.py` (73 lines) — read API
- `aws/lambda_upload.py` (326 lines) — upload endpoint
- `DEPLOY.md` (129 lines, later moved to `docs/EARLIER_DEPLOY.md`)

### IAM and credentials drama
Mid-deploy hit `AccessDeniedException`. Resolved by attaching to user `akhilgogineni19`:
- `AmazonDynamoDBFullAccess`
- `AWSLambda_FullAccess`
- `AmazonAPIGatewayAdministrator`
- `AmazonS3FullAccess`
- `IAMFullAccess`

### First post-deploy bug
Site loaded but asked for login + showed "rate courses" UI — not what we wanted. Demanded a fix back to the "explore similar topics" UI.

### The lazy-load redesign
Cold load was still ~5 seconds because the full topic catalog was being fetched. We pivoted to a lazy-load design:
- **Topic names** load upfront (cheap list query)
- **Connections** are fetched per-click against DynamoDB
- UI behaves identically to before, but startup is instant

### Cost
All within AWS free tier:
| Service | Usage | Cost |
|---|---|---|
| DynamoDB | ~115 MB storage, on-demand reads | ~$0/month |
| Lambda | few hundred calls/month | ~$0/month |
| API Gateway | few hundred HTTP calls/month | ~$0/month |
| S3 | ~17 MB static files | ~$0/month |

---

## Phase 2 — Upload pipeline + AI topic extraction

**Date:** Apr 20, 2026 (16:21–18:33)
**Repo:** Still `SIIP/` (AWS-backed)

The first real *feature addition* on top of the migration: users (read: you) can upload a syllabus and the system extracts topics automatically.

### New capability
- Drag-and-drop syllabus upload (PDF / TXT).
- Optional course blurb field.
- **Append-or-create behavior**: upload "CS 225" syllabus → either create CS 225 or merge into an existing CS 225.

### Bugs along the way
1. `Could not parse PDF: No module named 'pypdf'` — fixed by adding `pypdf` to the Lambda layer.
2. `Topic extraction failed: HTTP Error 400` — Claude API problem.

### Anthropic → Gemini switch
The Claude API rejection drove a key decision:
- **Add Gemini as an alternative provider**, switchable via `.env` `AI_PROVIDER`.
- First model tried: `gemini-3.1-flash-lite-preview`.
- First successful upload: 8 topics extracted.

This is the *first* point where the LLM provider became swappable — a pattern that survives to today (Gemini key is now provided by the user in the browser).

### A foreshadowing concern
At 18:25 you raised the question: *"Are we asking Gemini to score every topic pair? That's computationally exhausting."* This is the seed of the design principle that later crystallized as **math scores, LLM explains — never the reverse**.

---

## Phase 3 — AWS production deploy + pivot announcement

**Date:** Apr 20, 2026 (18:33–19:29)

- Terminated localhost; pushed AWS production.
- Pushed SIIP repo to GitHub.
- *Announcement*: "Afterward, something completely novel — I'll talk about it next."
- Posted a 118-line plan describing the next system; asked Claude to poke holes in it.
- Codename for the new system: **`akhil_app`**.

The plan called out things the AWS DynamoDB stack fundamentally couldn't do:
semantic embeddings, vector similarity, graph queries with community detection.
That's why the next phase moved off AWS entirely.

---

## Phase 4 — New repo: akhil_app with Postgres + Milvus + Neo4j

**Date:** Apr 20–21, 2026
**Repo:** `~/Desktop/Projects/UIUC Research/akhil_app/` (this repo)
**Commit:** `12a8fc1` "Full implementation — IR + vector hybrid similarity pipeline"

Explicit instruction: ***"Make `akhil_app` its own folder + its own GitHub repo, don't intersect with SIIP."***

### Why a new stack
AWS DynamoDB couldn't store 768-dim embeddings or run graph algorithms. The novel idea required three specialized stores.

### Tech stack (local Docker)
| Service | Role | Port |
|---|---|---|
| **PostgreSQL** | Course metadata, chunks, term counts | 5433 |
| **Milvus** (+ etcd + minio) | 768-dim vector store | 19530 |
| **Neo4j** (+ GDS plugin) | Course graph with Louvain communities + PageRank | 7687 |
| **FastAPI** | REST API | 8080 |

All orchestrated by `docker-compose.yml` (108 lines).

### Pipeline modules
- `pipeline/encoder.py` — SciNCL (`malteos/scincl`) sentence transformer, 768-dim L2-normalized.
- `pipeline/keyphrase.py` — KeyBERT for keyphrase extraction.
- `pipeline/chunker.py` — Topic-level chunking from course JSON.

### Scoring modules (initial)
- `scoring/language_model.py` — Dirichlet-smoothed unigram LMs (`μ = 2000`).
- `scoring/jsd.py` — Jensen–Shannon divergence for lexical similarity.
- `scoring/vector_similarity.py` — Milvus ANN-based semantic similarity.
- `scoring/driving_terms.py` — `min(P_A, P_B) · IDF` over shared vocab; top-5 per pair.
- `scoring/hybrid_scorer.py` — `0.4 · lex + 0.6 · sem`.

### Storage
- `storage/postgres_store.py` — courses, chunks, term_counts, similarity cache.
- `storage/milvus_store.py` — 768-dim FLAT index, COSINE metric.
- `storage/neo4j_store.py` — `(c:Course)`-`[:SIMILAR]`-`(c:Course)` model.

### API routes
- `GET /courses`, `GET /courses/{id}`
- `GET /similarity`
- `GET /graph`
- `POST /ingest/pdf`

### Scripts
- `scripts/seed.py` — load 33 courses from `data/mechse_syllabi.json`.
- `scripts/build_graph.py` — score every N(N-1)/2 pair, write Neo4j edges, run Louvain + PageRank.
- `scripts/reset.py` — destructive wipe.

### Bugs surfaced during bring-up
- Postgres auth error: `password authentication failed for user "siip"` → fixed via `.env`.
- D3 graph: **all nodes overlapping on top of each other.** Asked for it to look like the original SIIP graph.
- Quality flag: **CHEM 102 ↔ CS 101 scored similar** but obviously aren't. Eliot's category-divergence idea is the eventual fix.
- "Too many routes; feels like two websites (graph vs. explorer)" → triggers Phase 5.
- Asked where the **0.5 score threshold** for forming Neo4j edges was defined.

### Concept asks during this phase
- "I should be able to see the scoring of similarity and the categorization of the topics. Like a percentage on how one thing relates to chemistry (60%), physics 20%, gym (2%) etc." — seed of the 8-bin category distribution.
- "I had 2200 requests... that's gonna cost a lot" — first real cost anxiety with Gemini.
- "We need to have LLM usage... LLM is needed for outlining a neat and clear explanation on how a topic overlaps with two courses." — confirms LLM-as-narrator role.
- "Later on, users implement their own LLM API key." — seed of the browser-localStorage API key flow.

---

## Phase 5 — Unified UI + scoring overhaul

**Date:** Apr 21, 2026
**Commit:** `47fff11` "Unified UI, scoring overhaul, and project guide"

### UI consolidation
- **Merged `graph.html` + explorer into a single `public/index.html`** with tab switching.
- Dark theme throughout.
- Deleted `visualize.html`.
- New `public/upload.html` (530 lines) with PDF drag-drop + status polling.

### D3 graph polish
- Normalized PageRank for node sizing (5–21 px).
- Collision radius matches node radius → resolves the "all nodes overlapping" bug.

### Scoring math overhaul
- **Replaced Dirichlet-smoothed LM lexical scoring with TF-IDF cosine.** Reason: corpus-prior contamination across short documents was distorting lex scores. TF-IDF cosine is sparse by design — only shared terms contribute.
- **Calibrated SciNCL semantic scores** above the ~0.75 academic-text floor.
- Reduced `DIRICHLET_MU` 2000 → 200 (still kept LM/JSD as reference).
- Added `_DOMAIN_STOP` to chunker (drop common engineering stopwords).
- **Separated `raw_text` (embedded) from `term_text` (LM counts)** in chunker — embeddings shouldn't see boilerplate, term counts shouldn't see [SEP] markers.

### Docs
- Added `GUIDE.md` (273 lines) — the first full architectural reference.

---

## Phase 6 — Topics, categories, and the non-obvious pivot

**Date:** May 10–12, 2026
**Sessions:** `1d25f95a`, `a8d8abd8`, `7b03390e`
**Commit:** `66ad022` "Topics and LLM Augmentation"

This is the **biggest scientific pivot in the project**, driven by an email
from your supervisor Eliot Bethke: *"computing the semantic similarity on
topic:definition strings but weighting the results by the dissimilarity of
the predicted category composition is something I would love to try."*

### The new headline formula
```
non_obvious_score = sem_score × category_jsd
```
- High `sem_score` + high `category_jsd` (different categorical bins) = jackpot cross-domain match.
- High `sem_score` + low `category_jsd` (same department) = uninteresting same-department pair.

The `final_score = 0.4·lex + 0.6·sem` hybrid stays for the conventional ranking; `non_obvious_score` becomes the headline signal.

### The 8-category schema
LLM classifies each topic across 8 bins (sums to 1.0):
**Mechanics | Thermodynamics | Electrical | Fluids | Materials | Mathematics | Chemistry | Systems**

### New pipeline modules
- `pipeline/category_labeler.py` — Gemini batch labeling.
- `pipeline/topic_extractor.py` — single-call comprehensive analyze (description + objectives + topics + categories).
- `pipeline/llm_explainer.py` — on-demand explanation of a connection.
- `scoring/category_scorer.py` — JSD over the 8-bin distributions; `non_obvious_score` formula.

### New scripts
- `scripts/build_topic_graph.py` — topic-to-topic ANN; stores pairs with raw cosine ≥ 0.70 in `topic_similarity`.
- `scripts/label_categories.py` — backfill categories for existing courses.
- `scripts/backfill_embeddings.py` — find topics missing vectors and encode them.
- `scripts/explain_connections.py` — pre-generate explanations for top non-obvious pairs.

### New API surface
- `GET /topics/search?q=...` — keyword search over topic_categories.
- `GET /similarity/non-obvious?top=20` — ranked by `non_obvious_score`.
- `GET /similarity/categories?course=...` — 8-bin distribution.
- `GET /similarity/explain?a=...&b=...` — fetch (or generate) explanation.
- `GET /courses/{id}/topics` — topics with category distributions.

### LLM design principles that emerged this phase (your asks, in order)
1. *"The LLM simply outputs the initial data, percentages, etc. The backend should take that in and leverage the stacks so that the new course is just like any other course."* → **One LLM call per ingest**, the rest is deterministic math.
2. *"THE LLM MUST NOT DO ANYTHING ELSE BESIDES [extract initial data]."* → Reinforced.
3. *"There should be an LLM generation button for any explanation that hasn't been generated already."* → On-demand explanations only; pre-generation is wasteful.
4. *"The user should use his own API key to make the generation happen."* → API key flows from browser, never stored server-side.
5. *"There should be no fallback. If the gemini api key does not work, simply give an error message notifying me what the issue with my key is."* → No silent fallbacks anywhere.
6. *"Are we instigating multiple calls for one upload? I'd like it to be one giant request."* → Consolidated to **one giant prompt** per ingest covering description + objectives + topics + categories.
7. *"Don't be biased by domain... ensure scoring for courses and topics are done the same for all courses."* → No domain-specific calibration in scoring.

### UI changes during this phase
- Course click opens the full card on the **background container**, not the lightbulb icon.
- All sections (Topics, Tags, Description) become **collapsible dropdowns**.
- "Description" styled consistently with Topics/Tags for visual coherence.
- Matrix view: "LLM generate" button on empty cells.
- Better colors for the cards.
- Topics view: clickable course → drop-down with topics ranked by similarity.

### Specific user-flagged bugs this phase
- *"The topic tab is fucked... all I see is a big rainbow shaped figure on the screen"* → SVG was being stretched globally; fix scoped `svg { width: 100%; height: 100%; }` to `#graph-wrap svg` only.
- *"CS 521 does not have the 'why not obvious' button. why?"* → Not in global top-50 non-obvious; fix was an on-demand `/similarity/explain` endpoint for any pair.
- *"The core concept of iterative approximation [as the connection between] blockchain and math. It is ridiculous."* → Improved Gemini prompts; required concise + outlined response.
- *"Mechanics 12.5% Thermo 12.5% ... Why is the distribution entirely equal for this course?"* → Hidden default-fallback path was bypassing the LLM; removed.

### Bonus this phase
- Added `CLAUDE.md` (Claude-specific project notes).
- Added `Comprehensive Gameplan.md` and `How It Is.md` (both later removed in Phase 10 reorg).
- Added `ideas.md` for parked ideas:
  - LLM-feedback-based score refinement (downweight false positives identified by humans).
  - Topic-to-topic LLM explanations (phase 2 of topic similarity).

### Bonus: Docker revive
Mid-session you had to prune all Docker containers because Linux was out of disk. Asked to revive Milvus → restored. Also did a system-wide disk audit and freed everything except trash.

---

## Phase 7 — Tags system + edit tooling + Netlify config

**Date:** May 22, 2026
**Commit:** `413ded8` "Configure frontend for Netlify deployment and update API URLs"

### The tagging system
Recognized that no topic in the DB had any tags. Tags are *cross-domain buzz words*: e.g., "second-order-ode" should appear on both ME 340's spring-mass-damper and ECE 210's RLC circuit.

- Manual tag corpus authored across **35 courses / 938 topics** in `scripts/apply_tags.py` (1080 lines).
- `scripts/generate_tags.py` — Gemini-driven tag fill (used to bootstrap then hand-curated).
- Tags embedded in the DB so they participate in TF-IDF.

### Edit tooling — DB ↔ JSON round trip
The motivating ask: *"I'm hoping there's an easier way to change up the data manually, like in the json files... Maybe there's a function that could serialize all the data pertaining to a course into a json. We edit the json and then deserialize it back."*

So we built:
- `scripts/dump_courses.py` (213 lines) — full corpus dump → `data/courses_dump.json` (16,723 lines).
- `scripts/load_courses.py` (121 lines) — restore from dump.
- `scripts/edit_course.py` (226 lines) — single-course round-trip via `data/edit/<COURSE>.json` (e.g., `data/edit/ME_340.json`).
- `scripts/export_course_tags.py` + `scripts/import_course_tags.py` — tag-only round trip.

Workflow: `python scripts/edit_course.py dump ME_340` → edit JSON → `python scripts/edit_course.py load ME_340`.

### LLM explainer improvements
- Better prompts (243 lines of changes in `llm_explainer.py`).
- Output now structured as three labeled one-liners: **Shared math** / **Why surprising** / **Analogy** — replacing the verbose paragraphs the user complained about.

### Frontend → Netlify
- `netlify.toml` added.
- `public/index.html` updated to point at an external API base URL (configurable for deployment).
- Frontend ready for Netlify CDN hosting.

---

## Phase 8 — Containerization + Oracle Cloud attempt

**Date:** May 22–24, 2026
**Commits:** `4cb15e5`, `50973f6`, `5000df7`

### Containerize the backend
- `Dockerfile` (25 lines) — Python backend image.
- Extended `docker-compose.yml` with backend service (was previously just databases).

### Oracle Cloud Free Tier deploy attempt
The first attempt at a real cloud deployment beyond AWS read-only:
- `deploy/ORACLE.md` (307 lines, later expanded by 55 more) — full self-host guide.
- `deploy/oracle-install.sh` (49 lines) — one-liner installer (`curl ... | bash`).
- `deploy/bootstrap.sh` (160 lines) — VPS setup: Docker, UFW firewall, seed data, optional Caddy for TLS.
- `docker-compose.prod.yml` (29 lines) — production overrides.
- Frontend API URL switched to `window.location.origin` in production.

### Why it was expanded
Hit Oracle Cloud-specific issues:
- **AD (availability domain) capacity errors** — Free Tier VMs not always available.
- **Public IP issues** — needed explicit reserved IP attach steps.

Guide rewritten with troubleshooting blocks for both.

This Oracle path ultimately got quarantined to `deploy/legacy/` in Phase 10 — it's a *working* alternative, not the primary path.

---

## Phase 9 — $0/month production stack (Fly + Supabase + Aura + Netlify)

**Date:** May 24, 2026
**Commit:** `2f1ffcf` "Ship $0/month production stack + UI revamp + CI/CD"
**Session:** `c2298a7c`

The biggest single commit in the project — 2,615 insertions across 37 files.

### Stack pivot
| Before (local Docker / Oracle) | After (free-tier cloud) |
|---|---|
| Milvus + etcd + minio | **Supabase Postgres + pgvector** |
| Local Postgres | **Supabase Postgres** (same instance) |
| Local Neo4j + GDS | **Neo4j Aura Free** |
| Local Docker / Oracle | **Fly.io** (shared-cpu-1x, 2 GB, always-on) |
| - | **Netlify** CDN for static UI |
| ~$5/mo Oracle VM | **$0/mo** under Fly's $5/mo hobby credit |

### pgvector replacing Milvus
- `storage/pgvector_store.py` (177 lines) — pgvector implementation of the vector store interface.
- `storage/vector_store.py` (52 lines) — **dispatcher** selecting `pgvector` (prod) or `milvus` (local) based on `VECTOR_BACKEND` env var.
- This is a clean abstraction — code calls `vector_store.search(...)`, backend is swapped at boot.

### Fly.io backend
- `Dockerfile.api` (53 lines) — production image with full ML deps baked in: SciNCL model weights, CPU PyTorch, pdfplumber, all maintenance scripts.
- `fly.toml` (42 lines) — Fly app config (region `ord`, 2 GB, always-on).
- `requirements-api.txt` (30 lines) — production deps separate from dev.

### Bootstrapping scripts
- `deploy/free/00-install-flyctl.sh`
- `deploy/free/01-verify-services.sh` — health-check Supabase + Aura
- `deploy/free/02-deploy-fly.sh`
- `deploy/free/03-deploy-netlify.sh`
- `deploy/free/04-bootstrap-data.sh`
- `deploy/free/deploy.sh` — orchestrator
- `deploy/free/credentials.env.example` — env template
- `deploy/free/README.md` (200 lines) — end-to-end guide
- `deploy/free/GITHUB_ACTIONS.md` — CI/CD docs

### CI/CD
- `.github/workflows/deploy-fly.yml` — push to main rebuilds the Fly image, deploys, smoke-tests `/health`.
- `.github/workflows/deploy-netlify.yml` — push to main rebuilds frontend config (pointing at Fly), pushes to Netlify prod.
- 4 GitHub secrets required: `FLY_API_TOKEN`, `NETLIFY_AUTH_TOKEN`, `NETLIFY_SITE_ID`, `SIIP_API_URL`.

### Big UI revamp (for non-researcher users)
Goal: make it amenable to non-researchers (e.g., Eliot's team) who shouldn't have to read score math to use it.

- **Design tokens** (colors / spacing / radius / shadows / type) in CSS variables.
- **Light theme** opt-in via `data-theme="light"` + one-click toggle.
- **Welcome modal** on first visit with three entry points: *See the map* / *Pick a course* / *Search a topic*. Replayable via help icon.
- **Plain-language relabeling everywhere**:
  - Tabs renamed to "Map / Courses / Search Topics".
  - "Min similarity" → "Show only stronger connections" with help bubble.
  - "Non-obvious" → "Surprises".
  - Sort toggle: "Hybrid / Non-obvious" → "Most similar / Most surprising".
  - Sidebar: drops "Community" + "PageRank" decimals for "Group" + qualitative centrality ("Highly central", "Central", "Connected", "Peripheral").
- **Strength pills** ("Very strong / Strong / Moderate / Light") replace raw decimals in similar-courses lists. Raw scores still in tooltips.
- **Pathfinder collapsed by default** behind a "Find a path" toggle. From/To inputs autocomplete from `/courses`.
- **Topics view**: clickable suggestion chips (vibration, Laplace, …) + friendlier empty-state copy.
- **Upload page**: friendlier copy throughout, jargon replaced with what-happens-next plain English.
- **Graph aesthetics**:
  - Per-community node glow via `currentColor` + drop-shadow.
  - Edges color-shift with score (slate → indigo → cyan; in Surprises mode, slate → orange).
  - Frosted-glass legend + tooltip.
  - Subtle radial vignette canvas background.

### Big performance fix
`scoring/vector_similarity.py` rewritten: previously did per-chunk ANN call against remote DB (network-bound). New version **fetches all course embeddings once and computes cosine similarities in-memory with NumPy**. Graph build time: **~2 hours → ~2 minutes** against remote DB.

### Robustness
- `config.py` parses `DATABASE_URL` into individual Postgres fields; default SSL mode `require` for managed services.
- `api/routes/ingest.py` gracefully returns HTTP 503 instead of crashing at import time when optional ML deps are missing (so a slim API-only re-deploy still works for read endpoints).
- Old Oracle scripts moved into `deploy/oci/` (later → `deploy/legacy/oci/` in Phase 10).

---

## Phase 10 — Consolidation + performance + reorg

**Date:** May 26–27, 2026
**Session:** `c2298a7c`, `f949fc3f`

A cleanup sweep. Less new feature work, more "kill cruft, fix the slow stuff, make the file layout sensible".

### Commit `cc5dd68` — Consolidate UI + API on Fly, drop Netlify
- Netlify served the static UI; Fly served the API. Two deploys, two URLs, CORS hassle.
- New approach: **Fly serves both** — `Dockerfile.api` now bundles the UI; FastAPI mounts it at `/`.
- Deleted: `netlify.toml`, `deploy/free/03-deploy-netlify.sh`, `.github/workflows/deploy-netlify.yml`.
- `deploy/free/credentials.env.example` and `GITHUB_ACTIONS.md` simplified (4 secrets → 1).

### Commit `0f10aaa` — Speed up `/topics/search` 30s → <1s
Search was doing keyword filtering in Python after pulling all `topic_categories`. Pushed filtering into Postgres:
- New SQL with `ILIKE` on topic_text + JSONB category filtering.
- LRU cache layer in `api/routes/topics.py` (cleared on ingest).
- ~30× speedup; 30 s → <1 s.

### Commit `488f0d2` — Fix gray graph nodes on Aura
- **Symptom**: All graph nodes rendered gray (no community color).
- **Root cause**: Aura Free doesn't include the GDS (Graph Data Science) plugin. Louvain query was failing silently.
- **Fix**: `storage/neo4j_store.py` tries `gds.*` first, falls back to **NetworkX Louvain** computed in Python and written back as a `community` property. Same for PageRank.
- Added `networkx` + `python-louvain` to `requirements-api.txt`.

### Commit `6e480f9` — Unify local dev on shared cloud DBs
- **Old workflow**: local Postgres + Milvus + Neo4j in Docker, separate from prod.
- **New workflow**: laptop talks to the same Supabase + Aura that prod does.
- `deploy/free/link-local-env.sh` (78 lines) pulls Fly's env vars into your local `.env`.
- `.env.example` rewritten around cloud-first config.
- CI now also runs full deploys on push to main (`.github/workflows/deploy-fly.yml`).

### Commit `93a1d2c` — Full-stack ingest
PDF upload was populating only Postgres + vectors. Missing: topic graph + Neo4j.
- `api/routes/ingest.py` now drives the *full* pipeline end-to-end:
  1. Topic extraction (one Gemini call)
  2. SciNCL encoding → pgvector
  3. Postgres chunks + categories
  4. Pairwise scoring against all existing courses
  5. **Topic-to-topic ANN backfill** (`build_for_courses`)
  6. **Neo4j upsert** + edge refresh + Louvain + PageRank
- `storage/neo4j_store.py` gained the upsert helpers.

### Commit `77a9b0b` — CI hardening
- `.github/workflows/deploy-fly.yml` now verifies `FLY_API_TOKEN` exists before running.
- `deploy/free/setup-github-actions.sh` (49 lines) — one-shot script that registers all required GitHub secrets via `gh` CLI.

### Commit `eb3d5fb` — Reorganize file directory
Triggered by your ask: *"All stuff pertaining to math and scoring in one folder. All UI in another. All postgres in a folder. All milvus in a folder. Same with neo4j. Same with LLM. There should be a docs folder. There should be a place to store sample files."*

| Before | After |
|---|---|
| `pipeline/` (mix of LLM + chunker + encoder) | Split: `llm/`, `scoring/chunker.py`, `storage/vectors/encoder.py` |
| `storage/postgres_store.py` | `storage/postgres/store.py` |
| `storage/neo4j_store.py` | `storage/neo4j/store.py` |
| `storage/milvus_store.py` + `pgvector_store.py` + `vector_store.py` | `storage/vectors/` (all three + encoder) |
| `public/` | `ui/` |
| `Dockerfile`, `docker-compose.yml` | `deploy/legacy/Dockerfile`, `deploy/legacy/docker-compose.yml` |
| `deploy/ORACLE.md`, `oci/`, `bootstrap.sh`, `oracle-install.sh` | `deploy/legacy/` (quarantined but kept) |
| `docs/CS 568 Syllabus.txt`, `cs521 Syllabus.pdf`, `image.png` | `samples/` |
| Loose `PLAN.md`, `GUIDE.md`, `How It Is.md`, `Comprehensive Gameplan.md`, `ideas.md`, `instructions.md`, `docs/CorpusConvertInstructions.md`, `docs/ProcessPipeline.md` | **Deleted** — replaced by single `docs/ARCHITECTURE.md` (565 lines) |
| (no README) | `README.md` (142 lines) — entry point + doc index |
| `docs/EARLIER_DEPLOY.md` | (still doesn't exist yet — created Phase 11) |

Also removed: `pipeline/keyphrase.py` (KeyBERT no longer used).

After this commit, the directory tree is:
```
akhil_app/
├── README.md, CLAUDE.md
├── config.py
├── Dockerfile.api, fly.toml
├── requirements.txt, requirements-api.txt
├── api/{main.py, routes/}
├── scoring/{chunker, language_model, jsd, vector_similarity, driving_terms, category_scorer, hybrid_scorer}.py
├── llm/{topic_extractor, category_labeler, llm_explainer}.py
├── storage/
│   ├── postgres/store.py
│   ├── neo4j/store.py
│   └── vectors/{encoder, store, pgvector_store, milvus_store}.py
├── ui/{index.html, upload.html}
├── scripts/   (all maintenance + edit tooling)
├── data/      (source JSON + edit buffers)
├── samples/   (test fixtures for /upload)
├── docs/ARCHITECTURE.md
├── deploy/
│   ├── free/   (Supabase + Aura + Fly, production)
│   └── legacy/ (local Docker + Oracle, quarantined)
└── .model_cache/  (SciNCL weights, gitignored)
```

---

## Phase 11 — Today: history reconstruction

**Date:** May 27, 2026
**Session:** `f949fc3f`

You asked: *"Go through all past sessions. Identify all the changes we made from the beginning. I know we migrated from JSON to AWS first."*

We discovered:
- The AWS migration didn't happen in this repo — it happened in the *sibling* `~/Desktop/Projects/UIUC Research/SIIP/` repo.
- Model-side session transcripts for the SIIP work were **lost** (not on disk). Only the user-side prompt log (`~/.claude/history.jsonl`) survived for that phase.
- Created `docs/EARLIER_DEPLOY.md` — the original AWS deploy guide moved into docs.
- Created `docs/EARLIER_SESSION_HISTORY.md` — prompt-by-prompt reconstruction of the SIIP era from `history.jsonl`.
- Created **this file** (`docs/HISTORY.md`) — the full consolidated record.

---

## Phase 12 — Outage, single-Postgres rewrite, ONNX-ready

**Date:** June 3, 2026

**The trigger.** The live site stalled on "Loading knowledge graph…" with no
error in the UI. Investigation:

- `/health` was fine, but `/courses` and `/graph/all` returned 500.
- Both Supabase (`db.meaxjrfsexutfwtleoju.supabase.co`) and Neo4j Aura
  (`3e99620a.databases.neo4j.io`) resolved to **NXDOMAIN**. Public DNS no
  longer knew them.
- The README's own caveat (`deploy/free/README.md:97`) had warned about Aura
  Free auto-pausing after 3 days idle. The instances had drifted past pause
  into reclamation. There was no heartbeat to prevent it.

**The fortunate find.** `data/courses_dump.json` (committed weeks earlier
through `scripts/dump_courses.py`) contained the complete LLM-enriched
snapshot: 35 courses, 938 topics, every category distribution, every
cross-domain tag. The only data not preserved anywhere local was the cache of
on-demand LLM connection explanations — which were always lazy-generated.

**What we changed.**

1. **Neo4j removed entirely.** `storage/neo4j/` deleted; `community` and
   `pagerank` are now columns on the `courses` table. `scoring/graph_compute.py`
   runs the same NetworkX Louvain + PageRank that previously served as the
   Aura-Free fallback (`_run_community_detection_networkx`). At 35 nodes /
   ~123 edges this is sub-second and identical numerically. `api/routes/graph.py`
   reads from `similarity_cache` + NetworkX in-process. All NEO4J_* secrets,
   env vars, and shell-script references were stripped from `fly.toml`,
   `config.py`, `.env.example`, `deploy/free/*.sh`, `requirements*.txt`.

2. **`config.yml` as single source of truth.** Every non-secret knob —
   scoring weights (ALPHA_LEX/SEM/CAT), `min_score`, `sem_floor`, embedding
   model id and runtime, ANN parameters (HNSW m + ef_search), top-K values —
   moved into one YAML. `config.py` is now a thin loader with precedence
   `env > yaml > default`, preserving the legacy constant names so call sites
   didn't change.

3. **Pgvector HNSW + model versioning.** `chunk_embeddings` is now HNSW-indexed
   (cosine, `m=16`, `ef_search=64`) and every row carries a `model_version`
   column stamped with `config.EMBEDDING_MODEL_ID`. A mixed-runtime dataset
   (e.g., PyTorch FP32 rows next to ONNX INT8 rows) is now detectable in SQL.

4. **ONNX-INT8 serving (opt-in).** `scripts/export_scincl_onnx.py` exports
   SciNCL through `torch.onnx.export`, dynamic-quantises with
   `onnxruntime.quantization`, and runs a parity check against the FP32
   SentenceTransformer baseline on a 6-text golden set. The manifest
   (`models/MANIFEST.json`) records id, dim, sha256, size, and the worst-case
   cosine — and is only written if every golden cosine ≥ 0.998.
   `storage/vectors/encoder.py` dispatches between PyTorch and ONNX runtime
   based on `config.yml`. `Dockerfile.api` accepts
   `--build-arg EMBEDDING_RUNTIME=onnx` to drop PyTorch + sentence-transformers
   from the image. Until the export is run and committed, PyTorch stays the
   default — no behavioural change on the existing image.

5. **`query_log` table + middleware.** Every API request gets one row:
   endpoint, method, params (JSONB), status, latency_ms, hashed UA. The
   middleware in `api/main.py` is best-effort — a logging failure never causes
   a request failure. Becomes free offline-evaluation data.

6. **`scripts/restore_from_dump.py`.** Reads `data/courses_dump.json`,
   rebuilds: courses, chunks (via the existing chunker), term_counts,
   topic_categories (with tags), SciNCL embeddings (local — no Gemini),
   similarity_cache (pure math), topic_similarity (pure ANN), and finally
   `graph_compute.recompute_graph_stats()`. End-to-end with **zero Gemini
   calls**. Bringing a fresh Supabase project from empty to fully-populated
   is now one command.

7. **Heartbeat workflow.** `.github/workflows/heartbeat.yml` curls
   `/health` weekly so Supabase never crosses its inactivity threshold again.

8. **Docs.** `docs/ARCHITECTURE.md` rewritten for the new stack (added § 13
   "what changed and why"). `README.md` updated. This entry added.

**Output parity.** For any given `similarity_cache` (which is the input the
Neo4j path used too), the new in-process graph produces bit-identical
community IDs and PageRank values to the old NetworkX fallback. The UI
expects the same JSON shapes from `/graph/all`, `/graph/path`,
`/graph/neighbors`, `/graph/communities`. No client-side changes needed.

**What this leaves.** The Phase B follow-up — running the ONNX export on a
dev box, committing the bundle, flipping `config.yml`, rebuilding with
`--build-arg EMBEDDING_RUNTIME=onnx`. Drops the production image from
~1.5 GB to ~150 MB and removes PyTorch from the serving path entirely.

---

## Phase 13 — One pipeline; archive consolidation

**Date:** June 3, 2026

After Phase 12, the file tree still carried two parallel framings: the
cloud-backed pipeline (the one we actually use) and the original local-only
docker-compose pipeline (Postgres + Milvus + Neo4j containers, Oracle
self-host). Two pipelines doubled the surface area of every config and
muddied "which file is current?" judgements.

**The rule, stated explicitly:** there is *one* pipeline — cloud-backed
Supabase + Fly. Local development is `uvicorn` against the same Supabase the
live site uses. There is no parallel local-only flow.

**What moved into `archive/`:**

- `deploy/legacy/` → `archive/legacy-local-docker/` (entire folder, including
  `docker-compose.yml`, `docker-compose.prod.yml`, `Dockerfile`, `bootstrap.sh`,
  `oracle-install.sh`, `ORACLE.md`, and the `oci/` Oracle CLI scripts).
- `storage/vectors/milvus_store.py` → `archive/legacy-local-docker/milvus_store.py`.
- `archive/README.md` added — names each archived item and what would be
  required to revive it.

**What got simplified as a consequence:**

- `storage/vectors/store.py` was a pgvector-vs-milvus dispatcher; collapsed
  to a thin re-export of `pgvector_store`. Call sites (`from storage.vectors
  import store as vs`) still work.
- `VECTOR_BACKEND`, `MILVUS_HOST`, `MILVUS_PORT`, and `TOP_K_MILVUS` removed
  from `config.py` and `config.yml`. `TOP_K_MILVUS` was renamed to `TOP_K_ANN`
  (which is what it always meant) and kept.
- `VECTOR_BACKEND=pgvector` removed from `fly.toml`, `Dockerfile.api`,
  `.env.example`, and `deploy/free/link-local-env.sh` — pgvector is the
  implicit-and-only backend now.
- `pymilvus` and `marshmallow` dropped from `requirements.txt`.
- Misleading `from storage.vectors import store as milvus` aliases in
  `scripts/seed.py`, `scripts/build_topic_graph.py`, and
  `scripts/backfill_embeddings.py` renamed to `vs`.

**Docs:**

- README's "Optional: fully isolated local stack (legacy)" section replaced
  with a one-paragraph "One pipeline, not two" callout pointing at
  `archive/`.
- `deploy/free/README.md` had the same parallel section; same treatment.
- `docs/ARCHITECTURE.md` file layout block updated.
- `CLAUDE.md` updated.

No behavioural change to the live site — only code organisation. Anyone who
needs the old local docker stack can `cp -r archive/legacy-local-docker/
./old-stack/` and run from there, with the caveat (called out in
`archive/README.md`) that reviving it as-is would require reinstating the
Neo4j call sites removed in Phase 12.

---

## Phase 14 — Flatten deploy/

**Date:** June 3, 2026

`deploy/free/` had 9 files with cryptic numbered prefixes (`00-install-flyctl.sh`,
`01-verify-services.sh`, `02-deploy-fly.sh`, `04-bootstrap-data.sh`, plus
`deploy.sh`, `link-local-env.sh`, `setup-github-actions.sh`, a README, and
`credentials.env.example`). Hard to scan; the prefixes implied a strict order
that wasn't true; `04-bootstrap-data.sh` was made redundant by
`scripts/restore_from_dump.py` (which works without Gemini and bakes from
`data/courses_dump.json` in one pass).

**New shape (flat, single-purpose, descriptive names):**

```
deploy/
├── README.md                 one-page guide + troubleshooting
├── credentials.env.example
├── install-flyctl.sh         one-time
├── link-env.sh               writes /.env for local dev
├── deploy.sh                 verify Supabase + push secrets + fly deploy + smoke-test
└── setup-ci.sh               one-time GitHub Actions FLY_API_TOKEN
```

The old `01-verify-services.sh` and `02-deploy-fly.sh` are inlined into
`deploy.sh`. `04-bootstrap-data.sh` is gone — `flyctl ssh console -C 'python
scripts/restore_from_dump.py'` replaces it.

`.gitignore` cleaned: stale `deploy/legacy/*` and `deploy/free/credentials.env`
entries removed; `deploy/credentials.env` and `models/` (the ONNX bundle)
added.

All cross-references updated: project `README.md`, `.env.example`, `CLAUDE.md`,
`.github/workflows/deploy-fly.yml`. No code-path changes — just file
organization.

---

## Phase 15 — Aggressive prune of deploy/ and scripts/

**Date:** June 3, 2026

`deploy/` went from 7 files to 3:

```
deploy/
├── credentials.env       gitignored
├── deploy.sh             one-shot prod deploy
└── link-env.sh           writes /.env for local dev
```

`scripts/` went from 15 files to 6 (the load-bearing ones):

```
scripts/
├── restore_from_dump.py     rebuild from data/courses_dump.json (zero Gemini)
├── build_graph.py           re-score all pairs + refresh graph stats
├── build_topic_graph.py     topic-to-topic ANN (imported by ingest + restore)
├── dump_courses.py          refresh the canonical courses_dump.json
├── export_scincl_onnx.py    SciNCL → ONNX INT8 + MANIFEST.json
└── reset.py                 wipe (supports --restore)
```

Archived to `archive/deploy-extras/` and `archive/scripts-extras/`: the
`README.md`, `credentials.env.example`, `install-flyctl.sh`, `setup-ci.sh`,
and the stale Neo4j credentials backup from deploy/; nine
one-time / niche scripts from scripts/ (`seed.py`, `backfill_embeddings.py`,
`label_categories.py`, `explain_connections.py`, `load_courses.py`,
`edit_course.py`, `export_course_tags.py`, `import_course_tags.py`,
`generate_tags.py`, `apply_tags.py`).

`deploy.sh` no longer auto-calls `install-flyctl.sh` (archived); it
fails with the one-liner if flyctl isn't on PATH. `reset.py` dropped the
`--reseed` flag (which referenced the archived `seed.py`); `--restore` via
`restore_from_dump.py` is the canonical recovery path.

---

## Phase 16 — Isomorphism-aware tag enrichment

**Date:** June 3, 2026 (continuation)

The Phase 12 +tags experiment showed cross-domain gaps could be widened by
suffixing topic chunks with `. Related concepts: <tag list>`. But the existing
tags (5 per topic) were too shallow to surface real engineering isomorphisms.
The canonical example: spring-mass-damper (mẍ + cẋ + kx = f(t)) is
mathematically identical to RLC circuit (LQ̈ + RQ̇ + Q/C = V(t)) up to variable
renaming. Both should share tags like `second-order-linear-ode`,
`harmonic-oscillator-equation`, `isomorphic-to-rlc-circuit`,
`mechanical-electrical-analogy` — but the original tags went only as far as
`damping-ratio`, `natural-frequency`. The bridge vocabulary was missing.

**The rewrite.** `llm/topic_extractor.py` got two things:

1. A new system prompt for `analyze_course()` (used at production ingest) that
   demands 15-30 tags per topic from five named categories: structural-form,
   canonical-equation, isomorphism-bridge, concept-level, physical-mechanism.
   The prompt names ~40 isomorphism-bridge tags verbatim (e.g.,
   `isomorphic-to-rlc-circuit`, `mass-inductance-analogy`, `ohms-law-analogy`)
   so a future `/upload` produces tags from the same vocabulary as the seed
   corpus.

2. A new `enrich_topics()` function — separate from `analyze_course()` —
   that takes an EXISTING topic list and rewrites only its tags + categories
   without re-extracting topics. This was the right operation for retrofitting
   the seed corpus: the first naive attempt (running `analyze_course()` on a
   reconstructed syllabus) caused Gemini to re-derive its own topic list,
   matching only 9/60 ME 340 topics. The enrichment-only prompt forces
   verbatim-name passthrough; 60/60 matched.

**Tooling.** `scripts/enrich_tags.py` reads `data/courses_dump.json`, fires
35 parallel Gemini calls (8 concurrent, well under the 15-RPM free-tier
limit), merges results back, dedupes, writes the dump. ~85 seconds wall-clock.
Cost: ~$0.10 at `gemini-2.5-flash-lite` pricing. Idempotent.

**Outcome — 100% coverage.**

| Tag richness | Topics | % |
|---|---|---|
| ≥12 tags (well-enriched) | 805 | 85.8% |
| 7–11 tags (partial)      | 133 | 14.2% |
| <7 tags (un-enriched)    | 0   | 0.0% |

Getting to 100% required three passes and two merger fixes:

1. **First pass (concurrency=8, 35 courses).** 62% rich. For 7 courses,
   Gemini emitted `"<topic-name>: <description>"` as the `name` field,
   ignoring the prompt's "copy verbatim" rule. Exact-name matching missed
   all topics in those courses.

2. **Second pass (7 failed courses).** Added a colon-aware matcher: the
   merger now tries progressive colon-truncation (drops trailing `: …`
   segments) until match or all colons exhausted. Handles both
   `"Foo"`-LLM-emits-`"Foo: bar"` and `"Foo: bar"`-LLM-emits-`"Foo: bar: baz"`
   identically. Got to 99.5% rich.

3. **Third pass (4 holdouts).** The remaining 4 topics were in courses
   where Gemini's output simply dropped them — non-deterministic truncation
   when generating 30+ topics in one response. Fix: targeted re-enrichment
   with just the holdouts (3-4 topics per call) so Gemini can't run out
   of output tokens before reaching them. 100% coverage achieved.

Total Gemini spend: ~$0.12 across the three passes.

**Quality lift on the topics that did enrich.** `Damping ratio and natural
frequency` went from 5 tags to 15 including the explicit bridges
`isomorphic-to-rlc-circuit` and `mechanical-electrical-analogy`. `A nonlinear
pendulum` got 18 tags including `harmonic-oscillator-equation`,
`isomorphic-to-rlc-circuit`, `phase-space-analysis`, `poincare-map`. The
canonical spring-mass ↔ RLC isomorphism is now explicitly tagged.

**Re-embed.** After enrichment, `scripts/restore_from_dump.py` ran the full
chunker → SciNCL → score → graph pipeline against the new tags. The chunker
already concatenates tags into the SciNCL input as
`<course> [SEP] <topic>: <def>. Related concepts: <tags>.` — no chunker
changes needed.

---

## Phase 17 — ONNX serving + final file consolidation

**Date:** June 3, 2026 (continuation)

After the tag enrichment landed, two outstanding items got resolved:

**1. ONNX cutover.** `scripts/export_scincl_onnx.py` ran on a dev box,
produced `models/scincl-int8.onnx` (~110 MB) + tokenizer + `MANIFEST.json`
with parity-verified worst-case cosine ≥ 0.998 against the FP32 baseline.
`config.yml` flipped to `model.embedding.runtime: onnx`, `onnx_path:
models/scincl-int8.onnx`. `Dockerfile.api` now builds with
`--build-arg EMBEDDING_RUNTIME=onnx`, which:
- Drops `torch` (700 MB CPU wheel) + `sentence-transformers` from the image
- Adds `onnxruntime` (~50 MB) + `tokenizers` (~10 MB)
- Production image shrinks from ~1.5 GB to ~150 MB
- Cold-start time drops from ~12 s to ~3 s

The `models/` directory was de-gitignored to allow shipping the bundle
in-repo. (Alternative: GitHub Releases asset + Docker download step.
In-repo is simpler at this size; revisit if the bundle grows.)

**2. Single env file.** `deploy/credentials.env` + `deploy/link-env.sh`
collapsed into one `.env` at the repo root, read by both uvicorn (via
`python-dotenv` in `config.py`) and `deploy/deploy.sh` directly. No more
translation script.

**3. Aggressive archive.** Anything not needed for the active codebase
moved into `.archive/` (hidden directory, gitignored):
- `.archive/deploy-extras/` — `README.md`, `credentials.env.example`,
  `install-flyctl.sh`, `setup-ci.sh`, the stale Neo4j Aura backup
- `.archive/scripts-extras/` — 9 one-off scripts + `experiment_chunk_format.py`
- `.archive/data-source-files/` — raw `mechse_syllabi.json`,
  `course_info.json`, `instructors.json`, per-course edit buffers
- `.archive/samples/` — sample PDFs
- `.archive/legacy-local-docker/` — original docker-compose stack + Oracle
  self-host (carried over from Phase 13)

`deploy/` now contains exactly one file (`deploy.sh`). `scripts/` contains
six load-bearing scripts. The repo root has `.env` (gitignored) plus the
usual `README.md`, `config.py`, `config.yml`, `Dockerfile.api`, `fly.toml`,
`requirements*.txt`, `CLAUDE.md`.

`config.py` lost dead constants `SYLLABI_PATH`, `COURSE_INFO_PATH`,
`INSTRUCTORS_PATH` (only the archived `seed.py` ever read them).

---

## Phase 18 — Data-quality rework + semantic topic search

**Date:** June 4, 2026

Phase 17 got the app *deployable*; Phase 18 got the *signal* clean. The driving
complaint (`fix.txt`): top matches were dominated by baseline embedding noise and
same-domain near-duplicates, and the category enrichment from Phase 16 had
contaminated the department-faithful category distributions the non-obvious score
depends on. The goal was a final, deliberate pass over what gets embedded and how
it's scored — "this should be the last time we re-embed."

**1. Variant-C embedding text.** `scoring/chunker.py` dropped the course-name
prefix from each topic chunk. The embedded text is now
`"{topic}: {definition}. Related concepts: {tags}."` with **no** `<course>` prefix
— the implicit department signal that prefix carried was masking exactly the
cross-domain isomorphisms SciNCL exists to surface. The build is factored into two
shared helpers, `topic_chunk_text()` (embedded text) and `topic_term_text()`
(lex/TF-IDF text), so the ingest path and the merge-by-topic append path can never
drift in format. Tags now also feed `term_text`, with hyphens flattened
(`isomorphic-to-rlc-circuit` → `isomorphic rlc circuit`) so multi-word tags bridge
lexically too.

**2. CHEM glossary, term-counts only.** `mechse_syllabi.json` glossary entries
became `glossary`-type chunks that feed term counts (lex/TF-IDF) but are
deliberately **not** embedded — keeping ~100 glossary points per CHEM course out of
the vector space avoids making CHEM spuriously "central." Their value is purely
lexical bridging on genuine CHEM↔PHYS overlaps.

**3. Categories reverted.** `data/courses_dump.json` category distributions were
rolled back to their pre-Phase-16 (commit `385b853`) department-faithful values;
the enriched tags were kept. This re-surfaced the electrical↔mechanics family
(ECE 205 ↔ TAM 211/212) at the top of the non-obvious ranking.

**4. Scoring retune.** `sem_floor` 0.35 → **0.74** in `config.yml`, and the
non-obvious score now uses the *calibrated* semantic score
(`scoring/hybrid_scorer.py`) instead of raw cosine — this killed baseline-noise junk
(CS 521 ↔ ME 330 fell from rank #16 to #433). `min_score` 0.55 → 0.40 retuned the
NetworkX graph to ~129–140 edges.

**5. Shared-topics drill-down.** `scoring/vector_similarity.shared_topics()` computes
the *direct pairwise* topic-to-topic cosine matrix between two specific courses and
returns the strongest cross-course pairs — intentionally **not** served from the
global `topic_similarity` table, whose per-topic top-k is dominated by same-domain
near-duplicates and so discards the cross-domain bridges this view exists to show
(e.g. ECE 205 "Ohm's law" ↔ TAM 212 "Newton's laws" at 0.90). Exposed at
`/similarity/shared-topics` and rendered in the UI "why similar" modal.

> **Key insight:** the canonical ME 340 ↔ ECE 205 (damping ↔ RLC, 0.92) is a
> *topic-level* signal, not a *course-level* non-obvious rank — course averaging
> washes it out. That's expected behaviour, surfaced by the drill-down, not a bug.

**6. Ingestion hardening.** Both ingest paths dropped a dead `topic_definitions.json`
read. Append is now **merge-by-topic** (`_upsert_topics_into_course()` in
`api/routes/ingest.py`): an incoming topic that matches an existing one
(cosine ≥ `MERGE_COSINE` = 0.88) unions tags, takes the richer definition,
re-embeds, and overwrites the same `chunk_id` — no dilution from duplicate near-topics;
a genuinely new topic is added.

**7. The 905 stale-embedding bug (critical data fix).** `delete_course` had never
deleted a course's rows from `chunks` / `chunk_embeddings`, so every re-ingest left
the old vectors behind. The database had accumulated **905 stale embeddings (~49%)**
— old `topicpatch` / `definition` / `_app` chunks polluting every semantic
similarity. Fixed `delete_course` to cascade, added
`wipe_content_tables()`, and made `scripts/restore_from_dump.py` wipe to a clean
slate before reloading. The DB is now exactly **938 legitimate topic vectors**
(35 courses; MATH 415/463 contribute none — they have empty topic lists, so their
`non_obvious_score` is legitimately null).

**8. Semantic topic search.** `/topics/search` was rigid keyword matching — a
`topic_text ILIKE '%q%'` that returned *nothing* for a query like "Machine Learning."
It now embeds the query with the same SciNCL encoder
(`storage.vectors.store.search_topics_semantic`) and runs a pgvector HNSW ANN over
the 938 topic vectors, ranking by cosine; "RLC circuit" → ECE 205 at 0.93, and the
real ML topic (CS 568) surfaces where keyword search drew a blank. Results resolve
back to topic metadata via `pg_store.resolve_topic_chunks()`, which recovers the
clean `topic_text` by longest-prefix match against `topic_categories`. The
cross-domain phase-2 (shared-tag) pass is unchanged. With this in place,
`topic_definitions.json` is no longer read at request time — topic descriptions are
parsed straight from the embedded chunk text
(`pg_store.get_topic_descriptions_for_course`), so the UI shows exactly what was
embedded. The file is retained only as recovery input for `restore_from_dump.py`.

**Archive sweep.** Dead files removed from git and moved into the gitignored
`.archive/` (kept on disk), matching the Phase 17 convention:
- `data/ME_340_edit.json` — a one-off per-course edit buffer.
- `data/course_info.json`, `data/instructors.json` — orphaned source files. Their
  config constants (`COURSE_INFO_PATH`, `INSTRUCTORS_PATH`) were dropped in Phase 17
  and instructor data now lives inside `courses_dump.json`; nothing reads them.
- `scripts/enrich_tags.py` — the one-off Phase 16 tag/category re-derivation script,
  whose category output was reverted this phase (tags are now baked into the dump).
- `NEXT_CHAT_HANDOFF.md` — a session-handoff scratch doc, superseded by this section.

Retained as the recovery path: `scripts/restore_from_dump.py` +
`data/courses_dump.json` + `data/topic_definitions.json` + `data/mechse_syllabi.json`
(the last still read for CHEM glossaries). `similarity.py` was also decoupled from
`topic_definitions.json` (DB-backed descriptions like `topics.py`), so no request
path reads that file anymore — it survives only as restore input.

**Production runtime decision — PyTorch, ONNX parked.** The Phase 17 ONNX cutover
was re-examined and deliberately **not** adopted. Both exported artifacts exceed
GitHub's 100 MB/file limit (INT8 110 MB, FP32 `.data` 439 MB), so neither is
git-committable — shipping ONNX would require an external Release-asset/LFS download
step. INT8 also drifts to **0.9946** cosine against the PyTorch-built DB vectors
(fails the 0.998 gate; practically invisible in ranking, but a permanent
mixed-precision asterisk). Crucially, ONNX's *only* wins — smaller image, lower
memory, faster cold start — are already obtained for free: `fly.toml` runs one
always-on `shared-cpu-1x` 2 GB machine (`min_machines_running = 1`,
`auto_stop_machines = "off"`) at ~$3.89/mo, inside Fly's $5 hobby credit, so the
model is always warm (no cold start) and the bill is zero. ONNX would cost
simplicity, a hermetic build, and dual-encoder maintenance for nothing not already
in hand. The dormant 526 MB `models/` artifacts were deleted; the scaffolding
(`scripts/export_scincl_onnx.py`, the encoder's `_encode_onnx` path, the Dockerfile
dual-mode build, `config.yml` runtime switch) is kept so ONNX remains a clean opt-in
if scale ever changes the math.

---

## Phase 19 — Topic-search overhaul: lexical-primary, category-divergence cross-domain, tag prune + re-embed

**Date:** June 4, 2026 (continuation)

Phase 18 had made `/topics/search` purely semantic (embed the query, ANN over the
938 topic vectors). Live testing exposed that this was wrong for a search box:
**SciNCL is a scientific-document model, not a keyword matcher.** For a bare query
its cosines are near-uniform — the whole 938-vector corpus sits in a ~0.04-wide band
— so it returned the densest cluster near the corpus centroid (CHEM/thermodynamics)
instead of relevant topics. For "machine learning" the noise literally outscored the
signal: CHEM "Entropy" (0.856) ranked above the real ML topic CS 568 (0.838).

**1. Lexical-primary hybrid search.** `/topics/search` (`api/routes/topics.py`) is now
tokenized lexical matching first (`pg_store.search_topics_tokens` — any query token
against topic text + tags, ranked by exact-phrase > tokens-matched > text-over-tag).
This is precise and is the form a search box should have: "machine" → ME 370 Machine
dynamics / ME 371 Machine components. Tokenizing also fixed the original
"machine learning returns nothing" bug — the old code required the whole phrase as one
`ILIKE` substring; now `machine ∪ learning` both hit. SciNCL ANN survives **only as a
fallback** when a query has zero lexical hits (e.g. "photosynthesis").

**2. Cross-domain rework — category divergence, not shared tags.** The old phase-2
("surface any topic sharing any tag with any direct match") was a giant OR that
over-connected everything through a handful of broad tags. Replaced with
`scoring/vector_similarity.cross_domain_matches`: seeded from the strongest direct
matches, follow each seed topic's **stored embedding** (reliable, unlike a bare-query
vector) to its nearest neighbours in other courses, and keep only those that are both
semantically close (cosine ≥ 0.82) **and** category-divergent (JSD ≥ 0.25). Ranked by
`non_obvious = sem × cat_jsd` — the project thesis, applied at topic level. Each result
carries its source topic (for the explainer) and its real scores. This surfaces genuine
bridges: vibration → ECE 205 sinusoidal response, MATH 285 eigenvalues → TAM 445
principal stress directions, heat transfer → MATH 285 heat equation. The UI
(`ui/index.html`) now ranks the cross-domain section by non-obvious score and shows
`non-obvious % · semantic % · domain gap %` on each connection.

**3. Tag prune (the over-enrichment, part 2).** Phase 16's isomorphism-aware enrichment
had sprayed "bridge vocabulary" everywhere — `isomorphic-to-rlc-circuit` was on 159
topics, `mechanical-electrical-analogy` on 174. The genuine isomorphism applies to a
dozen second-order-ODE topics, not 159. Diagnosis showed **85% of the 8,960 tags are
specific single-topic descriptors (good)** — the rot was ~137 `isomorphic-to-*` /
`*-analogy` meta-tags (747 tag-slots). Pruned that whole class from both
`topic_categories` and the recovery dump. The canonical ME 340 ↔ ECE 205 isomorphism
survives via genuinely shared specific tags (`second-order-system`, `quality-factor`)
plus the embeddings — nothing real lost.

**4. The definitive re-embed.** Prune-only cleans the tag tables but not the *frozen
vectors* (the junk tags were baked into each chunk's embedded text), so 37% of vectors
still pulled toward the analogy cluster — e.g. MATH 221 "chain rule" sat at cosine 0.87
to ME 340 "beats". Since the dump was already pruned, `scripts/restore_from_dump.py`
re-embedded all 938 vectors from clean text (zero Gemini cost) — the "one last re-embed"
fix.txt anticipated. chain-rule↔beats fell 0.87 → 0.83; canonical pair preserved
(sem 0.875, non_obvious 0.221).

**5. `[SEP]` regression fix.** The re-embed surfaced a *pre-existing* bug:
`scripts/build_topic_graph._build_chunk_to_topic` still split chunk text on the `[SEP]`
separator that Phase 18's variant-C format removed, so `topic_similarity` had been
silently empty (0 pairs). Repointed it at the shared longest-prefix matcher →
9,380 pairs restored (the "similar topics in other courses" sublist works again).

**6. Search performance — 10.3 s → ~2.5 s (4×).** The pipeline had been doing ~50
separate round trips to the cloud Postgres (~195 ms each). Batched the hot paths:
`pg_store.get_topic_descriptions_bulk` (per-course descriptions in 2 queries, not 2·N);
`vector_similarity._seed_embeddings_bulk` (all seed vectors in 3 queries); a single
batched `resolve_topic_chunks` over every cross-domain hit; and
`pgvector_store.search_neighbors_batch` — **one LATERAL query that runs all seeds' ANN
at once** (still HNSW-indexed) instead of one round trip per seed. A `Cache-Control:
no-store` header on the search response also stops the browser replaying a stale payload.

**7. Second-layer tag hygiene + ingestion fix (the source of the rot).** The
analogy-tag prune left a second layer: controls / application vocabulary
(`transfer-function`, `state-space-model`, `frequency-response`, `material-processing`,
`fluid-dynamics`) mis-applied to off-domain topics — these are legit on real engineering
topics, so a global delete would lose signal. Fixed with a **category-aware prune**: a
curated set of engineering-domain tags is removed from a topic only when the topic has
< 0.15 category weight in that tag's domain (so ME 340 damping keeps `quality-factor`,
but MATH 221 "limit laws" loses `transfer-function`). 132 mis-applied tags dropped from
the dump, then a second re-embed baked it in — "Volumes → machine tools" disappeared
entirely. **Root cause fixed too:** the live ingest prompt (`llm/topic_extractor`
`_ANALYZE_SYSTEM`) had *ordered* this over-tagging ("always include isomorphism-bridge
tags", "be liberal — over-tag rather than under-tag"). Rewrote it to tag faithfully and
specifically (8–15 tags, analogy/bridge vocabulary explicitly forbidden), with the
reframe that cross-domain matching now comes from embeddings + category divergence — so
topics bridge by *genuinely* sharing real concepts, not by fabricated analogy tags. New
uploads now produce clean data by construction.

---

## Cross-cutting design principles that emerged

These weren't planned up front — they crystallized through arguments with reality. They are the rules the codebase now lives by.

### 1. Math scores, LLM explains — never the reverse
Decided after the Phase 6 cost incident (~2,208 Gemini requests). Scores must be deterministic and reproducible. The LLM is used in exactly two places:
- At ingest: **one** call extracts topics + categories.
- On user demand: **one** call generates an explanation.

The LLM never participates in scoring.

### 2. One LLM call per ingest
Initially the upload pipeline was making multiple Gemini calls (description, then topics, then categories, then objectives). Consolidated into one comprehensive `analyze_course()` call returning everything in one JSON response. Triggered by: *"Are we instigating multiple calls for one upload? I'd like it to be one giant request."*

### 3. User-owned API keys, browser-only
Gemini key lives in the browser's `localStorage` and is sent as `X-Api-Key`. Server never stores or sees it (only forwards it to Gemini). Missing key → HTTP 400 with the actual Gemini error.

### 4. No silent fallbacks
From your direct ask: *"If the gemini api key does not work, simply give an error message notifying me what the issue with my key is."* This rule applies everywhere now — bad key, rate limit, invalid JSON → ingest stops, partial data cleaned up, exact error surfaced.

### 5. No domain-specific calibration
From: *"Don't be biased by domain... ensure scoring for courses and topics are done the same for all courses."* Previously there was a 0.75 floor subtraction on sem_score (MechSE-specific calibration). Removed because it zeroed out cross-domain pairs — the very thing we were trying to surface.

### 6. Cache aggressively
- Topic categories: permanent (set on ingest, never re-computed).
- Similarity scores: recomputed only when a course changes.
- LLM explanations: generated once, cached forever in `similarity_cache.llm_explanation`. First viewer pays, everyone after gets it free.
- `/topics/search`: LRU cache, cleared on ingest.

### 7. Append, don't replace (for supplementary material)
`POST /ingest/append` lets you add lecture notes / slides to an existing course. Term counts **accumulate** (not overwritten); a course profile only gets richer over time.

### 8. Local dev = production (against shared cloud DBs)
After Phase 10, your laptop and Fly both talk to the same Supabase + Aura. Reduces "works on my machine" cycles to zero.

### 9. Math is for narrowing, LLM is for explaining
Topic-to-topic uses Milvus ANN to narrow ~10,000 possible topic pairs to ~6,748 candidates with `sem ≥ 0.70`. Only the candidates the user actually clicks into get an LLM explanation. This is the *parked Phase 2* you put in `ideas.md`.

---

## Repo split — what lives where

| Repo / path | Phases | Stack snapshot | Status |
|---|---|---|---|
| `~/Desktop/Projects/UIUC Research/SIIP/` | 0–3 | DynamoDB + Lambda + API Gateway + S3 + cPanel-style UI | Archived |
| `~/Desktop/Projects/UIUC Research/akhil_app/` (this) | 4–11 | Fly + Supabase pgvector + Aura + Gemini | **Active** |

Transcripts:
- SIIP session transcripts: **lost**. Only `~/.claude/history.jsonl` (user prompts) survives.
- akhil_app session transcripts: present at `~/.claude/projects/-home-dogeyboy19-Desktop-Projects-UIUC-Research-akhil-app/*.jsonl` (six sessions, ~13 MB total).

---

## Numerical snapshot (May 2026)

| Metric | Count |
|---|---|
| Courses | 35 |
| Chunks | 1,744 |
| Topics labeled | 1,669 |
| Similarity pairs computed | 595 (= 35·34/2) |
| Pairs with `non_obvious_score` | 528 |
| Cached LLM explanations | 466 |
| Neo4j edges (final_score ≥ 0.55) | 123 |
| Topic similarity pairs (ANN, ≥ 0.70) | 6,748 |
| Production cost | $0/mo (Fly hobby credit + Supabase free + Aura free) |

---

## Future work (parked in `ideas.md` and `docs/ARCHITECTURE.md` §15)

1. **Topic-to-topic LLM explanations** — phase 2. Same 3-field format (shared_math / why_surprising / analogy), cached forever per pair. Cost: ~$0.001/pair.
2. **Score refinement via LLM/human feedback** — use explanation quality to up/down-weight `sem_score`, stored in a separate `refined_score` column. Reflects human + LLM judgement over time.
3. **Topics inline in Explorer** — currently topics live in their own view; embedding them in the per-course panel would shorten "this course is interesting" → "here's a related topic in another course".
