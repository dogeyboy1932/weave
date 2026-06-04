# Handoff — SIIP/Armchair, mid-deployment state

You are continuing work on **`/home/dogeyboy19/Desktop/Projects/UIUC Research/akhil_app/`**. The previous chat got far but ran out of context. Read this end-to-end before doing anything.

## What this project is (one screen)

UIUC MechSE curriculum-mapping tool ("Armchair"). Live at <https://siip-armchair-akhil.fly.dev>. Supervisor: Eliot Bethke (bethke2@illinois.edu).

Goal: surface **non-obvious cross-domain connections** between course topics. The canonical example is on the supervisor's slide deck (image in `.archive/samples/image.png`): ME 340 spring-mass-damper (`mẍ + cẋ + kx = f(t)`) is **isomorphic** to ECE 210 RLC circuit (`LQ̈ + RQ̇ + Q/C = V(t)`) — same 2nd-order ODE up to variable renames. The engine is built to surface that.

Scoring math (don't change):
```
sem_score    = symmetrised SciNCL cosine via pgvector ANN
lex_score    = TF-IDF cosine
category_jsd = JSD over 8-bin per-topic distributions (LLM-labelled)
final_score  = (ALPHA_LEX·lex + ALPHA_SEM·sem_cal + ALPHA_CAT·(1-cat_jsd)) / Σ
non_obvious  = sem · category_jsd                       ← headline signal
```
All math is deterministic. The LLM (Gemini) only **parses syllabi** at ingest and **renders human-readable explanations** on demand. The LLM never compares courses.

**Read `docs/ARCHITECTURE.md` and `docs/HISTORY.md` for the deeper picture.** CLAUDE.md at the repo root is the short pointer file you should also read.

## Current stack

- **One DB** — Supabase Postgres + pgvector (HNSW-indexed). DATABASE_URL in `.env`.
- **One pipeline** — local dev = `uvicorn` against the same Supabase production uses.
- **One env file** — `.env` at the repo root. Read by `config.py` (via python-dotenv) and `deploy/deploy.sh` directly.
- **Config knobs** — `config.yml` (single source of truth, env overrides win).
- **Embedding model** — SciNCL (`malteos/scincl`), PyTorch runtime today, ONNX runtime is in-progress (see below).
- **Graph stats** — `scoring/graph_compute.py` runs NetworkX Louvain + PageRank over `similarity_cache`, writes `community` + `pagerank` columns on `courses`. No Neo4j.
- **Six scripts in `scripts/`** — `restore_from_dump.py` (canonical rebuild), `build_graph.py`, `build_topic_graph.py`, `dump_courses.py`, `export_scincl_onnx.py`, `reset.py`. Anything else lives in `.archive/scripts-extras/`.

## What's been done this session (already on local disk, not yet pushed)

1. **Phase 12-15 rewrite is committed locally** in `385b853` ("Lightened model weight and reembedded data"). Includes: Neo4j removal, `scoring/graph_compute.py`, `chunker.py` `+tags` suffix, pgvector HNSW index, `query_log` middleware, config.yml introduction, `scripts/restore_from_dump.py`, ONNX export script, archive consolidation. `git log --oneline origin/main..HEAD` shows this is 1 commit ahead of origin/main.

2. **Phase 16 — tag enrichment (UNCOMMITTED).** Modified files:
   - `llm/topic_extractor.py` — new `enrich_topics()` function + new `_ENRICH_SYSTEM` prompt that bakes in isomorphism vocabulary (`isomorphic-to-rlc-circuit`, `mechanical-electrical-analogy`, `force-voltage-analogy`, etc.). Asks for 15-25 tags per topic from 5 categories: structural-form, canonical-equation, isomorphism-bridge, concept-level, physical-mechanism.
   - `scripts/enrich_tags.py` — parallel Gemini caller, 8 concurrent, with a colon-progressive matcher that handles LLM-name-mangling.
   - `data/courses_dump.json` — **100% of 938 topics enriched** (805 ≥12 tags, 133 7-11 tags, 0 un-enriched). Spent ~$0.12 on Gemini API.
   - `docs/HISTORY.md` — Phase 16 entry documents the 3-pass process.

3. **Phase 17 (UNCOMMITTED, ONNX export verified, runtime not yet smoke-tested).** Files changed for the ONNX cutover:
   - `scripts/export_scincl_onnx.py` — fixed pooling (mean → CLS).
   - `storage/vectors/encoder.py` — fixed pooling (mean → CLS) in `_encode_onnx`.
   - `Dockerfile.api` — uncommented `COPY models/ ./models/`.
   - `.gitignore` — removed `models/` ignore.
   - `fly.toml` — set `[build.args] EMBEDDING_RUNTIME = "onnx"`, dropped VM memory from 2gb → 1gb.
   - `models/scincl-fp32.onnx` + `models/MANIFEST.json` + tokenizer files — created by the export, **not yet `git add`-ed.**
   - `config.yml` — **NOT yet flipped to `runtime: onnx`.** Still says pytorch. Do this just before pushing once runtime is smoke-tested.

## In-progress background process

**`scripts/restore_from_dump.py` was running at session end.** PID 1026343. Started ~10 min before this handoff. Was at step 4/8 (computing pairwise similarity) — should finish naturally in ~2-5 more min. Tail the log to check:

```bash
tail -f /tmp/claude-1000/-home-dogeyboy19-Desktop-Projects-UIUC-Research-akhil-app/691f052b-5407-463f-8285-f03c11c6c7da/tasks/bji8itfit.output
```

If the log ends with `✓ Restore complete`, it's done; the DB now has embeddings reflecting the +tags suffix with the enriched tag set.

If it crashed, restart with `python scripts/restore_from_dump.py`.

## ONNX — fixed and ready to ship (status when this handoff was written)

### Fixed and verified
The pooling bug (sentence-transformers uses **CLS-token pooling** for SciNCL,
not mean pooling) was patched in two places:

- `scripts/export_scincl_onnx.py` — `_mean_pool` → `_cls_pool` (just `last_hidden[:, 0, :]`)
- `storage/vectors/encoder.py:_encode_onnx` — same swap

Re-ran the FP32 export. **Parity check passed perfectly:**
```
cosines: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
worst = 1.0000  (gate: 0.998)
```

### Current artifact state
`models/` contents (~440 MB total):
```
MANIFEST.json                 533 B
scincl-fp32.onnx              103 KB     (graph metadata only)
scincl-fp32.onnx.data         439 MB     ← external weights — GitHub's 100 MB limit blocks this
special_tokens_map.json
tokenizer_config.json
tokenizer.json
vocab.txt
```

### ⚠️ The 439 MB problem
GitHub's hard limit is 100 MB per file. The ONNX export with `torch.onnx.export`
externalized the weights into `scincl-fp32.onnx.data`. **You can't commit it
to git as-is.** Three options, ordered by preference:

1. **Try INT8 again.** Now that pooling is correct, INT8 quantization
   (`python scripts/export_scincl_onnx.py` without `--skip-quantize`) should
   bring the total to ~110 MB — *might* fit GitHub's 100 MB limit per file
   and almost certainly passes the 0.998 parity gate. Try this first.

2. **GitHub Release asset.** Upload the 439 MB bundle as a release asset
   (no size limit), and add a Docker build step that `wget`s it during
   image build. Cleaner long-term but adds build complexity.

3. **Git LFS.** Track `*.onnx*` in LFS. Works but everyone who clones pays
   the bandwidth, and Fly's remote builder needs LFS support enabled.

My recommendation: do (1) first. If INT8 doesn't pass parity, do (2).

### Still to do for ONNX
1. **Flip `config.yml`** to:
   ```yaml
   model:
     embedding:
       runtime:   onnx
       onnx_path: models/scincl-fp32.onnx
       id:        scincl-v1-fp32   # keep — embeddings in DB are stamped this
   ```
2. **Optionally try INT8** (`python scripts/export_scincl_onnx.py` without
   `--skip-quantize`) and see if INT8 passes the 0.998 gate now that pooling
   is correct. If it does, ship the smaller bundle. If not, FP32 is fine.
3. **Verify the runtime encoder works end-to-end:**
   ```bash
   uvicorn api.main:app --port 8080 --reload
   # Should print "Loading SciNCL ONNX from models/scincl-fp32.onnx" then
   # "Model ready (onnx)" on first encode call. /upload should still work.
   ```
4. **Push** (see next section).

If runtime fails for any reason, revert these files to PyTorch defaults:
- `Dockerfile.api`: change `COPY models/ ./models/` back to its comment block
- `.gitignore`: re-add `models/` line
- `fly.toml`: drop the `[build.args]` block; memory back to `2gb`
- `config.yml`: keep `runtime: pytorch`

## The categories-vs-tags discovery (defer for now, but record it)

After Phase 16 enrichment + Phase 16 re-embed completed, the canonical
ME 340 ↔ ECE 205 isomorphism pair moved like this:

| | Before enrichment | After enrichment | Δ |
|---|---|---|---|
| `sem_score` (SciNCL cosine) | 0.8629 | 0.8746 | +0.012 ✓ |
| `category_jsd` (department distance) | 0.4243 | 0.3601 | **−0.064 ✗** |
| `non_obvious_score` (`sem × cat_jsd`) | 0.3661 | 0.3149 | **−0.051 ✗** |

The **sem score moved the right way** (SciNCL now sees the isomorphism more
strongly thanks to the richer +tags chunks). But the **category JSD dropped**
because the enrichment also rewrote the 8-bin category distributions, and the
LLM, when asked to assign categories with the new isomorphism-aware tag
vocabulary in mind, started giving e.g. ME 340 damping topics non-trivial
"Electrical" weight (because the isomorphism really is electrical). This made
ME 340's and ECE 205's average category vectors more similar overall, which
shrank the JSD.

**User's hypothesis** (worth investigating next): include explicit course-domain
signal in the embedding text itself — e.g., `"Mechanical Engineering :: ME 340 :: <topic>"`
or `"Electrical Engineering :: ECE 210 :: <topic>"`. The leading domain prefix
would push SciNCL toward keeping the department signal in the embedding,
while the tag suffix still surfaces the cross-domain isomorphism. The pair
would land at: high `sem_score` (isomorphism), high `category_jsd` (different
departments) → high `non_obvious_score`. Best of both worlds.

**Net effect on the live system right now**:
- The new top-5 non-obvious is arguably *more* interesting (3 CS↔PHYS pairs
  surface, all of which look genuinely cross-domain).
- The single canonical ME 340 ↔ ECE 205 pair ranks slightly lower.
- Some users may not notice. Eliot might.

**Three options for next session:**
1. Ship as-is. The top-5 is broadly better. Treat ME 340 ↔ ECE 205 regression as acceptable.
2. **Revert just the categories** (keep new tags). Pull the pre-Phase-16
   categories from git history at commit `385b853` and write them back into
   the current dump. Re-score. ~10 min. Probably the safest.
3. Try the user's proposal: re-embed with course-domain prefix in the chunk
   text (e.g., `"Mechanical Engineering :: ME 340 [SEP] Topic: …"`) and see if
   the canonical pair recovers without touching categories. ~10 min + re-embed.

I lean **option 2** for the immediate next session, with option 3 as a
follow-up experiment.

## What's left to do, in order

1. **`restore_from_dump.py` finished cleanly.** DB state:
   - `chunk_embeddings`: 1,851
   - `similarity_cache`: 630
   - `courses with community`: 36 (CS 411 came back as a stale-data artefact — it
     was deleted earlier in the session but the dump-restore upserts didn't
     drop courses missing from the dump; safe to delete manually if you want).
   - 179 edges above MIN_SCORE=0.55, 7 communities.

2. **ONNX is fixed.** `models/scincl-fp32.onnx` exists, parity = 1.0000. Just
   flip `config.yml` (see ONNX section above) and test. Optionally try INT8.

3. **Decide on the categories question** (see "categories-vs-tags discovery"
   section above). My recommendation: revert categories from `385b853`, keep
   new tags.

4. **Verify a canonical isomorphism pair improved** (sanity check the enrichment worked):
   ```bash
   source .venv/bin/activate && python -c "
   from storage.postgres.store import _Conn
   with _Conn() as cur:
     cur.execute(\"SELECT final_score, sem_score, category_jsd, non_obvious_score FROM similarity_cache WHERE (course_a='ECE 205' AND course_b='ME 340') OR (course_a='ME 340' AND course_b='ECE 205')\")
     print(cur.fetchone())"
   ```
   The earlier baseline (before enrichment) was: `final=0.5678, sem=0.8629, cat_jsd=0.4243, non_obvious=0.3661`. Expected after enrichment: `sem_score` should rise toward 0.88-0.92 if the isomorphism tags landed.

5. **Run the local-uvicorn smoke test** before pushing:
   ```bash
   uvicorn api.main:app --port 8080 --reload &
   for i in $(seq 1 60); do curl -sf http://localhost:8080/health >/dev/null && break; sleep 0.5; done
   curl -s http://localhost:8080/courses | python -c "import sys,json; print(f'/courses: {len(json.load(sys.stdin))} courses')"
   curl -s 'http://localhost:8080/graph/all?min_score=0.4' | python -c "import sys,json; d=json.load(sys.stdin); print(f'/graph/all: {len(d[\"nodes\"])} nodes, {len(d[\"edges\"])} edges')"
   kill %1
   ```
   Expected: 35 courses, 35 nodes, ~600 edges.

6. **Single commit + push:**
   ```bash
   git add -A
   git commit -m "Phase 16-17: 100% isomorphism-aware tag enrichment + ONNX runtime"
   git push origin main
   ```
   Push triggers `.github/workflows/deploy-fly.yml`, which runs `flyctl deploy`. **If ONNX is enabled**, the Fly build will use `--build-arg EMBEDDING_RUNTIME=onnx` (set in fly.toml) and ship the smaller image.

7. **Watch CI:** `gh run watch` (or visit the Actions tab). On success, smoke-test the live URL:
   ```bash
   curl -s https://siip-armchair-akhil.fly.dev/health
   curl -s "https://siip-armchair-akhil.fly.dev/graph/all?min_score=0.4" | python -c "import sys,json; d=json.load(sys.stdin); print(len(d['nodes']), len(d['edges']))"
   ```

## Key files to read before doing anything

- `CLAUDE.md` — 2-minute orientation
- `docs/ARCHITECTURE.md` — full system design
- `docs/HISTORY.md` — Phases 12-17, every decision with reasoning
- `config.yml` — every tunable
- `scripts/restore_from_dump.py` — the canonical rebuild pipeline
- `scripts/export_scincl_onnx.py` — the ONNX path (with the broken pooling)
- `storage/vectors/encoder.py` — `_encode_pytorch` (works) and `_encode_onnx` (needs same CLS fix)
- `llm/topic_extractor.py` — both `analyze_course()` (ingest) and `enrich_topics()` (Phase 16)

## Constraints / preferences from the user

- **Cost-sensitive on Gemini API.** Use `gemini-2.5-flash-lite` (it's already the default). Total spend today: ~$0.12. Don't burn money.
- **"One pipeline" — no parallel local-only stack.** Anything in `.archive/` is dead-code; don't revive without asking.
- **Honest about partial success.** When something doesn't work (like the INT8 parity), document it and pick a sensible fallback. Don't pretend.
- **Aggressive about archiving.** Move dead code into `.archive/` rather than deleting.
- **Match output JSON shapes exactly** — the UI in `ui/index.html` expects specific keys on `/graph/all`, `/similarity/*`, etc.

## Sample of what good enriched tags look like

After Phase 16, `Damping ratio and natural frequency` in ME 340:
```
tags (15): [
  'second-order-system', 'undamped-natural-frequency', 'damped-natural-frequency',
  'characteristic-equation', 'oscillation-behavior', 'energy-dissipation',
  'transient-response', 'stability-analysis', 'lumped-parameter-model',
  'isomorphic-to-rlc-circuit', 'damping-resistance-analogy', 'quality-factor',
  'parameter-identification', 'system-characterization', 'frequency-response'
]
```

And `Rlc circuits` in ECE 205:
```
tags (15): [
  'second-order-system', 'second-order-ode', 'natural-frequency', 'damping-ratio',
  'underdamped-response', 'overdamped-response', 'critically-damped-response',
  'resonance', 'energy-oscillation', 'isomorphic-to-spring-mass-damper',
  'transfer-function-denominator', 'quality-factor', 'damped-oscillation',
  'transient-behavior', 'second-order-differential-equation'
]
```

Both sides tag the isomorphism explicitly. SciNCL should now embed them closer together than before.

## Final note

The user is patient about quality but allergic to: hardcoded rules, fake API costs, files that don't pull their weight, and lying about success. Keep that in mind when proposing fixes or workarounds.

Good luck.
