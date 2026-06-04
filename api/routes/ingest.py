import json
import tempfile
from pathlib import Path

import numpy as np

from typing import Optional
from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, UploadFile

import config
from scoring.driving_terms     import compute_idf
from scoring.hybrid_scorer     import score_pair
from scoring.language_model    import build_all_lms
from scoring.category_scorer   import course_category_vector
from storage.vectors import store as vs
from storage.postgres import store as pg_store
from scoring import graph_compute

# Heavy ingest-only deps (PDF parsing, ML model, LLM extractors). These are
# omitted from the slim production container — when missing the upload routes
# return 503 instead of crashing at import time.
try:
    import pdfplumber  # type: ignore
    from scoring.chunker          import chunk_course, topic_chunk_text, topic_term_text, tokenize
    from scoring.vector_similarity import _topic_name
    from storage.vectors.encoder          import encode
    from llm.topic_extractor  import analyze_course, extract_topics
    from llm.category_labeler import label_topic
    _INGEST_AVAILABLE = True
    _INGEST_DISABLED_REASON = ""
except ImportError as _e:
    _INGEST_AVAILABLE = False
    _INGEST_DISABLED_REASON = (
        "PDF/topic-ingest disabled in this deployment (missing optional deps: "
        f"{_e.name}). Run seed.py / build_graph.py locally to refresh data."
    )

router = APIRouter()


def _rebuild_topic_graph(course_id: str):
    """Refresh topic_similarity rows for a course (Topics tab cross-course hits)."""
    from scripts.build_topic_graph import build_for_courses
    build_for_courses([course_id], top_k=10, min_score=0.70)


def _require_ingest():
    if not _INGEST_AVAILABLE:
        raise HTTPException(503, detail=_INGEST_DISABLED_REASON)

# Tracks background ingest state so the UI can poll for errors
# {course_id: {"status": "running"|"done"|"error", "message": str}}
_ingest_status: dict[str, dict] = {}

_ALLOWED_EXTENSIONS = {'.pdf', '.txt'}


def _file_to_text(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == '.txt':
        return raw.decode('utf-8', errors='replace')
    # PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(raw)
        tmp = Path(f.name)
    try:
        parts = []
        with pdfplumber.open(tmp) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return '\n'.join(parts)
    finally:
        tmp.unlink(missing_ok=True)


def _run_ingest(course_id: str, course_name: str, raw_text: str, api_key: str | None = None):
    """
    Full ingestion pipeline for a new course.

    LLM steps (1, 2, 6): topic extraction, category labeling, explanation generation.
    Stack steps (3, 4, 5, 7): chunking, embedding, scoring, graph.
    """
    _ingest_status[course_id] = {"status": "running", "message": ""}
    print(f"\n[ingest] ── Starting '{course_name}' ({course_id}) ──")
    try:
     _run_ingest_inner(course_id, course_name, raw_text, api_key)
     _ingest_status[course_id] = {"status": "done", "message": ""}
     # Topics search results just got stale; drop any cached responses.
     from api.routes.topics import invalidate_search_cache
     invalidate_search_cache()
    except Exception as e:
        msg = str(e)
        print(f"[ingest] ✗ FAILED: {msg}")
        _ingest_status[course_id] = {"status": "error", "message": msg}
        # Clean up partial data so the course isn't left in a broken state
        try:
            pg_store.delete_course(course_id)
            vs.delete_course(course_id)
        except Exception as cleanup_err:
            print(f"[ingest] cleanup error: {cleanup_err}")


def _run_ingest_inner(course_id: str, course_name: str, raw_text: str, api_key: str | None = None):
    from google import genai
    llm = genai.Client(api_key=api_key)

    # ── Step 1: ONE LLM call — description + objectives + topics + categories ───────
    print(f"[ingest] Step 1/5 — LLM: full course analysis …")
    analysis = analyze_course(course_id, course_name, raw_text, client=llm)
    # analysis: {description, objectives, topics: [{name, description, categories}]}
    topics      = analysis['topics']
    description = analysis['description']
    objectives  = analysis['objectives']
    print(f"[ingest]   ✓ {len(topics)} topics, description + objectives — all in one call")

    topic_names = [t['name'] for t in topics]
    topic_tags  = {t['name']: t.get('tags', []) for t in topics}
    # Definitions come straight from the LLM analysis — no dependency on the
    # training-only data/topic_definitions.json, which the LLM output fully
    # supersedes for every extracted topic.
    topic_defs  = {f"{course_id}: {t['name']}": t['description'] for t in topics}

    # ── Step 2: Stack — chunk, embed, store ──────────────────────────────────────
    print(f"[ingest] Step 2/5 — Chunking + embedding …")
    course_dict = {
        'id': course_id, 'name': course_name,
        'description': description,
        'objectives': objectives,
        'topics': topic_names,
        'definitions': [],
    }
    chunks, term_counts = chunk_course(course_dict, topic_defs, topic_tags=topic_tags)

    pg_store.upsert_course(course_id, course_name, description)
    pg_store.upsert_chunks([
        (c['chunk_id'], c['course_id'], c['chunk_type'], c['raw_text'], '[]')
        for c in chunks
    ])
    pg_store.upsert_term_counts(course_id, term_counts)

    embed_chunks = [c for c in chunks if c['chunk_type'] == 'topic']
    if embed_chunks:
        embeddings = encode([c['raw_text'] for c in embed_chunks])
        vs.insert_chunks([
            {'chunk_id': embed_chunks[i]['chunk_id'], 'course_id': course_id,
             'embedding': embeddings[i]}
            for i in range(len(embed_chunks))
        ])
    print(f"[ingest]   ✓ {len(chunks)} chunks stored, {len(embed_chunks)} embedded ({vs.BACKEND_NAME})")

    # ── Step 3: Store category distributions + tags (from step 1, no extra LLM call) ──
    for t in topics:
        pg_store.upsert_topic_category(course_id, t['name'], t['categories'], t.get('tags', []))
    print(f"[ingest]   ✓ {len(topics)} category distributions + tags stored")

    # ── Steps 4 + 5: Math — similarity scoring with category vectors ──────────────
    print(f"[ingest] Steps 4-5/7 — Hybrid scoring + non_obvious_score …")
    all_counts = pg_store.get_all_term_counts()
    lms        = build_all_lms(all_counts)
    idf        = compute_idf(all_counts)

    # Build category vector for every course (so all pairs get category_jsd)
    category_vecs: dict = {}
    for cid in lms:
        dists = pg_store.get_topic_categories_for_course(cid)
        if dists:
            category_vecs[cid] = course_category_vector(dists)

    pair_count = 0
    for other_id in lms:
        if other_id != course_id:
            score_pair(
                course_id, other_id, lms, idf, all_counts,
                category_vecs=category_vecs if category_vecs else None,
            )
            pair_count += 1
    print(f"[ingest]   ✓ {pair_count} pairs scored (hybrid + non_obvious)")

    # ── Step 6: Topic-level ANN graph (Postgres topic_similarity + vector store) ─
    print(f"[ingest] Step 6/7 — Topic-to-topic similarity graph …")
    try:
        _rebuild_topic_graph(course_id)
        print(f"[ingest]   ✓ topic_similarity updated")
    except Exception as e:
        print(f"[ingest]   ⚠ Topic graph rebuild failed: {e}")

    # ── Step 7: Graph stats — Louvain + PageRank reclustering (in-process) ─────────
    print(f"[ingest] Step 7/7 — Recomputing community clusters + PageRank …")
    try:
        summary = graph_compute.recompute_graph_stats()
        print(f"[ingest]   ✓ {summary['nodes']} nodes, {summary['edges']} edges, "
              f"{summary['communities']} communities")
    except Exception as e:
        print(f"[ingest]   ⚠ Graph recompute skipped: {e}")

    print(f"[ingest] ── '{course_name}' fully integrated — {len(chunks)} chunks, "
          f"{pair_count} pairs scored. Explanations generated on user request. ──\n")


@router.get("/status/{course_id}")
def ingest_status(course_id: str):
    """Poll this after upload to check if ingestion succeeded or failed."""
    return _ingest_status.get(course_id, {"status": "unknown", "message": ""})


@router.post("/pdf")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file:        UploadFile = File(...),
    course_id:   str        = Form(...),
    course_name: str        = Form(...),
    x_api_key:   Optional[str] = Header(None),
):
    """
    Upload a PDF syllabus. Runs the full 7-step ingestion pipeline in the background:

    LLM (steps 1, 2, 6): topic extraction → category labeling → explanation generation
    Stack (steps 3, 4, 5, 7): chunking → embedding → scoring → graph

    After ~60 s the new course is fully integrated: similarity scores, category
    distributions, non_obvious_score, and LLM explanations are all populated.
    """
    _require_ingest()
    if not x_api_key:
        raise HTTPException(400, detail="API key required. Set your Gemini API key using the ⚙ button in the app.")
    if Path(file.filename).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail="Only PDF and TXT files are supported")

    raw  = await file.read()
    text = _file_to_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(400, detail="Could not extract text from the file")

    background_tasks.add_task(_run_ingest, course_id, course_name, text, x_api_key)
    return {
        "message":    f"Ingestion started for '{course_name}'.",
        "course_id":  course_id,
        "steps":      [
            "1. LLM: topic extraction + category labeling (Gemini)",
            "2. Postgres: course, chunks, term_counts, topic_categories",
            f"3. Vectors: SciNCL embed → {vs.BACKEND_NAME}",
            "4. Postgres: hybrid scoring for all course pairs",
            "5. Postgres: topic-to-topic similarity (topic_similarity)",
            "6. Postgres: Louvain communities + PageRank (in-process recluster)",
        ],
        "note": "Scores and category data ready in ~60 s. Explanations generated on-demand via the UI.",
    }


def _run_append(course_id: str, course_name: str, raw_text: str, api_key: str | None = None):
    """
    Append additional material (lecture notes, slides, extra reading) to an existing course.

    Topic-centric merge: extracted topics that match an existing course topic
    (cosine ≥ MERGE_COSINE) upgrade it in place — union tags, richer definition,
    re-embed, overwrite the same chunk (one chunk per topic, no dilution).
    Genuinely new topics are added as fresh chunks. All similarity scores for the
    course are then recomputed.
    """
    _ingest_status[course_id] = {"status": "running", "message": ""}
    print(f"\n[append] ── Appending material to '{course_name}' ({course_id}) ──")
    try:
        _run_append_inner(course_id, course_name, raw_text, api_key)
        _ingest_status[course_id] = {"status": "done", "message": ""}
        from api.routes.topics import invalidate_search_cache
        invalidate_search_cache()
    except Exception as e:
        msg = str(e)
        print(f"[append] ✗ FAILED: {msg}")
        _ingest_status[course_id] = {"status": "error", "message": msg}


# Cosine above which an incoming (appended) topic is treated as the SAME topic
# as an existing one and merged in place rather than added as a new chunk. Same-
# course re-coverage of a concept lands well above this; genuinely new topics
# fall below it. Tunable; lives here rather than config.yml as an internal knob.
MERGE_COSINE = 0.88


def _parse_definition(raw_text: str, name: str) -> str:
    """Recover a topic's definition from its stored variant-C chunk text
    ("{name}: {definition}. Related concepts: ...")."""
    body = raw_text
    prefix = f"{name}: "
    if body.startswith(prefix):
        body = body[len(prefix):]
    idx = body.rfind(". Related concepts:")
    if idx != -1:
        body = body[:idx]
    return body.strip()


def _upsert_topics_into_course(course_id: str, course_name: str, topics: list[dict]) -> dict:
    """
    Merge a freshly-analysed set of topics into an existing course, topic-by-topic
    (the lecture-materials path).

    For each incoming topic, embed its variant-C text and compare to the course's
    existing topic embeddings:
      • cosine ≥ MERGE_COSINE → MERGE: keep the existing topic's name + categories
        (categories stay department-faithful), union the tags, keep the richer
        (longer) definition, re-embed and OVERWRITE the existing chunk in place.
        One chunk per topic — no duplicate embedding, so no mean-of-max dilution.
      • otherwise → ADD as a brand-new topic chunk, indistinguishable afterwards
        from any topic ingested originally.

    term_counts: only newly-ADDED topics contribute (accumulated). Merged topics
    are already represented, so they are not re-counted (no lex double-count); the
    re-embedding carries their upgraded signal in sem, the stronger channel.

    Returns {"merged": n, "added": m, "embedded": k}.
    """
    safe_id = course_id.replace(' ', '_').replace('/', '_')

    # Existing topic state: chunk_id → {name, definition, embedding}.
    existing_emb  = vs.get_chunks_with_embeddings(course_id)          # {chunk_id: ndarray}
    existing_text = dict(pg_store.get_chunks_for_course(course_id))   # {chunk_id: raw_text}
    cat_rows      = pg_store.get_topic_category_rows_for_course(course_id)
    existing_tags = {d['topic_text']: d.get('tags', []) for d in cat_rows}
    existing_cats = {d['topic_text']: d.get('categories', {}) for d in cat_rows}

    match_ids:  list[str] = []
    match_vecs: list = []
    name_by_id: dict[str, str] = {}
    next_idx = 0
    for cid, raw in existing_text.items():
        if '__topic__' not in cid:
            continue
        try:
            next_idx = max(next_idx, int(cid.rsplit('__', 1)[1]) + 1)
        except ValueError:
            pass
        name_by_id[cid] = _topic_name(raw)
        if existing_emb.get(cid) is not None:
            match_ids.append(cid)
            match_vecs.append(existing_emb[cid])
    M = np.stack(match_vecs).astype(np.float32) if match_vecs else None

    merged = added = 0
    new_term_counts: dict[str, int] = {}

    for t in topics:
        name     = t['name']
        new_def  = t.get('description', '')
        new_tags = t.get('tags', [])
        new_emb  = encode([topic_chunk_text(name, new_def, new_tags)])[0]

        best_cid, best_cos = None, -1.0
        if M is not None:
            sims = M @ new_emb
            j = int(sims.argmax())
            best_cos, best_cid = float(sims[j]), match_ids[j]

        if best_cid is not None and best_cos >= MERGE_COSINE:
            # ── MERGE into the existing topic (overwrite in place) ──
            ex_name = name_by_id[best_cid]
            ex_def  = _parse_definition(existing_text[best_cid], ex_name)
            ex_tags = existing_tags.get(ex_name, [])
            union_tags  = list(dict.fromkeys([*ex_tags, *new_tags]))
            richer_def  = new_def if len(new_def) > len(ex_def) else ex_def
            merged_text = topic_chunk_text(ex_name, richer_def, union_tags)
            merged_emb  = encode([merged_text])[0]
            pg_store.upsert_chunks([(best_cid, course_id, 'topic', merged_text, '[]')])
            vs.insert_chunks([{'chunk_id': best_cid, 'course_id': course_id, 'embedding': merged_emb}])
            pg_store.upsert_topic_category(course_id, ex_name, existing_cats.get(ex_name, {}), union_tags)
            # keep the match matrix current so later incoming topics see the update
            M[match_ids.index(best_cid)] = merged_emb
            merged += 1
        else:
            # ── ADD as a brand-new topic ──
            new_cid = f"{safe_id}__topic__{next_idx}"
            next_idx += 1
            new_text = topic_chunk_text(name, new_def, new_tags)
            pg_store.upsert_chunks([(new_cid, course_id, 'topic', new_text, '[]')])
            vs.insert_chunks([{'chunk_id': new_cid, 'course_id': course_id, 'embedding': new_emb}])
            pg_store.upsert_topic_category(course_id, name, t.get('categories', {}), new_tags)
            # new topics contribute to lex term counts
            for tok in tokenize(topic_term_text(name, new_def, new_tags)):
                new_term_counts[tok] = new_term_counts.get(tok, 0) + 1
            # make the new topic matchable for subsequent incoming topics
            match_ids.append(new_cid)
            name_by_id[new_cid] = name
            M = new_emb.reshape(1, -1).astype(np.float32) if M is None else np.vstack([M, new_emb])
            added += 1

    if new_term_counts:
        pg_store.accumulate_term_counts(course_id, new_term_counts)

    return {"merged": merged, "added": added, "embedded": merged + added}


def _run_append_inner(course_id: str, course_name: str, raw_text: str, api_key: str | None = None):
    from google import genai
    llm = genai.Client(api_key=api_key)

    # ── Step 1: ONE LLM call — full analysis of the new material ─────────────────
    print(f"[append] Step 1/4 — LLM: full analysis of new material …")
    analysis = analyze_course(course_id, course_name, raw_text, client=llm)
    topics   = analysis['topics']
    print(f"[append]   ✓ {len(topics)} topics extracted and labeled in one call")

    # ── Step 2: Merge topics into the course (upsert-by-topic, no dilution) ───────
    print(f"[append] Step 2/4 — Merging topics (re-embed existing, add new) …")
    stats = _upsert_topics_into_course(course_id, course_name, topics)
    print(f"[append]   ✓ {stats['merged']} topics upgraded in place, "
          f"{stats['added']} new topics added ({vs.BACKEND_NAME})")

    # ── Steps 4: Math — recompute all similarity scores for this course ────────────
    print(f"[append] Step 4/5 — Recomputing similarity scores …")
    all_counts = pg_store.get_all_term_counts()
    lms        = build_all_lms(all_counts)
    idf        = compute_idf(all_counts)

    category_vecs: dict = {}
    for cid in lms:
        dists = pg_store.get_topic_categories_for_course(cid)
        if dists:
            category_vecs[cid] = course_category_vector(dists)

    pair_count = 0
    for other_id in lms:
        if other_id != course_id:
            score_pair(
                course_id, other_id, lms, idf, all_counts,
                category_vecs=category_vecs if category_vecs else None,
            )
            pair_count += 1
    print(f"[append]   ✓ {pair_count} pairs rescored")

    # ── Step 5: Topic-level ANN graph ───────────────────────────────────────────
    print(f"[append] Step 5/6 — Rebuilding topic-to-topic pairs …")
    try:
        _rebuild_topic_graph(course_id)
        print(f"[append]   ✓ topic_similarity updated")
    except Exception as e:
        print(f"[append]   ⚠ Topic graph rebuild failed: {e}")

    # ── Step 6: Graph stats — Louvain + PageRank (in-process) ─────────────────────
    print(f"[append] Step 6/6 — Recomputing community clusters + PageRank …")
    try:
        summary = graph_compute.recompute_graph_stats()
        print(f"[append]   ✓ {summary['nodes']} nodes, {summary['edges']} edges, "
              f"{summary['communities']} communities")
    except Exception as e:
        print(f"[append]   ⚠ Graph recompute skipped: {e}")

    print(f"[append] ── Done — topics merged into '{course_name}', "
          f"{pair_count} pairs rescored ──\n")


@router.post("/append")
async def append_material(
    background_tasks: BackgroundTasks,
    file:      UploadFile = File(...),
    course_id: str        = Form(..., description="Existing course ID to append material to"),
    x_api_key: Optional[str] = Header(None),
):
    """
    Append supplementary material (lecture notes, slides, readings) to an existing course.

    Extracts new topics via LLM, labels them, embeds and stores all new chunks
    alongside existing ones, accumulates term counts, and recomputes all similarity
    scores. Cached LLM explanations for this course's pairs are invalidated so
    they reflect the enriched course content on next request.
    """
    _require_ingest()
    if not x_api_key:
        raise HTTPException(400, detail="API key required. Set your Gemini API key using the ⚙ button in the app.")
    if Path(file.filename).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail="Only PDF and TXT files are supported")

    # Verify the course exists
    courses = {c[0]: c[1] for c in pg_store.get_all_courses()}
    if course_id not in courses:
        raise HTTPException(404, detail=f"Course '{course_id}' not found. Upload it as a new course first.")

    course_name = courses[course_id]
    raw  = await file.read()
    text = _file_to_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(400, detail="Could not extract text from the file")

    background_tasks.add_task(_run_append, course_id, course_name, text, x_api_key)
    return {
        "message":    f"Append started for '{course_name}' ({course_id}).",
        "course_id":  course_id,
        "course_name": course_name,
        "note":       "New topics extracted, embedded, and scored in ~60 s.",
    }
