import json
import time
import tempfile
from pathlib import Path

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
    from scoring.chunker          import chunk_course
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
    local_defs  = {f"{course_id}: {t['name']}": t['description'] for t in topics}
    with open(config.DEFINITIONS_PATH) as f:
        global_defs = json.load(f)
    merged_defs = {**global_defs, **local_defs}

    # ── Step 2: Stack — chunk, embed, store ──────────────────────────────────────
    print(f"[ingest] Step 2/5 — Chunking + embedding …")
    course_dict = {
        'id': course_id, 'name': course_name,
        'description': description,
        'objectives': objectives,
        'topics': topic_names,
        'definitions': [],
    }
    chunks, term_counts = chunk_course(course_dict, merged_defs, topic_tags=topic_tags)

    pg_store.upsert_course(course_id, course_name, description)
    pg_store.upsert_chunks([
        (c['chunk_id'], c['course_id'], c['chunk_type'], c['raw_text'], '[]')
        for c in chunks
    ])
    pg_store.upsert_term_counts(course_id, term_counts)

    embed_chunks = [c for c in chunks if c['chunk_type'] in ('topic', 'definition')]
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

    New topics are extracted and labeled, new chunks are embedded and stored alongside
    existing ones, term counts are accumulated (not replaced), and all similarity scores
    for this course are recomputed from the full combined dataset.
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


def _run_append_inner(course_id: str, course_name: str, raw_text: str, api_key: str | None = None):
    from google import genai
    llm = genai.Client(api_key=api_key)

    # ── Step 1: ONE LLM call — full analysis of the new material ─────────────────
    print(f"[append] Step 1/4 — LLM: full analysis of new material …")
    analysis    = analyze_course(course_id, course_name, raw_text, client=llm)
    topics      = analysis['topics']
    print(f"[append]   ✓ {len(topics)} topics extracted and labeled in one call")

    topic_names = [t['name'] for t in topics]
    topic_tags  = {t['name']: t.get('tags', []) for t in topics}

    for t in topics:
        pg_store.upsert_topic_category(course_id, t['name'], t['categories'], t.get('tags', []))

    # ── Step 2: Stack — chunk new material, embed, accumulate term counts ─────────
    print(f"[append] Step 2/4 — Chunking + embedding new material …")
    local_defs = {f"{course_id}: {t['name']}": t['description'] for t in topics}
    with open(config.DEFINITIONS_PATH) as f:
        global_defs = json.load(f)
    merged_defs = {**global_defs, **local_defs}

    course_dict = {
        'id': course_id, 'name': course_name,
        'description': raw_text[:2000],
        'objectives': [], 'topics': topic_names, 'definitions': [],
    }
    chunks, term_counts = chunk_course(course_dict, merged_defs, topic_tags=topic_tags)

    # Prefix chunk IDs with a timestamp so they don't collide with existing chunks

    ts = int(time.time())
    safe_id = course_id.replace(' ', '_').replace('/', '_')
    for c in chunks:
        c['chunk_id'] = c['chunk_id'].replace(f"{safe_id}__", f"{safe_id}_app{ts}__", 1)

    pg_store.upsert_chunks([
        (c['chunk_id'], c['course_id'], c['chunk_type'], c['raw_text'], '[]')
        for c in chunks
    ])
    # Accumulate: add new term counts on top of existing ones
    pg_store.accumulate_term_counts(course_id, term_counts)

    embed_chunks = [c for c in chunks if c['chunk_type'] in ('topic', 'definition')]
    if embed_chunks:
        embeddings = encode([c['raw_text'] for c in embed_chunks])
        vs.insert_chunks([
            {'chunk_id': embed_chunks[i]['chunk_id'], 'course_id': course_id,
             'embedding': embeddings[i]}
            for i in range(len(embed_chunks))
        ])
    print(f"[append]   ✓ {len(chunks)} new chunks stored, {len(embed_chunks)} embedded ({vs.BACKEND_NAME})")

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

    print(f"[append] ── Done — {len(chunks)} new chunks added to '{course_name}', "
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
