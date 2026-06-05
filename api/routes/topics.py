import time
from collections import OrderedDict
from threading import Lock
from typing import Optional
from fastapi import APIRouter, Query, Header, HTTPException, Response
from pydantic import BaseModel
import storage.postgres.store as pg_store
from storage.vectors import store as vector_store
from storage.vectors.encoder import encode
from scoring.chunker import tokenize
from scoring.vector_similarity import cross_domain_matches
from llm.llm_explainer import explain_topic_connection

router = APIRouter()


# ── Tiny LRU for /topics/search responses ───────────────────────────────────
# Same query within _SEARCH_TTL seconds returns the cached payload instantly
# instead of re-running the full Postgres pipeline. Cleared on PDF ingest
# (see invalidate_search_cache below).
_SEARCH_CACHE: "OrderedDict[tuple[str, int], tuple[float, dict]]" = OrderedDict()
_SEARCH_LOCK  = Lock()
_SEARCH_MAX   = 128
_SEARCH_TTL   = 300.0  # seconds


def _cache_get(key: tuple[str, int]) -> Optional[dict]:
    with _SEARCH_LOCK:
        hit = _SEARCH_CACHE.get(key)
        if not hit:
            return None
        ts, payload = hit
        if time.time() - ts > _SEARCH_TTL:
            _SEARCH_CACHE.pop(key, None)
            return None
        _SEARCH_CACHE.move_to_end(key)
        return payload


def _cache_put(key: tuple[str, int], payload: dict) -> None:
    with _SEARCH_LOCK:
        _SEARCH_CACHE[key] = (time.time(), payload)
        _SEARCH_CACHE.move_to_end(key)
        while len(_SEARCH_CACHE) > _SEARCH_MAX:
            _SEARCH_CACHE.popitem(last=False)


def invalidate_search_cache() -> None:
    """Call after any write to topic_categories / topic_similarity."""
    with _SEARCH_LOCK:
        _SEARCH_CACHE.clear()


class ExplainRequest(BaseModel):
    source_course: str
    source_topic:  str
    target_course: str
    target_topic:  str
    shared_tags:   list[str] = []
    source_tags:   list[str] = []
    target_tags:   list[str] = []
    force:         bool = False
    feedback:      str = ""   # optional steer for a regeneration


class SignRequest(BaseModel):
    source_course: str
    source_topic:  str
    target_course: str
    target_topic:  str
    signed_by:     str


class SaveTopicExplanationRequest(BaseModel):
    source_course: str
    source_topic:  str
    target_course: str
    target_topic:  str
    explanation:   str


@router.get("/course-info")
def course_info(course: str = Query(..., description="Course ID")):
    """
    Returns full tag summary and topic list for a course — used by the detail panel.
    Tags are aggregated across all topics, sorted by frequency.
    """
    summary = pg_store.get_course_tag_summary(course)

    # Enrich topics with descriptions parsed from the embedded topic chunks.
    descriptions = pg_store.get_topic_descriptions_for_course(course)
    topics_enriched = []
    for t in summary["topics"]:
        topics_enriched.append({
            **t,
            "description": descriptions.get(t['topic'], ""),
        })

    return {
        "course_id": course,
        "tags":      summary["tags"],
        "topics":    topics_enriched,
    }


@router.get("/search")
def search_topics(
    response: Response,
    q:   str = Query(..., min_length=2, description="Topic concept or phrase"),
    top: int = Query(40, ge=1, le=100),
):
    """
    Lexical-primary hybrid topic search.

    Phase 1 — Lexical (primary): tokenize the query and match each token against
    topic text + tags. Precise and fast, and the form users expect from a search
    box: "machine" → ME 370 Machine dynamics / ME 371 Machine components, not
    centroid noise. Tokenizing also fixes the old whole-phrase ILIKE bug where
    "machine learning" matched nothing (now machine ∪ learning both hit).

    Phase 1-fallback — Semantic: ONLY when lexical finds nothing, embed the query
    with SciNCL and run pgvector ANN over the topic vectors. SciNCL is a scientific
    *document* model — its cosine is near-uniform for bare keywords (the top-50 of
    a 938-vector search span ~0.04), so it returns relevant results only for
    abstract-shaped phrases and is a last resort, not the primary signal.

    Phase 2 — Related: topics from OTHER courses sharing exact tags with the
    phase-1 hits, e.g. ECE 210's RLC Circuit ⇔ ME 340's Spring-Mass-Damper via
    "second-order-ode" / "natural-frequency".

    Performance: one lexical query for phase 1, then every cross-row lookup
    (per-topic neighbors, per-course descriptions, per-pair course similarity) is
    batched. Identical queries are served from a 5-minute in-process LRU cache.
    """
    # Results depend on live data + search code; never let the browser serve a
    # stale cached payload (this is what made an old semantic response linger).
    response.headers["Cache-Control"] = "no-store"

    cache_key = (q.strip().lower(), top)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Phase 1: lexical token match (primary). Fall back to SciNCL ANN only when
    # there is no lexical hit at all.
    tokens = tokenize(q) or [q.strip().lower()]
    matches = pg_store.search_topics_tokens(tokens, q.strip(), top)
    search_mode = "lexical"
    if not matches:
        qvec = encode([q])[0]
        hits = vector_store.search_topics_semantic(qvec, limit=top)
        matches = pg_store.resolve_topic_chunks(hits)  # cosine-ranked, has descriptions
        for m in matches:
            m['matched_by']   = 'semantic'
            m['matched_tags'] = []
        search_mode = "semantic"
    # Descriptions for every direct-match course in one batched call.
    descs = pg_store.get_topic_descriptions_bulk([m['course_id'] for m in matches])
    for m in matches:
        m.setdefault('description', descs.get(m['course_id'], {}).get(m['topic_text'], ''))

    # Phase 2: cross-domain (non-obvious) connections. Seeded from the strongest
    # direct matches, we follow each seed topic's EMBEDDING to its nearest
    # neighbours in other courses and keep only those that are both semantically
    # close and category-divergent (non_obvious = sem × cat_jsd). This replaces
    # the old "share any tag with any direct match" union, which over-connected
    # everything through a handful of broad tags.
    seeds = [
        {'course_id': m['course_id'], 'topic_text': m['topic_text'],
         'categories': m.get('categories') or {}, 'tags': m.get('tags') or [],
         'description': m.get('description', '')}
        for m in matches[:6]
    ]
    related = cross_domain_matches(seeds)
    for r in related:
        r['explanation'] = None
        r['signed_by']   = None

    # Batched neighbor lookup for direct matches only (one query for all of them).
    if matches:
        neighbor_map = pg_store.get_similar_topics_bulk(
            [(m['course_id'], m['topic_text']) for m in matches],
            per_limit=6,
        )
        for m in matches:
            m['topic_matches'] = neighbor_map.get((m['course_id'], m['topic_text']), [])

    # All unique courses across both phases
    seen: set = set()
    courses: list[str] = []
    for m in matches + related:
        if m['course_id'] not in seen:
            seen.add(m['course_id'])
            courses.append(m['course_id'])

    # Batched pairwise similarities (one query for all C*(C-1)/2 pairs).
    course_pair_keys: list[tuple[str, str]] = [
        (ca, cb) for i, ca in enumerate(courses) for cb in courses[i + 1:]
    ]
    sim_map = pg_store.get_similarities_bulk(course_pair_keys)
    pairs = list(sim_map.values())

    payload = {
        "query":        q,
        "mode":         search_mode,   # 'lexical' (primary) or 'semantic' (fallback)
        "matches":      matches,   # direct text/tag hits
        "related":      related,   # cross-domain via shared tags
        "courses":      courses,
        "course_pairs": pairs,
    }
    _cache_put(cache_key, payload)
    return payload


@router.post("/explain")
def explain_topic(
    body: ExplainRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    Generate (or retrieve cached) LLM explanation for a cross-domain topic pair.
    Pass X-Api-Key header with a Gemini key to generate. Omit to retrieve only cached.
    Pass force=true to regenerate even if a cached explanation exists.
    """
    cached = pg_store.get_topic_explanation(
        body.source_course, body.source_topic,
        body.target_course, body.target_topic,
    )
    if cached and cached.get("explanation") and not body.force:
        return {**cached, "cached": True}

    if not x_api_key:
        if cached:
            return {**cached, "cached": True}
        return {"explanation": None, "signed_by": None, "cached": False}

    # Fetch full context from DB — descriptions, categories, course names
    src_ctx = pg_store.get_topic_context(body.source_course, body.source_topic)
    tgt_ctx = pg_store.get_topic_context(body.target_course, body.target_topic)

    # Topic descriptions parsed from the embedded topic chunks (falls back to the
    # course description when a topic has no definition).
    src_desc = pg_store.get_topic_descriptions_for_course(body.source_course).get(
        body.source_topic, "") or src_ctx.get("course_description", "")
    tgt_desc = pg_store.get_topic_descriptions_for_course(body.target_course).get(
        body.target_topic, "") or tgt_ctx.get("course_description", "")

    # Use shared_tags from the request, fall back to overlap between both tag lists
    shared = body.shared_tags
    if not shared:
        shared = list(set(src_ctx["tags"]) & set(tgt_ctx["tags"]))

    # Determine if this is a non-obvious connection by category divergence
    def _jsd_cats(c1: dict, c2: dict) -> float:
        import math
        cats = set(c1) | set(c2)
        if not cats:
            return 0.0
        p = [c1.get(k, 0) for k in cats]
        q = [c2.get(k, 0) for k in cats]
        sp, sq = sum(p), sum(q)
        if sp == 0 or sq == 0:
            return 0.0
        p = [x / sp for x in p]
        q = [x / sq for x in q]
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        def _kl(a, b):
            return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)
        return (_kl(p, m) + _kl(q, m)) / 2

    cat_jsd = _jsd_cats(src_ctx.get("categories", {}), tgt_ctx.get("categories", {}))
    is_non_obvious = cat_jsd >= 0.25 or body.source_course != body.target_course

    try:
        from google import genai
        client = genai.Client(api_key=x_api_key)
        result = explain_topic_connection(
            source_course=body.source_course,
            source_topic=body.source_topic,
            target_course=body.target_course,
            target_topic=body.target_topic,
            shared_tags=shared,
            source_tags=src_ctx["tags"],
            target_tags=tgt_ctx["tags"],
            source_description=src_desc,
            target_description=tgt_desc,
            source_categories=src_ctx["categories"],
            target_categories=tgt_ctx["categories"],
            source_course_name=src_ctx.get("course_name"),
            target_course_name=tgt_ctx.get("course_name"),
            is_non_obvious=is_non_obvious,
            feedback=body.feedback,
            client=client,
        )
    except Exception as e:
        raise HTTPException(502, detail=str(e))

    parts = []
    for key, label in [("connection",  "Connection"),
                        ("in_a",        "In " + body.source_course),
                        ("in_b",        "In " + body.target_course),
                        ("surprise",    "Why non-obvious")]:
        val = (result.get(key) or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    explanation = "\n".join(parts)

    signed_by = cached.get("signed_by") if cached else None
    pg_store.upsert_topic_explanation(
        body.source_course, body.source_topic,
        body.target_course, body.target_topic,
        explanation, signed_by,
    )
    return {"explanation": explanation, "signed_by": signed_by, "cached": False}


@router.post("/save-explanation")
def save_topic_explanation(body: SaveTopicExplanationRequest):
    """Save a manually-edited explanation for a topic pair (overwrites LLM-generated)."""
    cached = pg_store.get_topic_explanation(
        body.source_course, body.source_topic,
        body.target_course, body.target_topic,
    )
    signed_by = cached.get("signed_by") if cached else None
    pg_store.upsert_topic_explanation(
        body.source_course, body.source_topic,
        body.target_course, body.target_topic,
        body.explanation, signed_by,
    )
    return {"ok": True}


@router.post("/sign")
def sign_topic_explanation(body: SignRequest):
    """Attach a professor signature to a topic connection explanation."""
    if not body.signed_by.strip():
        raise HTTPException(400, detail="signed_by cannot be empty")
    pg_store.sign_topic_explanation(
        body.source_course, body.source_topic,
        body.target_course, body.target_topic,
        body.signed_by.strip(),
    )
    return {"ok": True, "signed_by": body.signed_by.strip()}
