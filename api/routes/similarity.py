from fastapi import APIRouter, Header, HTTPException, Query
from typing import Optional
from pydantic import BaseModel
import storage.postgres.store as pg_store
from scoring.category_scorer import course_category_vector, CATEGORIES
from llm.llm_explainer import explain_connection

router = APIRouter()


class SaveExplanationRequest(BaseModel):
    a: str
    b: str
    explanation: str


@router.get("/categories")
def get_course_categories(
    course: str = Query(..., description="Course ID, e.g. ME 340"),
):
    """
    Averaged 8-category probability distribution for a course, computed from
    all its LLM-labeled topics. Use this to understand a course's disciplinary
    composition and compare it to another course's distribution.
    """
    dists = pg_store.get_topic_categories_for_course(course)
    if not dists:
        raise HTTPException(
            404, detail=f"No category data for '{course}'. Run scripts/restore_from_dump.py.",
        )
    vec = course_category_vector(dists)
    return {
        "course_id":   course,
        "topic_count": len(dists),
        "categories":  {cat: round(vec[i], 4) for i, cat in enumerate(CATEGORIES)},
    }


@router.get("")
def get_similarity(
    a: str = Query(..., description="First course ID, e.g. ME 410"),
    b: str = Query(..., description="Second course ID, e.g. ME 310"),
):
    """
    Pre-computed hybrid similarity between two courses.

    - **final_score**: α·lex_score + (1-α)·sem_score
    - **lex_score**: TF-IDF cosine similarity
    - **sem_score**: SciNCL cosine similarity (calibrated)
    - **category_jsd**: Jensen-Shannon divergence between category distributions [0,1]
    - **non_obvious_score**: sem_score × category_jsd — high = non-obvious cross-domain connection
    - **llm_explanation**: Claude-generated explanation (if available)
    """
    result = pg_store.get_similarity(a, b)
    if result is None:
        raise HTTPException(
            404,
            detail=f"No similarity found for '{a}' ↔ '{b}'. "
                   "Run scripts/build_graph.py first.",
        )
    return result


@router.get("/neighbors")
def get_neighbors(
    course: str = Query(..., description="Course ID"),
    top:    int  = Query(10, ge=1, le=50),
    sort:   str  = Query("hybrid", description="Sort by: 'hybrid' or 'non_obvious'"),
):
    """Top-N most similar (or most non-obviously connected) courses."""
    if sort not in ("hybrid", "non_obvious"):
        raise HTTPException(400, detail="sort must be 'hybrid' or 'non_obvious'")
    neighbors = pg_store.get_neighbors(course, top, sort=sort)
    if not neighbors:
        raise HTTPException(404, detail=f"No neighbors found for '{course}'")
    return {"course": course, "neighbors": neighbors, "sort": sort}


@router.get("/non-obvious")
def get_non_obvious(
    top:     int   = Query(20, ge=1, le=200, description="Number of pairs to return"),
    min_sem: float = Query(0.0, ge=0.0, le=1.0, description="Minimum semantic score filter"),
):
    """
    Top non-obvious course connections ranked by non_obvious_score = sem × category_JSD.

    Pairs with high semantic similarity but very different category distributions represent
    cross-domain connections that are structurally similar but not obviously related.
    Requires scripts/restore_from_dump.py (or scripts/build_graph.py once
    topic_categories is populated) to have filled similarity_cache.
    """
    pairs = pg_store.get_top_non_obvious(top=top, min_sem=min_sem)
    if not pairs:
        raise HTTPException(
            404,
            detail="No non-obvious pairs found. Run scripts/restore_from_dump.py.",
        )
    return {"pairs": pairs, "count": len(pairs)}


@router.post("/explanation")
def save_explanation(body: SaveExplanationRequest):
    """Save a manually-edited explanation for a course pair."""
    result = pg_store.get_similarity(body.a, body.b)
    if result is None:
        raise HTTPException(404, detail=f"No similarity record for '{body.a}' ↔ '{body.b}'")
    pg_store.update_llm_explanation(body.a, body.b, body.explanation)
    return {"ok": True}


@router.get("/explain")
def get_explanation(
    a: str = Query(..., description="First course ID"),
    b: str = Query(..., description="Second course ID"),
    force: bool = Query(False, description="Force regeneration even if cached"),
    x_api_key: Optional[str] = Header(None, description="Gemini API key for on-demand generation"),
):
    """
    Return the cached LLM explanation for a course pair.

    If no explanation is cached and `X-Api-Key` is provided, generates one using
    the caller's Gemini API key and caches the result. Without a key, returns an
    empty explanation — the UI shows a 'Generate' button instead.
    """
    result = pg_store.get_similarity(a, b)
    if result is None:
        raise HTTPException(404, detail=f"No similarity record for '{a}' ↔ '{b}'")

    if not result.get("llm_explanation") or force:
        if not x_api_key:
            # No cached explanation and no key — let the UI show the Generate button
            return {
                "course_a":          result["course_a"],
                "course_b":          result["course_b"],
                "sem_score":         result["sem_score"],
                "category_jsd":      result["category_jsd"],
                "non_obvious_score": result["non_obvious_score"],
                "llm_explanation":   "",
                "generated":         False,
            }

        ca = result["course_a"]
        cb = result["course_b"]
        ctx_a = pg_store.get_course_explain_context(ca)
        ctx_b = pg_store.get_course_explain_context(cb)

        # Enrich topic list with descriptions from topic_definitions.json
        import json as _json
        from pathlib import Path
        import config as _cfg
        defs: dict = {}
        defs_path = Path(_cfg.DEFINITIONS_PATH)
        if defs_path.exists():
            with open(defs_path) as f:
                defs = _json.load(f)
        descs_a = {t: defs.get(f"{ca}: {t}", "") for t in ctx_a["topics"]}
        descs_b = {t: defs.get(f"{cb}: {t}", "") for t in ctx_b["topics"]}

        try:
            from google import genai
            llm = genai.Client(api_key=x_api_key)
            llm_result = explain_connection(
                course_a=ca, topics_a=ctx_a["topics"],
                course_b=cb, topics_b=ctx_b["topics"],
                sem_score=result["sem_score"] or 0.0,
                cat_jsd=result["category_jsd"] or 0.0,
                non_obvious_score=result["non_obvious_score"] or 0.0,
                name_a=ctx_a["name"], name_b=ctx_b["name"],
                cats_a=ctx_a["categories"], cats_b=ctx_b["categories"],
                topic_descs_a=descs_a, topic_descs_b=descs_b,
                client=llm,
            )
        except Exception as e:
            raise HTTPException(502, detail=str(e))
        parts = []
        for key, label in [("connection",  "Connection"),
                            ("in_a",        "In " + ca),
                            ("in_b",        "In " + cb),
                            ("surprise",    "Why non-obvious")]:
            val = (llm_result.get(key) or "").strip()
            if val:
                parts.append(f"{label}: {val}")
        explanation = "\n".join(parts)
        if explanation:
            pg_store.update_llm_explanation(ca, cb, explanation)
        result["llm_explanation"] = explanation

    return {
        "course_a":          result["course_a"],
        "course_b":          result["course_b"],
        "sem_score":         result["sem_score"],
        "category_jsd":      result["category_jsd"],
        "non_obvious_score": result["non_obvious_score"],
        "llm_explanation":   result["llm_explanation"] or "",
        "generated":         True,
    }
