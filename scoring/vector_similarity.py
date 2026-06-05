"""
Semantic similarity via SciNCL embeddings.

Original implementation issued one ANN query per chunk per pair (~100 round-trips
per pair). That's fine over a UNIX socket to local Milvus but catastrophic over
the internet to a managed pgvector. This version:

  1. Fetches each course's embeddings exactly ONCE (process-wide cache).
  2. Computes pairwise cosine similarity as a single numpy matmul.

For 33 courses × ~50 chunks each that's 33 SELECTs total + pure CPU after.
"""
import numpy as np
from storage.vectors.store import get_embeddings_for_course

_cache: dict[str, np.ndarray | None] = {}


def _matrix(course_id: str) -> np.ndarray | None:
    """Return (n_chunks, 768) matrix for a course, or None if no embeddings."""
    if course_id not in _cache:
        embs = get_embeddings_for_course(course_id)
        _cache[course_id] = (
            np.stack(embs).astype(np.float32) if embs else None
        )
    return _cache[course_id]


def sem_sim(course_a_id: str, course_b_id: str) -> float:
    """
    Symmetric semantic similarity via best-match averaging.

    For each chunk in A, find its nearest neighbour in B → mean → directed(A→B).
    Symmetrise: 0.5 * (directed(A→B) + directed(B→A)).

    Embeddings are L2-normalised at encode time, so cosine == dot product.
    Returns a value in [0, 1] (negative scores clamped to 0).
    """
    A = _matrix(course_a_id)
    B = _matrix(course_b_id)

    if A is None or B is None:
        return 0.0

    sims = A @ B.T  # (na, nb) cosine similarities

    scores_ab = sims.max(axis=1)  # best B-match for each A-chunk
    scores_ba = sims.max(axis=0)  # best A-match for each B-chunk

    directed_ab = float(np.mean(np.clip(scores_ab, 0.0, 1.0)))
    directed_ba = float(np.mean(np.clip(scores_ba, 0.0, 1.0)))

    return 0.5 * (directed_ab + directed_ba)


def _topic_name(raw_text: str) -> str:
    """
    Recover the topic name from a variant-C topic chunk.

    The chunker writes topic chunks as "{topic}: {definition}. Related
    concepts: ...", so the name is everything before the first ": ".
    Falls back to the whole string if no colon is present.
    """
    if not raw_text:
        return "?"
    head = raw_text.split(": ", 1)[0].strip()
    return head or "?"


def shared_topics(
    course_a_id: str,
    course_b_id: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Direct pairwise topic overlap between two specific courses — the "why
    similar" drill-down.

    Computes the full topic-to-topic cosine matrix between A and B and returns
    the strongest cross-course pairs, deduplicated so each topic appears at most
    once (one best partner per topic). This is intentionally NOT served from the
    global `topic_similarity` table: that table keeps only each topic's global
    top-k neighbours, which are dominated by same-domain near-duplicates and so
    discard exactly the cross-domain bridges this view exists to surface
    (e.g. ECE 205 "Ohm's law" ↔ TAM 212 "Newton's laws" at 0.90).

    Returns [{topic_a, topic_b, score}, ...] sorted by score desc.
    """
    import storage.postgres.store as pg_store
    from storage.vectors.store import get_chunks_with_embeddings

    emb_a = get_chunks_with_embeddings(course_a_id)
    emb_b = get_chunks_with_embeddings(course_b_id)
    if not emb_a or not emb_b:
        return []

    # Topic chunks only (exclude any non-topic chunk types that may be embedded).
    ids_a = [cid for cid in emb_a if "__topic__" in cid]
    ids_b = [cid for cid in emb_b if "__topic__" in cid]
    if not ids_a or not ids_b:
        return []

    names_a = dict(pg_store.get_chunks_for_course(course_a_id))
    names_b = dict(pg_store.get_chunks_for_course(course_b_id))

    A = np.stack([emb_a[c] for c in ids_a]).astype(np.float32)
    B = np.stack([emb_b[c] for c in ids_b]).astype(np.float32)
    M = A @ B.T  # (na, nb), cosine since embeddings are L2-normalised

    # Greedy: take highest cosines first, skip a topic once it's already paired.
    order = np.argsort(M, axis=None)[::-1]
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    out: list[dict] = []
    for flat in order:
        i, j = divmod(int(flat), M.shape[1])
        if i in seen_a or j in seen_b:
            continue
        score = float(M[i, j])
        if score < min_score:
            break
        out.append({
            "topic_a": _topic_name(names_a.get(ids_a[i], "")),
            "topic_b": _topic_name(names_b.get(ids_b[j], "")),
            "score":   round(score, 4),
        })
        seen_a.add(i)
        seen_b.add(j)
        if len(out) >= top_k:
            break
    return out


def _seed_embeddings_bulk(seeds: list[dict]) -> dict[tuple[str, str], np.ndarray]:
    """
    {(course_id, topic_text): stored embedding} for the requested seed topics,
    fetched for ALL seed courses in three queries total (not three per course).

    Maps each embedded chunk back to its clean topic_text via the same
    longest-prefix rule used elsewhere (pg_store._match_topic_text).
    """
    import storage.postgres.store as pg_store
    from storage.postgres.store import _Conn
    from storage.vectors.pgvector_store import _parse_vector

    course_ids = list({s["course_id"] for s in seeds})
    wanted = {(s["course_id"], s["topic_text"]) for s in seeds}
    if not course_ids:
        return {}

    with _Conn() as cur:
        cur.execute(
            "SELECT chunk_id, course_id, raw_text FROM chunks "
            "WHERE course_id = ANY(%s::text[]) AND chunk_type='topic'",
            (course_ids,),
        )
        chunk_rows = cur.fetchall()
        cur.execute(
            "SELECT course_id, topic_text FROM topic_categories "
            "WHERE course_id = ANY(%s::text[])",
            (course_ids,),
        )
        texts_by_course: dict[str, list[str]] = {}
        for cid, tt in cur.fetchall():
            texts_by_course.setdefault(cid, []).append(tt)

        # Resolve each seed topic to its chunk_id, then fetch just those vectors.
        chunk_for_seed: dict[tuple[str, str], str] = {}
        for chunk_id, cid, raw in chunk_rows:
            tt = pg_store._match_topic_text(raw, texts_by_course.get(cid, []))
            if tt is not None and (cid, tt) in wanted:
                chunk_for_seed[(cid, tt)] = chunk_id
        if not chunk_for_seed:
            return {}
        cur.execute(
            "SELECT chunk_id, embedding FROM chunk_embeddings "
            "WHERE chunk_id = ANY(%s::text[])",
            (list(chunk_for_seed.values()),),
        )
        emb_by_chunk = {r[0]: _parse_vector(r[1]) for r in cur.fetchall()}

    return {
        key: emb_by_chunk[cid]
        for key, cid in chunk_for_seed.items()
        if cid in emb_by_chunk
    }


# Cross-domain discovery knobs. A neighbour must be semantically close to a seed
# topic (its query-relevant match) AND land in a divergent category mix to count
# as a non-obvious bridge — this is the project's non_obvious = sem × cat_jsd,
# applied at topic level. SEM_FLOOR weeds out loosely-related topics; JSD_FLOOR
# (the same 0.25 the explainer uses) weeds out same-domain near-duplicates.
_XD_SEM_FLOOR = 0.82
_XD_JSD_FLOOR = 0.25
# A bridge must clear this non_obvious (= sem × jsd) to be shown at all — it trims
# the weak tail (vague queries whose "cross-domain" hits sit at ~0.37-0.45) so the
# list is genuine bridges only. Strong matches run 0.55-0.80.
_XD_NONOBV_FLOOR = 0.45


def cross_domain_matches(
    seeds: list[dict],
    top_k: int = 18,
    per_seed: int = 12,
    sem_floor: float = _XD_SEM_FLOOR,
    jsd_floor: float = _XD_JSD_FLOOR,
    nonobv_floor: float = _XD_NONOBV_FLOOR,
) -> list[dict]:
    """
    Cross-domain (non-obvious) topic matches for a search query — the replacement
    for the old "share any tag with any direct match" phase 2.

    seeds: the query's strong direct matches, each {course_id, topic_text,
    categories, tags, description}. For each seed we take its STORED embedding
    (reliable, unlike a bare-keyword query vector), pull its nearest topic
    neighbours in OTHER courses via pgvector ANN, and keep only those that are
    both semantically close (cosine ≥ sem_floor) and category-divergent
    (JSD ≥ jsd_floor). Ranked by non_obvious = cosine × JSD, deduped to the best
    bridge per target topic, with the seed it connects to attached for the
    explainer. shared_tags is the genuine specific-tag overlap (a clean label,
    not the discovery mechanism).
    """
    import storage.postgres.store as pg_store
    from storage.vectors.store import search_neighbors_batch
    from scoring.category_scorer import (
        course_category_vector, compute_category_jsd, compute_non_obvious,
    )

    # All seed embeddings in ~3 queries.
    seed_emb = _seed_embeddings_bulk(seeds)
    direct_keys = {(s["course_id"], s["topic_text"]) for s in seeds}

    # One ANN round trip for ALL seeds (LATERAL), then resolve every hit in a
    # SINGLE batch (resolve_topic_chunks is two queries regardless of size).
    embedded_seeds = [s for s in seeds
                      if (s["course_id"], s["topic_text"]) in seed_emb]
    neighbours = search_neighbors_batch(
        [(seed_emb[(s["course_id"], s["topic_text"])], s["course_id"])
         for s in embedded_seeds],
        limit=per_seed * 3,
    )
    raw_hits: list[tuple[dict, dict]] = []   # (seed, hit{chunk_id,course_id,score})
    uniq_hits: dict[str, dict] = {}
    for s, hits in zip(embedded_seeds, neighbours):
        for h in hits:
            raw_hits.append((s, h))
            uniq_hits.setdefault(h["chunk_id"], h)

    meta_by_chunk = {r["chunk_id"]: r
                     for r in pg_store.resolve_topic_chunks(list(uniq_hits.values()))}

    best: dict[tuple[str, str], dict] = {}
    seed_vecs = {(s["course_id"], s["topic_text"]): course_category_vector([s.get("categories") or {}])
                 for s in seeds}
    for s, h in raw_hits:
        n = meta_by_chunk.get(h["chunk_id"])
        if n is None:
            continue
        key = (n["course_id"], n["topic_text"])
        if key in direct_keys:
            continue
        sem = float(h["score"])
        if sem < sem_floor:
            continue
        jsd = compute_category_jsd(
            seed_vecs[(s["course_id"], s["topic_text"])],
            course_category_vector([n.get("categories") or {}]))
        if jsd < jsd_floor:
            continue
        non_obv = compute_non_obvious(sem, jsd)
        if non_obv < nonobv_floor:
            continue
        prev = best.get(key)
        if prev and prev["non_obvious_score"] >= non_obv:
            continue
        best[key] = {
            "course_id":          n["course_id"],
            "topic_text":         n["topic_text"],
            "categories":         n["categories"],
            "tags":               n["tags"],
            "description":        n["description"],
            "sem_score":          round(sem, 4),
            "category_jsd":       round(jsd, 4),
            "non_obvious_score":  round(non_obv, 4),
            "shared_tags":        sorted(set(s.get("tags") or []) & set(n.get("tags") or [])),
            "source_course":      s["course_id"],
            "source_topic":       s["topic_text"],
            "source_tags":        s.get("tags") or [],
            "source_categories":  s.get("categories") or {},
            "source_description": s.get("description", ""),
        }

    return sorted(best.values(),
                  key=lambda r: r["non_obvious_score"], reverse=True)[:top_k]


def clear_cache() -> None:
    """Invalidate the in-process embedding cache (e.g. after re-ingest)."""
    _cache.clear()
