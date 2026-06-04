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


def clear_cache() -> None:
    """Invalidate the in-process embedding cache (e.g. after re-ingest)."""
    _cache.clear()
