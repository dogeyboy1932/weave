from itertools import combinations
from tqdm import tqdm

import config
from scoring.jsd               import lex_sim as _lex_sim, jsd as _jsd
from scoring.vector_similarity import sem_sim as _sem_sim
from scoring.driving_terms     import compute_idf, driving_terms as _driving_terms
from scoring.category_scorer   import (
    compute_category_jsd, compute_non_obvious,
    course_category_vector,
)
from storage.postgres import store as pg_store


_SEM_FLOOR = config.SEM_FLOOR  # SciNCL cosine baseline for random eng-text pairs


def _calibrate_sem(raw: float) -> float:
    """
    Stretch SciNCL cosine similarity so that the ~0.35 baseline maps to 0.
    Without calibration, all engineering pairs score 0.40–0.80 and the graph
    becomes fully connected. With calibration, true outliers separate clearly.
    """
    return max(0.0, (raw - _SEM_FLOOR) / (1.0 - _SEM_FLOOR))


def _hybrid(lex: float, sem: float, cat_jsd: float | None) -> float:
    """
    Weighted combination of lex, calibrated_sem, and category similarity.

    Weights (ALPHA_LEX, ALPHA_SEM, ALPHA_CAT) are read from config and auto-normalized.
    When cat_jsd is unavailable, ALPHA_CAT is redistributed proportionally to lex and sem.
    """
    w_lex = config.ALPHA_LEX
    w_sem = config.ALPHA_SEM
    w_cat = config.ALPHA_CAT
    cal   = _calibrate_sem(sem)

    if cat_jsd is None:
        total = w_lex + w_sem
        return (w_lex * lex + w_sem * cal) / total
    else:
        cat_sim = 1.0 - cat_jsd
        total = w_lex + w_sem + w_cat
        return (w_lex * lex + w_sem * cal + w_cat * cat_sim) / total


def score_pair(
    course_a: str,
    course_b: str,
    lms: dict,
    idf: dict,
    all_term_counts: dict | None = None,
    category_vecs: dict | None = None,
) -> dict:
    """
    Compute hybrid similarity for one pair and persist results.

    Lexical:      TF-IDF cosine (sparse term overlap).
    Semantic:     SciNCL cosine (raw, no floor calibration).
    Category:     1 - category_JSD (same-domain pairs score higher in hybrid).
    Hybrid:       normalized weighted sum of the three components.
    Non-obvious:  sem × category_JSD  (high only when similar meaning, different domain).
    """
    raw_a = all_term_counts.get(course_a) if all_term_counts else {}
    raw_b = all_term_counts.get(course_b) if all_term_counts else {}
    lm_a  = lms[course_a]
    lm_b  = lms[course_b]

    lex   = _lex_sim(raw_a, raw_b, idf)
    j     = _jsd(lm_a, lm_b)
    sem   = _sem_sim(course_a, course_b)
    terms = _driving_terms(lm_a, lm_b, idf, raw_a, raw_b)

    cat_jsd = None
    non_obvious = None
    if category_vecs and course_a in category_vecs and course_b in category_vecs:
        cat_jsd = compute_category_jsd(category_vecs[course_a], category_vecs[course_b])
        non_obvious = compute_non_obvious(sem, cat_jsd)

    final = _hybrid(lex, sem, cat_jsd)

    pg_store.upsert_similarity(
        course_a, course_b, final, lex, sem, j, terms,
        category_jsd=cat_jsd, non_obvious_score=non_obvious,
    )

    return {
        'final': final, 'lex': lex, 'sem': sem, 'jsd': j, 'terms': terms,
        'category_jsd': cat_jsd, 'non_obvious': non_obvious,
    }


def score_all_pairs(lms: dict, all_term_counts: dict) -> int:
    """
    Compute and persist similarity for every N*(N-1)/2 course pair.
    Returns the number of edges that cleared MIN_SCORE (used downstream by
    scoring.graph_compute.recompute_graph_stats to colour and size nodes).
    """
    course_ids = list(lms.keys())
    idf        = compute_idf(all_term_counts)

    # Load category distributions once — null-safe, falls back to no-op
    category_vecs = {}
    for cid in course_ids:
        dists = pg_store.get_topic_categories_for_course(cid)
        if dists:
            category_vecs[cid] = course_category_vector(dists)
    if category_vecs:
        print(f"  Category vectors loaded for {len(category_vecs)}/{len(course_ids)} courses.")

    pairs      = list(combinations(course_ids, 2))
    edge_count = 0

    print(f"Scoring {len(pairs)} pairs across {len(course_ids)} courses …")
    for course_a, course_b in tqdm(pairs, unit="pair"):
        result = score_pair(
            course_a, course_b, lms, idf, all_term_counts,
            category_vecs=category_vecs if category_vecs else None,
        )
        if result['final'] >= config.MIN_SCORE:
            edge_count += 1

    print(f"Done — {edge_count} edges above MIN_SCORE={config.MIN_SCORE}")
    return edge_count
