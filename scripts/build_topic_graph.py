"""
Pre-compute topic-to-topic semantic similarity using pgvector ANN search.

For every labeled topic, finds the top-K most similar topics from other courses
using SciNCL cosine similarity (same floor calibration as course-level sem_sim).
Results are stored in the topic_similarity table.

Run from akhil_app/ directory:
    python scripts/build_topic_graph.py
    python scripts/build_topic_graph.py --top-k 10 --min-score 0.1
    python scripts/build_topic_graph.py --course "CS 521"   # one course only

Cost: $0 — pure pgvector ANN (HNSW indexed), no LLM calls.
Time: ~2-5 min for all 1,648 topics.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.postgres import store as pg_store
from storage.vectors import store as vs



def _build_chunk_to_topic(course_id: str, topic_texts: list[str]) -> dict:
    """
    Build {chunk_id: topic_text} for a course by matching chunk raw_text
    against known topic_texts from topic_categories.
    """
    chunks = pg_store.get_chunks_for_course(course_id)
    mapping = {}
    for chunk_id, raw_text in chunks:
        if '[SEP]' not in raw_text:
            continue
        after_sep = raw_text.split('[SEP]', 1)[1].strip()
        after_sep_lower = after_sep.lower()
        for topic_text in topic_texts:
            topic_lower = topic_text.lower()
            if after_sep_lower == topic_lower or after_sep_lower.startswith(topic_lower + ':'):
                mapping[chunk_id] = topic_text
                break
    return mapping


def build_for_courses(course_ids: list[str], top_k: int, min_score: float):
    pg_store.init_schema()

    # Pre-load all (course_id, topic_text) → used for reverse mapping ANN results
    print("Loading topic registry…")
    all_topic_texts: dict[str, list[str]] = {}
    for cid in course_ids:
        texts = pg_store.get_topic_texts_for_course(cid, limit=500)
        if texts:
            all_topic_texts[cid] = texts

    if not all_topic_texts:
        print("No labeled topics found. Run label_categories.py first.")
        return

    # Build a global chunk_id → (course_id, topic_text) lookup for ANN result mapping
    print("Building chunk→topic lookup across all courses…")
    global_chunk_map: dict[str, tuple[str, str]] = {}
    all_course_ids = list(pg_store.get_all_term_counts().keys())
    for cid in all_course_ids:
        texts = all_topic_texts.get(cid) or pg_store.get_topic_texts_for_course(cid, limit=500)
        if not texts:
            continue
        mapping = _build_chunk_to_topic(cid, texts)
        for chunk_id, topic_text in mapping.items():
            global_chunk_map[chunk_id] = (cid, topic_text)

    print(f"  Mapped {len(global_chunk_map)} chunks to topic texts")

    total_stored = 0
    for course_id in course_ids:
        topic_texts = all_topic_texts.get(course_id)
        if not topic_texts:
            print(f"  [{course_id}] no labeled topics — skipping")
            continue

        # Get {chunk_id: embedding} for this course
        chunk_embeddings = vs.get_chunks_with_embeddings(course_id)
        # Restrict to topic chunks that have a known topic_text
        local_map = _build_chunk_to_topic(course_id, topic_texts)

        print(f"  [{course_id}] {len(local_map)} topics to process…", flush=True)
        rows = []

        for chunk_id, topic_text in local_map.items():
            embedding = chunk_embeddings.get(chunk_id)
            if embedding is None:
                continue

            hits = vs.search_global_excluding(embedding, course_id, limit=top_k)
            for hit in hits:
                raw_score = hit["score"]
                if raw_score < min_score:
                    continue
                other = global_chunk_map.get(hit["chunk_id"])
                if other is None:
                    continue
                other_course, other_topic = other
                if other_course == course_id:
                    continue
                rows.append((course_id, topic_text, other_course, other_topic, raw_score))

        if rows:
            # Deduplicate: keep highest score per (course_a, topic_a, course_b, topic_b)
            best: dict[tuple, float] = {}
            for ca, ta, cb, tb, score in rows:
                key = (ca, ta, cb, tb)
                if score > best.get(key, -1):
                    best[key] = score
            rows = [(ca, ta, cb, tb, score) for (ca, ta, cb, tb), score in best.items()]
            pg_store.upsert_topic_similarities(rows)
            total_stored += len(rows)
            print(f"    ✓ {len(rows)} topic pairs stored")
        else:
            print(f"    (no pairs above threshold)")

    print(f"\nDone — {total_stored} topic similarity pairs stored in topic_similarity.")


def main():
    parser = argparse.ArgumentParser(description="Build topic-to-topic similarity graph")
    parser.add_argument("--top-k",    type=int,   default=10,  help="Similar topics to find per topic")
    parser.add_argument("--min-score", type=float, default=0.70, help="Min raw SciNCL cosine score to store")
    parser.add_argument("--course",   type=str,   default=None, help="Process only this course ID")
    args = parser.parse_args()

    pg_store.init_schema()

    if args.course:
        course_ids = [args.course]
    else:
        course_ids = list(pg_store.get_all_term_counts().keys())

    print(f"Building topic graph for {len(course_ids)} courses "
          f"(top_k={args.top_k}, min_score={args.min_score})…\n")
    build_for_courses(course_ids, top_k=args.top_k, min_score=args.min_score)


if __name__ == "__main__":
    main()
