"""
Restore the full SIIP dataset from data/courses_dump.json.

This is the recovery path when the cloud Postgres has been wiped (e.g. after
the free-tier instance was reclaimed). The dump is the canonical snapshot of
all LLM-enriched data — descriptions, objectives, topic definitions, 8-bin
category distributions, cross-domain tags — for all 35 courses / 938 topics.

What this script does:
  1. courses        — upsert metadata from the dump
  2. chunks         — rebuild via scoring.chunker (pure string operations)
  3. term_counts    — derived by the chunker
  4. chunk_embeddings — SciNCL encode (local ML, no Gemini)
  5. topic_categories — write the dump's categories + tags as-is
  6. similarity_cache — recompute every pair (deterministic math, no LLM)
  7. topic_similarity — topic-to-topic ANN (pure pgvector)
  8. courses.community, courses.pagerank — Louvain + PageRank in-process

Zero Gemini calls. The only thing not restored is the cache of on-demand LLM
explanations; those regrow naturally as users click "Generate explanation".

Run:
    python scripts/restore_from_dump.py
    python scripts/restore_from_dump.py --in data/courses_dump.json
    python scripts/restore_from_dump.py --skip-topic-graph   # quick path

After running locally, your Supabase DB matches the live URL.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
from scoring.chunker        import chunk_course
from scoring                import graph_compute
from scoring.hybrid_scorer  import score_all_pairs
from scoring.language_model import build_all_lms
from storage.postgres       import store as pg_store
from storage.vectors        import store as vs
from storage.vectors.encoder import encode


def _course_dict_from_dump(cid: str, entry: dict) -> dict:
    """Translate dump shape → the dict shape scoring.chunker.chunk_course expects."""
    return {
        'id':          cid,
        'name':        entry.get('name', cid),
        'description': entry.get('description', ''),
        'objectives':  entry.get('objectives', []),
        'topics':      [t['name'] for t in entry.get('topics', [])],
        'definitions': [],
    }


def _topic_tags_from_entry(entry: dict) -> dict[str, list[str]]:
    """{topic_name: [tag, ...]} for chunker's topic_tags arg."""
    return {
        t['name']: t.get('tags', [])
        for t in entry.get('topics', [])
        if t.get('name')
    }


def _topic_defs_from_dump(dump: dict) -> dict:
    """Build the {"<COURSE>: <topic>": "<definition>"} map the chunker reads."""
    out: dict = {}
    for cid, entry in dump.items():
        for t in entry.get('topics', []):
            name = t.get('name', '').strip()
            desc = t.get('description', '').strip()
            if name and desc:
                out[f"{cid}: {name}"] = desc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=str(config.COURSES_DUMP_PATH))
    ap.add_argument("--skip-topic-graph", action="store_true",
                    help="Skip the topic-to-topic ANN backfill (faster).")
    args = ap.parse_args()

    dump_path = Path(args.infile)
    if not dump_path.exists():
        print(f"ERROR: {dump_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(dump_path) as f:
        dump: dict = json.load(f)

    print(f"Loaded dump: {len(dump)} courses, "
          f"{sum(len(v.get('topics', [])) for v in dump.values())} topics")

    print("\n[1/8] Initialising schema + vector collection …")
    pg_store.init_schema()
    vs.get_or_create_collection()

    topic_defs = _topic_defs_from_dump(dump)

    print(f"[2/8] Upserting courses + chunks + term counts + categories …")
    for cid, entry in tqdm(dump.items(), unit="course"):
        # 1. course row
        pg_store.upsert_course(
            course_id   = cid,
            name        = entry.get('name', cid),
            description = entry.get('description', ''),
            prereqs     = entry.get('prereqs', ''),
            credits     = int(entry.get('credits', 0) or 0),
            sequence    = 0,
            course_type = entry.get('course_type', ''),
            instructors = entry.get('instructors', ''),
        )

        # 2. chunks + 3. term counts
        course_dict = _course_dict_from_dump(cid, entry)
        chunks, term_counts = chunk_course(
            course_dict, topic_defs, topic_tags=_topic_tags_from_entry(entry)
        )
        pg_store.upsert_chunks([
            (c['chunk_id'], c['course_id'], c['chunk_type'], c['raw_text'], '[]')
            for c in chunks
        ])
        pg_store.upsert_term_counts(cid, term_counts)

        # 5. topic_categories — categories + tags as recorded in the dump
        for t in entry.get('topics', []):
            pg_store.upsert_topic_category(
                course_id   = cid,
                topic_text  = t['name'],
                categories  = t.get('categories', {}),
                tags        = t.get('tags', []),
            )

    print("[3/8] Encoding chunks with SciNCL (local — no Gemini calls) …")
    for cid, entry in tqdm(dump.items(), unit="course"):
        course_dict = _course_dict_from_dump(cid, entry)
        chunks, _ = chunk_course(
            course_dict, topic_defs, topic_tags=_topic_tags_from_entry(entry)
        )
        embed_chunks = [c for c in chunks if c['chunk_type'] in ('topic', 'definition')]
        if not embed_chunks:
            continue
        embeddings = encode([c['raw_text'] for c in embed_chunks])
        vs.insert_chunks([
            {
                'chunk_id':  embed_chunks[i]['chunk_id'],
                'course_id': cid,
                'embedding': embeddings[i],
            }
            for i in range(len(embed_chunks))
        ])

    print("[4/8] Computing pairwise similarity (lex + sem + category_jsd + non_obvious) …")
    all_counts = pg_store.get_all_term_counts()
    lms        = build_all_lms(all_counts)
    edge_count = score_all_pairs(lms, all_counts)
    print(f"      → {edge_count} pairs above MIN_SCORE={config.MIN_SCORE}")

    if not args.skip_topic_graph:
        print("[5/8] Building topic-to-topic similarity (pure ANN) …")
        from scripts.build_topic_graph import build_for_courses
        build_for_courses(list(dump.keys()), top_k=10, min_score=0.70)

    print("[6/8] Recomputing community clusters + PageRank …")
    summary = graph_compute.recompute_graph_stats()
    print(f"      → {summary['nodes']} nodes, {summary['edges']} edges, "
          f"{summary['communities']} communities")

    print("\n✓ Restore complete. The site should now serve identical results "
          "to its pre-outage state (minus on-demand LLM explanations, which "
          "regrow as users click 'Generate explanation').")


if __name__ == '__main__':
    main()
