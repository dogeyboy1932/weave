"""
Compute all pairwise similarity scores and refresh the knowledge graph.

Pure-Postgres path — no Neo4j. After scoring all pairs, Louvain + PageRank
run in-process via scoring/graph_compute.py and write community/pagerank
back to the courses table.

Run from the akhil_app/ directory:
    python scripts/build_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.language_model import build_all_lms
from scoring.hybrid_scorer  import score_all_pairs
from scoring                import graph_compute
from storage.postgres       import store as pg_store


def main():
    print("Loading term counts from PostgreSQL …")
    all_counts = pg_store.get_all_term_counts()
    if not all_counts:
        print("ERROR: No term counts found. Run scripts/restore_from_dump.py first.")
        sys.exit(1)
    print(f"  {len(all_counts)} courses found.")

    print("Building Dirichlet-smoothed language models …")
    lms = build_all_lms(all_counts)

    edge_count = score_all_pairs(lms, all_counts)

    print("\nRecomputing community clusters + PageRank …")
    summary = graph_compute.recompute_graph_stats()
    print(f"  {summary['nodes']} nodes, {summary['edges']} edges, "
          f"{summary['communities']} communities")

    print(f"\nGraph build complete — {edge_count} edges above MIN_SCORE.")
    print("Start the API: uvicorn api.main:app --port 8080 --reload")


if __name__ == '__main__':
    main()
