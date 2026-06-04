"""
In-process knowledge graph: Louvain communities + PageRank computed via
NetworkX over the similarity_cache rows in Postgres.

Replaces the previous Neo4j-Aura path. The algorithm is the same one already
used as the Aura-Free fallback (see git history of storage/neo4j/store.py:
_run_community_detection_networkx), so existing node colours and node sizes
are preserved bit-identically for any given input.

Edges below config.MIN_SCORE never feed the graph view, matching the old
behaviour where Neo4j only carried edges with final_score ≥ MIN_SCORE.
"""
from __future__ import annotations

import config
from storage.postgres import store as pg_store


def recompute_graph_stats() -> dict:
    """
    Read similarity_cache + courses from Postgres, rebuild the weighted graph,
    run Louvain + PageRank, write (community, pagerank) back to courses.

    Returns a small summary {'nodes': n, 'edges': m, 'communities': k}.
    """
    import networkx as nx
    from networkx.algorithms import community as nx_comm

    node_ids = [c[0] for c in pg_store.get_all_courses()]
    if not node_ids:
        return {"nodes": 0, "edges": 0, "communities": 0}

    raw_edges = pg_store.get_all_similarity_edges()
    edges = [
        (a, b, max(s, 0.01))
        for a, b, s in raw_edges
        if s >= config.MIN_SCORE
    ]

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for a, b, weight in edges:
        if G.has_edge(a, b):
            G[a][b]['weight'] = max(G[a][b]['weight'], weight)
        else:
            G.add_edge(a, b, weight=weight)

    if edges:
        communities = nx_comm.louvain_communities(G, weight='weight', seed=42)
        pr = nx.pagerank(G, weight='weight')
    else:
        communities = [{n} for n in node_ids]
        pr = {n: 1.0 / len(node_ids) for n in node_ids}

    # Deterministic community IDs: order by the smallest member id (string sort)
    # so the same input always produces the same colour assignment.
    comm_map: dict[str, int] = {}
    for cid, members in enumerate(sorted(communities, key=lambda s: min(s))):
        for node in members:
            comm_map[node] = cid

    pg_store.upsert_course_graph_stats([
        {
            'id':        nid,
            'community': int(comm_map.get(nid, 0)),
            'pagerank':  float(pr.get(nid, 0.0)),
        }
        for nid in node_ids
    ])

    return {
        "nodes":       len(node_ids),
        "edges":       G.number_of_edges(),
        "communities": len(communities),
    }


def shortest_path(from_id: str, to_id: str) -> list[dict]:
    """
    Find the shortest conceptual path between two courses. Each hop is a
    similarity edge above MIN_SCORE; the edge_score reported is the original
    final_score (not the BFS weight).

    Returns [] if no path exists. Matches the previous Neo4j payload shape.
    """
    import networkx as nx

    nodes = pg_store.get_graph_nodes()
    node_by_id = {n['id']: n for n in nodes}
    if from_id not in node_by_id or to_id not in node_by_id:
        return []

    raw_edges = pg_store.get_all_similarity_edges()
    edges = [(a, b, s) for a, b, s in raw_edges if s >= config.MIN_SCORE]

    G = nx.Graph()
    G.add_nodes_from(node_by_id.keys())
    for a, b, s in edges:
        G.add_edge(a, b, score=s)

    try:
        path_ids = nx.shortest_path(G, from_id, to_id)
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []

    scores = [None]
    for i in range(1, len(path_ids)):
        scores.append(G[path_ids[i - 1]][path_ids[i]]['score'])

    return [
        {
            'course_id':  pid,
            'name':       node_by_id[pid]['name'],
            'edge_score': scores[i],
        }
        for i, pid in enumerate(path_ids)
    ]


def get_communities() -> dict[int, list[str]]:
    """Return {community_id: [course_id, ...]} for courses with a community."""
    nodes = pg_store.get_graph_nodes()
    out: dict[int, list[str]] = {}
    for n in nodes:
        if n['community'] is None:
            continue
        out.setdefault(int(n['community']), []).append(n['id'])
    return dict(sorted(out.items()))


def get_neighbors(course_id: str, top: int = 10) -> list[dict]:
    """
    Top-N neighbours from similarity_cache (matches the old /graph/neighbors shape).

    Identical payload to the Neo4j version: course_id, name, score, lex_score,
    sem_score, driving_terms.
    """
    name_by_id = {c[0]: c[1] for c in pg_store.get_all_courses()}
    neighbors = pg_store.get_neighbors(course_id, top=top, sort='hybrid')
    return [
        {
            'course_id':     n['course_id'],
            'name':          name_by_id.get(n['course_id'], n['course_id']),
            'score':         n['final_score'],
            'lex_score':     n['lex_score'],
            'sem_score':     n['sem_score'],
            'driving_terms': n['driving_terms'],
        }
        for n in neighbors
    ]
