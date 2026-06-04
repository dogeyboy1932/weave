"""
Knowledge graph endpoints.

Backed by Postgres (similarity_cache + courses) and NetworkX in-process.
Previously these endpoints read from Neo4j Aura; the rewrite preserves the
same response shapes so ui/index.html does not need to change.
"""
from fastapi import APIRouter, HTTPException, Query

from scoring import graph_compute
from storage.postgres import store as pg_store

router = APIRouter()


@router.get("/path")
def shortest_path(
    from_course: str = Query(..., alias="from", description="Start course ID"),
    to_course:   str = Query(..., alias="to",   description="End course ID"),
):
    """
    Shortest conceptual path between two courses. Each hop is a similarity
    edge above MIN_SCORE; edge_score is the original final_score.
    """
    path = graph_compute.shortest_path(from_course, to_course)
    if not path:
        raise HTTPException(
            404,
            detail=f"No path found between '{from_course}' and '{to_course}'",
        )
    return {"from": from_course, "to": to_course, "path": path}


@router.get("/communities")
def get_communities():
    """
    Louvain community assignments. Each key is a community ID;
    value is the list of course IDs in that community.
    """
    communities = graph_compute.get_communities()
    if not communities:
        raise HTTPException(
            404,
            detail="No communities found. Run scripts/build_graph.py first.",
        )
    return communities


@router.get("/all")
def full_graph(min_score: float = Query(0.4, ge=0.0, le=1.0)):
    """All nodes + edges for graph visualisation (edges filtered by min_score)."""
    nodes = pg_store.get_graph_nodes()
    edges = pg_store.get_graph_edges(min_score)
    return {"nodes": nodes, "edges": edges}


@router.get("/neighbors")
def graph_neighbors(
    course: str = Query(..., description="Course ID"),
    top:    int  = Query(10, ge=1, le=50),
):
    """Top-N graph neighbours for a course (same payload as the old Neo4j route)."""
    neighbors = graph_compute.get_neighbors(course, top)
    if not neighbors:
        raise HTTPException(404, detail=f"No graph neighbours for '{course}'")
    return {"course": course, "neighbors": neighbors}
