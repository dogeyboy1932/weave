"""
Vector store. Pgvector (in Supabase Postgres) is the only backend.

This module used to dispatch between pgvector and a local-Milvus backend; the
Milvus path moved to .archive/legacy-local-docker/ when the project consolidated
on a single cloud-backed pipeline (see archive/README.md and docs/HISTORY.md
Phase 12). Imports across the codebase still do:

    from storage.vectors import store as vs

— so we keep this thin re-export to avoid a noisy rename across every call site.
"""
from storage.vectors.pgvector_store import (  # noqa: F401
    DIM,
    get_or_create_collection,
    insert_chunks,
    get_embeddings_for_course,
    get_chunk_embedding,
    get_chunks_with_embeddings,
    search_in_course,
    search_global_excluding,
    search_topics_semantic,
    search_neighbors_batch,
    delete_course,
    drop_collection,
)

BACKEND_NAME = "pgvector"
