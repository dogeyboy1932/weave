from contextlib import asynccontextmanager
from hashlib import sha256
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import storage.postgres.store as pg_store
from storage.vectors import store as vs
from api.routes import courses, similarity, graph, ingest, topics

UI_DIR = Path(__file__).parent.parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pg_store.init_schema()
    vs.get_or_create_collection()
    yield


app = FastAPI(
    title="Weave Semantic Similarity API",
    description=(
        "Hybrid IR + vector course-similarity engine. "
        "Uses Dirichlet-smoothed JSD (lexical) + SciNCL cosine similarity (semantic)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Query logging middleware ─────────────────────────────────────────────────
# Records every request to query_log: endpoint, params, status, latency. Never
# blocks the response — a failed log write is swallowed. Skips /health, /static,
# and the UI HTML routes to keep the table focused on real API calls.
_LOG_SKIP_PREFIXES = ("/health", "/static", "/docs", "/openapi.json", "/favicon")


@app.middleware("http")
async def query_log_middleware(request: Request, call_next):
    path = request.url.path
    if path in ("/", "/upload") or path.startswith(_LOG_SKIP_PREFIXES):
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        ua = request.headers.get("user-agent", "")
        ua_hash = sha256(ua.encode()).hexdigest()[:16] if ua else None
        pg_store.insert_query_log(
            endpoint=path,
            method=request.method,
            query_params=dict(request.query_params),
            status_code=response.status_code,
            result_count=None,  # set by handlers if interesting; null otherwise
            latency_ms=latency_ms,
            user_agent_hash=ua_hash,
        )
    except Exception:
        pass

    return response


app.include_router(courses.router,    prefix="/courses",    tags=["courses"])
app.include_router(similarity.router, prefix="/similarity", tags=["similarity"])
app.include_router(topics.router,     prefix="/topics",     tags=["topics"])
app.include_router(graph.router,      prefix="/graph",      tags=["graph"])
app.include_router(ingest.router,     prefix="/ingest",     tags=["ingest"])


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/upload", include_in_schema=False)
def upload_page():
    return FileResponse(UI_DIR / "upload.html", headers={"Cache-Control": "no-store"})
