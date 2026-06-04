"""
Process-wide config. Single source of truth is config.yml; environment
variables override individual keys (so Fly secrets and .env still work).

Precedence: env var  >  config.yml  >  built-in default.

This module preserves the names used elsewhere in the codebase
(POSTGRES_HOST, ALPHA_LEX, SCINCL_MODEL, …) so call sites do not change.
"""
import os
from pathlib import Path
from urllib.parse import urlparse, unquote

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent
load_dotenv(_ROOT / '.env')


# ── Load YAML ──────────────────────────────────────────────────────────────
def _load_yaml() -> dict:
    path = _ROOT / 'config.yml'
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


_CFG = _load_yaml()


def _get(env_key: str, *yaml_path: str, default=None, cast=None):
    """Env var beats YAML beats default. Optional type cast at the end."""
    raw = os.environ.get(env_key)
    if raw is None and yaml_path:
        node = _CFG
        found = True
        for k in yaml_path:
            if not isinstance(node, dict) or k not in node:
                found = False
                break
            node = node[k]
        raw = node if found else default
    elif raw is None:
        raw = default
    if cast is not None and raw is not None:
        return cast(raw)
    return raw


# ── Database URL parsing ───────────────────────────────────────────────────
def _parse_database_url(url: str) -> dict | None:
    """Parse a postgres:// connection URL into individual fields."""
    if not url:
        return None
    p = urlparse(url)
    if p.scheme not in ('postgres', 'postgresql'):
        return None
    return {
        'host':     p.hostname or 'localhost',
        'port':     p.port or 5432,
        'db':       (p.path or '/').lstrip('/') or 'postgres',
        'user':     unquote(p.username) if p.username else '',
        'password': unquote(p.password) if p.password else '',
        'sslmode':  'require' if 'sslmode=require' in (p.query or '') else None,
    }


_pg_url = _parse_database_url(os.environ.get('DATABASE_URL', ''))


# ── PostgreSQL (DATABASE_URL takes priority over individual env vars) ──────
POSTGRES_HOST     = (_pg_url or {}).get('host')     or os.environ.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT     = (_pg_url or {}).get('port')     or int(os.environ.get('POSTGRES_PORT', 5432))
POSTGRES_DB       = (_pg_url or {}).get('db')       or os.environ.get('POSTGRES_DB', 'siip')
POSTGRES_USER     = (_pg_url or {}).get('user')     or os.environ.get('POSTGRES_USER', 'siip')
POSTGRES_PASSWORD = (_pg_url or {}).get('password') or os.environ.get('POSTGRES_PASSWORD', '')
POSTGRES_SSLMODE  = (
    (_pg_url or {}).get('sslmode')
    or _get('POSTGRES_SSLMODE', 'storage', 'postgres_sslmode', default=None)
)


# ── Scoring ────────────────────────────────────────────────────────────────
ALPHA_LEX           = float(_get('ALPHA_LEX', 'scoring', 'alpha_lex', default=0.3))
ALPHA_SEM           = float(_get('ALPHA_SEM', 'scoring', 'alpha_sem', default=0.5))
ALPHA_CAT           = float(_get('ALPHA_CAT', 'scoring', 'alpha_cat', default=0.2))
ALPHA               = float(_get('ALPHA',     'scoring', 'alpha',     default=0.4))
DIRICHLET_MU        = float(_get('DIRICHLET_MU', 'scoring', 'dirichlet_mu', default=2000.0))
SEM_FLOOR           = float(_get('SEM_FLOOR',    'scoring', 'sem_floor',    default=0.35))
MIN_SCORE           = float(_get('MIN_SCORE',    'scoring', 'min_score',    default=0.55))


# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_ANN           = int(_get('TOP_K_ANN',           'retrieval', 'top_k_ann',           default=5))
TOP_K_DRIVING_TERMS = int(_get('TOP_K_DRIVING_TERMS', 'retrieval', 'top_k_driving_terms', default=8))
NON_OBVIOUS_TOP_K   = int(_get('NON_OBVIOUS_TOP_K',   'retrieval', 'non_obvious_top_k',   default=50))

ANN_INDEX_TYPE     = str(_get('ANN_INDEX_TYPE',     'retrieval', 'ann', 'index_type',     default='hnsw')).lower()
ANN_DISTANCE       = str(_get('ANN_DISTANCE',       'retrieval', 'ann', 'distance',       default='cosine')).lower()
ANN_HNSW_M         = int(_get('ANN_HNSW_M',         'retrieval', 'ann', 'hnsw_m',         default=16))
ANN_HNSW_EF_SEARCH = int(_get('ANN_HNSW_EF_SEARCH', 'retrieval', 'ann', 'hnsw_ef_search', default=64))


# ── Model ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_ID      = _get('EMBEDDING_MODEL_ID',      'model', 'embedding', 'id',        default='scincl-v1-fp32')
EMBEDDING_MODEL_RUNTIME = _get('EMBEDDING_MODEL_RUNTIME', 'model', 'embedding', 'runtime',   default='pytorch')
EMBEDDING_DIM           = int(_get('EMBEDDING_DIM',       'model', 'embedding', 'dim',       default=768))
SCINCL_MODEL            = _get('SCINCL_MODEL',            'model', 'embedding', 'hf_name',   default='malteos/scincl')
SCINCL_CACHE_DIR        = _get('SCINCL_CACHE_DIR',        'model', 'embedding', 'cache_dir', default=str(_ROOT / '.model_cache'))
EMBEDDING_ONNX_PATH     = _get('EMBEDDING_ONNX_PATH',     'model', 'embedding', 'onnx_path', default=None)


# ── Data paths ─────────────────────────────────────────────────────────────
DATA_DIR          = _ROOT / 'data'
DEFINITIONS_PATH  = DATA_DIR / 'topic_definitions.json'   # read by chunker / ingest
COURSES_DUMP_PATH = DATA_DIR / 'courses_dump.json'        # canonical restore source


# ── LLM (Gemini) ───────────────────────────────────────────────────────────
GEMINI_API_KEY       = os.environ.get('GEMINI_API_KEY', '')   # always env-only (secret)
CATEGORY_LABEL_MODEL = _get('CATEGORY_LABEL_MODEL', 'model', 'llm', 'category_label_model', default='gemini-2.5-flash-lite')
LLM_EXPLAIN_MODEL    = _get('LLM_EXPLAIN_MODEL',    'model', 'llm', 'explain_model',         default='gemini-2.5-flash')
