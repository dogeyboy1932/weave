# Hugging Face Spaces image (Docker SDK).  $0 host — free tier is 2 vCPU /
# 16 GB RAM, so SciNCL's ~1.5 GB resident footprint is a non-issue.
#
# This is the sibling of Dockerfile.api (Fly.io).  Both build the same app;
# they differ only in host conventions:
#   - HF runs the container as a NON-root user (UID 1000) with $HOME=/home/user.
#   - HF routes traffic to the port declared by `app_port` in README.md
#     (set to 7860 there, matching PORT below).
#   - Secrets (DATABASE_URL) are injected as env vars at runtime via the
#     Space's Settings → Secrets, not baked here.
#
# Database stays on Supabase (Postgres + pgvector) — nothing stateful lives
# in this container, so HF's ephemeral filesystem is fine.
#
# See deploy/HUGGINGFACE.md for the one-time setup + push flow.

FROM python:3.12-slim

# System deps: libpq for psycopg2, libgomp for torch CPU's OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as UID 1000. Create that user up front so the model cache and
# app code are owned by (and writable to) the runtime user.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app
RUN chown -R user:user /app

USER user

# CPU-only torch first so sentence-transformers doesn't pull the 2 GB CUDA build.
RUN pip install --no-cache-dir --user --upgrade pip \
 && pip install --no-cache-dir --user --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3.0"

COPY --chown=user:user requirements-api.txt ./
RUN pip install --no-cache-dir --user -r requirements-api.txt

# Bake the SciNCL weights (~420 MB) into the image so cold starts don't
# re-download them. Cache lives under $HOME so it's writable at runtime too.
ENV SCINCL_CACHE_DIR=/home/user/.model_cache
ENV HF_HOME=/home/user/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('malteos/scincl', cache_folder='/home/user/.model_cache')"

ENV NLTK_DATA=/home/user/nltk_data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/home/user/nltk_data', quiet=True)"

# App code.
COPY --chown=user:user api/        ./api/
COPY --chown=user:user scoring/    ./scoring/
COPY --chown=user:user storage/    ./storage/
COPY --chown=user:user llm/        ./llm/
COPY --chown=user:user scripts/    ./scripts/
COPY --chown=user:user config.py   ./
COPY --chown=user:user config.yml  ./
COPY --chown=user:user data/       ./data/
COPY --chown=user:user ui/         ./ui/

ENV POSTGRES_SSLMODE=require
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
