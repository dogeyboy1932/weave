#!/usr/bin/env bash
# One-shot production deploy.
#
#   1. Verify .env (at the repo root) exists and DATABASE_URL is set.
#   2. Verify Supabase is reachable + ensure pgvector extension is installed.
#   3. Make sure flyctl is installed and you're authenticated.
#   4. Ensure the Fly app exists (create if not).
#   5. Push secrets (DATABASE_URL, optional GEMINI_API_KEY) and clear stale
#      Neo4j secrets from earlier deploys.
#   6. flyctl deploy (remote builder).
#   7. Poll /health until 200 (or fail with logs).
#
# Day-to-day deploys: just `git push origin main` — GitHub Actions runs this
# same path automatically. Use this script when you want to watch the deploy
# from your laptop or you don't have CI configured.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── 1. Load credentials ──────────────────────────────────────────────────────
# .env at the repo root is the only source of truth for secrets — both for
# local uvicorn and for this deploy. See .env contents for the expected keys.
if [[ ! -f .env ]]; then
  cat >&2 <<'EOF'
Missing .env at the repo root. Create it with:

  DATABASE_URL=postgresql://postgres:<PASSWORD>@db.<REF>.supabase.co:5432/postgres
  POSTGRES_SSLMODE=require
  GEMINI_API_KEY=                # optional
  FLY_APP_NAME=siip-armchair-akhil
  FLY_API_TOKEN=                 # optional, for non-interactive CI
EOF
  exit 1
fi
set -a; source .env; set +a
: "${DATABASE_URL:?DATABASE_URL required in .env}"

# ── 2. Verify Supabase + pgvector ────────────────────────────────────────────
echo "==> Verifying Supabase + installing pgvector extension"
python3 - <<PY
import os, psycopg2
conn = psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")
cur  = conn.cursor()
cur.execute("SELECT version()")
print("  Postgres :", cur.fetchone()[0].split(",")[0])
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()
cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
print("  pgvector :", cur.fetchone()[0])
cur.close(); conn.close()
PY

# ── 3. Ensure flyctl + auth ──────────────────────────────────────────────────
export PATH="$HOME/.fly/bin:$PATH"
if ! command -v flyctl >/dev/null 2>&1; then
  echo "flyctl not found. Install it once:" >&2
  echo "  curl -L https://fly.io/install.sh | sh" >&2
  echo "  export PATH=\"\$HOME/.fly/bin:\$PATH\"" >&2
  exit 1
fi
if [[ -n "${FLY_API_TOKEN:-}" ]]; then
  export FLY_ACCESS_TOKEN="$FLY_API_TOKEN"
fi
if ! flyctl auth whoami >/dev/null 2>&1; then
  echo "Not authenticated to Fly.io. Run: flyctl auth login" >&2
  exit 1
fi
echo "==> Authenticated as: $(flyctl auth whoami 2>&1)"

# ── 4. App exists? ───────────────────────────────────────────────────────────
APP="${FLY_APP_NAME:-siip-armchair-akhil}"
if flyctl status --app "$APP" >/dev/null 2>&1; then
  echo "==> Reusing existing Fly app: $APP"
else
  echo "==> Creating Fly app: $APP"
  flyctl apps create "$APP" --org personal
fi

# Sync fly.toml's app name to the chosen one (no-op if already matching).
if ! grep -q "^app = \"$APP\"" fly.toml; then
  echo "==> Updating fly.toml app name to: $APP"
  sed -i.bak "s/^app = .*/app = \"$APP\"/" fly.toml
  rm -f fly.toml.bak
fi

# ── 5. Push secrets ──────────────────────────────────────────────────────────
echo "==> Setting Fly secrets…"
SECRETS=( "DATABASE_URL=$DATABASE_URL" )
# Optional Gemini key — enables LLM-dependent endpoints (PDF ingest topic
# extraction, on-demand explanations). UI users can also bring their own.
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  SECRETS+=( "GEMINI_API_KEY=$GEMINI_API_KEY" )
fi
# Clear stale Neo4j secrets from pre-Phase-12 deploys (no-op if already absent).
flyctl secrets unset --app "$APP" NEO4J_URI NEO4J_USER NEO4J_PASSWORD 2>/dev/null || true
flyctl secrets set --app "$APP" --stage "${SECRETS[@]}" >/dev/null

# ── 6. Deploy ────────────────────────────────────────────────────────────────
echo "==> Building & deploying (5-10 min on first deploy)…"
flyctl deploy --app "$APP" --remote-only --wait-timeout 900

# ── 7. Smoke test ────────────────────────────────────────────────────────────
URL="https://${APP}.fly.dev"
echo ""
echo "==> Waiting for $URL/health to return 200…"
for i in $(seq 1 30); do
  if curl -sf "${URL}/health" >/dev/null; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✓ SIIP is live at: $URL"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "  Graph view:  $URL"
    echo "  API docs:    $URL/docs"
    echo "  Logs:        flyctl logs --app $APP"
    echo "  Status:      flyctl status --app $APP"
    exit 0
  fi
  printf '.'
  sleep 5
done

echo ""
echo "/health did not pass in 150 s. Investigate:"
echo "  flyctl logs --app $APP"
exit 1
