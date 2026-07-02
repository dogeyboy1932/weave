#!/usr/bin/env bash
# One-command deploy of the CURRENT (Phase 12) architecture on an Oracle Cloud
# Always-Free ARM VM — or any Ubuntu host.
#
# Current architecture is a SINGLE container (Dockerfile.api) that talks to an
# external Supabase Postgres via DATABASE_URL. There are NO local databases
# (the old Neo4j/Milvus/etcd/minio stack is gone — see .archive/).
#
# Usage on a fresh Ubuntu 22.04 ARM VM (as the 'ubuntu' user):
#
#   DATABASE_URL='postgresql://...supabase...?sslmode=require' \
#     bash <(curl -fsSL https://raw.githubusercontent.com/dogeyboy1932/armchair/main/deploy/oracle/setup.sh)
#
# Optional: GEMINI_API_KEY=... to pre-bake explanations (users can also paste
# their own key in the UI).
#
# Re-running is safe/idempotent: it pulls latest, rebuilds, and restarts.
set -euo pipefail

REPO="https://github.com/dogeyboy1932/armchair.git"
DIR="${SIIP_DIR:-$HOME/armchair}"
BRANCH="${SIIP_BRANCH:-main}"
PORT="${PORT:-8080}"
IMAGE="armchair-api"
CONTAINER="armchair"

SUDO=""; [[ "${EUID:-$(id -u)}" -ne 0 ]] && SUDO="sudo"

log() { printf '\n==> %s\n' "$*"; }

# ── 1. Base packages ────────────────────────────────────────────────────────
log "Installing base packages (git, curl)…"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl ca-certificates

# ── 2. Swap (torch + SciNCL build is memory-hungry; add 4G if RAM < 8G) ─────
mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
if [[ "$mem_kb" -lt 8000000 && ! -f /swapfile ]]; then
  log "RAM < 8 GB — adding a 4 GB swapfile so the image build doesn't OOM…"
  $SUDO fallocate -l 4G /swapfile
  $SUDO chmod 600 /swapfile
  $SUDO mkswap /swapfile
  $SUDO swapon /swapfile
  echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null
fi

# ── 3. Docker ───────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker…"
  curl -fsSL https://get.docker.com | $SUDO sh
  $SUDO usermod -aG docker "$USER" || true
fi

# ── 4. Linux firewall: Oracle Ubuntu images REJECT everything but 22 ─────────
#     Insert an ACCEPT for our port ahead of the default REJECT rule.
if command -v iptables >/dev/null 2>&1; then
  log "Opening port $PORT in the host iptables (Oracle blocks it by default)…"
  if ! $SUDO iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
    $SUDO iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT
  fi
  # Persist across reboots if the tooling is present.
  $SUDO apt-get install -y -qq netfilter-persistent iptables-persistent 2>/dev/null || true
  command -v netfilter-persistent >/dev/null 2>&1 && $SUDO netfilter-persistent save || true
fi

# ── 5. Clone / update the repo ──────────────────────────────────────────────
if [[ -d "$DIR/.git" ]]; then
  log "Repo exists — pulling latest…"
  git -C "$DIR" pull origin "$BRANCH" || true
else
  log "Cloning $REPO → $DIR…"
  git clone --branch "$BRANCH" --depth 1 "$REPO" "$DIR"
fi
cd "$DIR"

# ── 6. .env (DATABASE_URL is required; GEMINI_API_KEY optional) ─────────────
if [[ ! -f .env ]]; then
  if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "ERROR: no .env found and DATABASE_URL not provided." >&2
    echo "       Re-run with DATABASE_URL='postgresql://...supabase...?sslmode=require'" >&2
    echo "       (or scp your local .env to $DIR/.env first)." >&2
    exit 1
  fi
  log "Writing .env from environment…"
  {
    echo "DATABASE_URL=${DATABASE_URL}"
    echo "POSTGRES_SSLMODE=require"
    [[ -n "${GEMINI_API_KEY:-}" ]] && echo "GEMINI_API_KEY=${GEMINI_API_KEY}"
  } > .env
fi

# ── 7. Build (ARM-safe: Dockerfile.api picks the arm torch wheel) & run ──────
DOCKER="docker"; command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || DOCKER="$SUDO docker"

log "Building image (first build downloads torch + SciNCL — 20–40 min on ARM)…"
$DOCKER build -f Dockerfile.api -t "$IMAGE" .

log "Restarting container…"
$DOCKER rm -f "$CONTAINER" 2>/dev/null || true
$DOCKER run -d --name "$CONTAINER" --restart always \
  --env-file .env -p "${PORT}:8080" "$IMAGE"

# ── 8. Wait for health ──────────────────────────────────────────────────────
log "Waiting for /health…"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    IP=$(curl -fsS ifconfig.me 2>/dev/null || echo YOUR_ORACLE_IP)
    log "✓ Live at: http://${IP}:${PORT}"
    exit 0
  fi
  sleep 5
done

echo "Health check didn't pass yet — check logs with: $DOCKER logs -f $CONTAINER" >&2
exit 1
