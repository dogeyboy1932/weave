#!/usr/bin/env bash
# In-place deploy of the CURRENT (Phase 12) architecture on an Oracle Cloud
# Always-Free ARM VM — or any Ubuntu host.
#
# The code is expected to ALREADY be on the VM (rsync'd up from your laptop),
# so this script does NOT clone from GitHub — the repo is private. It just
# provisions Docker + firewall + swap, then builds and runs the single
# container (Dockerfile.api) against your external Supabase (DATABASE_URL).
#
# Typical use — from your laptop:
#   rsync -az --delete \
#     --exclude '.git' --exclude '.archive' --exclude '__pycache__' \
#     --exclude '.venv' --exclude 'node_modules' --exclude '.model_cache' \
#     "/path/to/weave/"  ubuntu@ORACLE_IP:~/weave/
#   ssh ubuntu@ORACLE_IP 'cd ~/weave && bash deploy/oracle/setup.sh'
#
# DATABASE_URL is read from the rsync'd .env. If you excluded .env, pass it:
#   ssh ubuntu@ORACLE_IP 'cd ~/weave && DATABASE_URL=... bash deploy/oracle/setup.sh'
#
# Re-running is safe/idempotent: rebuilds and restarts.
set -euo pipefail

PORT="${PORT:-8080}"
IMAGE="weave-api"
CONTAINER="weave"

# Repo root = two levels up from this script (deploy/oracle/setup.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DIR"

SUDO=""; [[ "${EUID:-$(id -u)}" -ne 0 ]] && SUDO="sudo"
log() { printf '\n==> %s\n' "$*"; }

if [[ ! -f Dockerfile.api ]]; then
  echo "ERROR: Dockerfile.api not found in $DIR — did the rsync land here?" >&2
  exit 1
fi

# ── 1. Base packages ────────────────────────────────────────────────────────
log "Installing base packages (curl, ca-certificates)…"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq curl ca-certificates

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
DOCKER="docker"; docker info >/dev/null 2>&1 || DOCKER="$SUDO docker"

# ── 4. Linux firewall: Oracle Ubuntu images REJECT everything but 22 ─────────
if command -v iptables >/dev/null 2>&1; then
  log "Opening port $PORT in host iptables (Oracle blocks it by default)…"
  if ! $SUDO iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
    $SUDO iptables -I INPUT -p tcp --dport "$PORT" -j ACCEPT
  fi
  $SUDO apt-get install -y -qq netfilter-persistent iptables-persistent 2>/dev/null || true
  command -v netfilter-persistent >/dev/null 2>&1 && $SUDO netfilter-persistent save || true
fi

# ── 5. .env (DATABASE_URL required; GEMINI_API_KEY optional) ────────────────
if [[ ! -f .env ]]; then
  if [[ -z "${DATABASE_URL:-}" ]]; then
    echo "ERROR: no .env in $DIR and DATABASE_URL not set." >&2
    echo "       Either rsync your .env up, or re-run with DATABASE_URL=..." >&2
    exit 1
  fi
  log "Writing .env from environment…"
  {
    echo "DATABASE_URL=${DATABASE_URL}"
    echo "POSTGRES_SSLMODE=require"
    [[ -n "${GEMINI_API_KEY:-}" ]] && echo "GEMINI_API_KEY=${GEMINI_API_KEY}"
  } > .env
fi

# ── 6. Build (ARM-safe torch) & run ─────────────────────────────────────────
log "Building image (first build downloads torch + SciNCL — 20–40 min on ARM)…"
$DOCKER build -f Dockerfile.api -t "$IMAGE" .

log "Restarting container…"
$DOCKER rm -f "$CONTAINER" 2>/dev/null || true
$DOCKER run -d --name "$CONTAINER" --restart always \
  --env-file .env -p "${PORT}:8080" "$IMAGE"

# ── 7. Wait for health ──────────────────────────────────────────────────────
log "Waiting for /health…"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    IP=$(curl -fsS ifconfig.me 2>/dev/null || echo YOUR_ORACLE_IP)
    log "✓ Live at: http://${IP}:${PORT}"
    exit 0
  fi
  sleep 5
done

echo "Health check didn't pass yet — check logs: $DOCKER logs -f $CONTAINER" >&2
exit 1
