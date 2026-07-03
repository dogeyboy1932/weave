# Deploy Weave on Oracle Cloud (Always Free, $0/mo)

Current architecture (Phase 12): **one container** (`Dockerfile.api`) that connects
to your existing **Supabase Postgres** via `DATABASE_URL`. No databases run on the
VM. This makes the Oracle deploy a single build + run — far simpler than the
legacy multi-service stack in `.archive/`.

- **Cost:** $0/mo on Oracle's Always-Free ARM tier
- **Hands-on time:** ~15 min console + one paste; ~20–40 min automated first build
- **You still need:** your Supabase `DATABASE_URL` (same one Fly uses — it's in your local `.env`)

---

## Part 1 — Create the VM (Oracle web console)

1. Sign up / sign in at https://cloud.oracle.com → **Start for free**.
2. **☰ menu → Compute → Instances → Create instance**:

   | Setting | Value |
   |---|---|
   | Name | `weave-prod` |
   | Image | **Ubuntu 22.04** (Canonical) |
   | Shape | **Ampere → VM.Standard.A1.Flex** |
   | OCPUs / Memory | **2 OCPU / 12 GB** (min **1 / 6**) |
   | Boot volume | 50 GB (default) |

3. **Networking → Edit → Assign public IPv4 address: Yes** (required to SSH / share a link).
4. **SSH keys → Generate a key pair for me** → download the private key. **Create.**
5. Wait for **State = Running**, then copy the **Public IP**.

### Open port 8080 in Oracle's firewall (Security List)

Oracle blocks everything but port 22 by default:

1. Instance page → **Primary VNIC → Subnet** → its **Security List**.
2. **Add ingress rules**: Source `0.0.0.0/0`, IP Protocol **TCP**, Destination port **8080**.

*(The setup script opens the Linux-level firewall automatically — both layers are required.)*

---

## Part 2 — Deploy (rsync — the repo is private, so no GitHub clone on the VM)

The repo is private, so the VM can't pull it. Instead, copy the code
**up from your laptop** with rsync — no GitHub credentials ever touch the cloud VM.

**From your laptop** (run in the repo root, `weave/`). Your `.env` (with
`DATABASE_URL`) is copied up too so the container can reach Supabase:

```bash
rsync -az --delete \
  --exclude '.git' --exclude '.archive' --exclude '__pycache__' \
  --exclude '.venv' --exclude 'node_modules' --exclude '.model_cache' \
  -e "ssh -i ~/Downloads/ssh-key-*.key" \
  ./  ubuntu@YOUR_ORACLE_IP:~/weave/
```

Then run the installer on the VM:

```bash
ssh -i ~/Downloads/ssh-key-*.key ubuntu@YOUR_ORACLE_IP \
  'cd ~/weave && bash deploy/oracle/setup.sh'
```

> If you'd rather NOT copy `.env`, add `--exclude '.env'` to the rsync and pass the
> URL to the installer instead: `... 'cd ~/weave && DATABASE_URL="..." bash deploy/oracle/setup.sh'`

The script installs Docker, adds swap if needed, opens the host firewall, builds the
ARM image (downloads torch + SciNCL — 20–40 min first time), starts the container
with `--restart always`, and prints:

```
✓ Live at: http://YOUR_ORACLE_IP:8080
```

**Re-deploys:** re-run the same two commands (rsync, then the ssh install). rsync only
sends changed files; the Docker layer cache makes rebuilds fast unless deps changed.

---

## Part 3 — Verify

```bash
curl http://127.0.0.1:8080/health          # {"status":"ok"} on the server
```

- App: `http://YOUR_ORACLE_IP:8080`
- API docs: `http://YOUR_ORACLE_IP:8080/docs`

---

## Day-to-day

```bash
# On the VM:
docker logs -f weave                     # follow logs
docker restart weave                     # bounce
docker rm -f weave                        # stop (image stays)

# To update: re-run the rsync + install from your laptop (see Part 2).
```

---

## Troubleshooting

**"Out of host capacity" on VM.Standard.A1.Flex** — the most common issue; free ARM
capacity is oversubscribed. In order:
1. On Create instance → **Placement**, switch **Availability Domain** to AD-2, then AD-3.
2. Drop the shape to **1 OCPU / 6 GB** (the script adds 4 GB swap to compensate).
3. Retry off-peak (early morning US time).
4. Last resort: a new account with a different home region (`uk-london-1`,
   `eu-frankfurt-1`, `ap-osaka-1`) — Ashburn/Phoenix are worst.

**Page won't load but `/health` works on the server** — the Oracle **Security List**
ingress rule for 8080 (Part 1) is missing. The host firewall is handled by the script.

**Build OOMs** — confirm swap is active (`free -h`); rerun the script.

**Optional HTTPS** — add ingress rules for 80/443, point a domain's A record at the
IP, and put Caddy in front of `:8080`. (Ask and I'll add a Caddy compose.)

---

## Cost & limits

Always-Free ARM: up to **4 OCPU + 24 GB RAM total** per tenancy, 200 GB storage,
10 TB egress/mo — all **$0**. Stopping the VM stays free (storage only).
