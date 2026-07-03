# Deploying Weave to Hugging Face Spaces (free, $0/mo)

This is the **free** replacement for the Fly.io host. Fly's old $5/mo hobby
credit is gone, so an always-on 2 GB machine there now bills ~$4–6/mo. HF
Spaces' free tier gives **2 vCPU / 16 GB RAM** — SciNCL's ~1.5 GB footprint is
trivial — and it builds straight from the `Dockerfile` at the repo root.

**What stays the same:** the database. Postgres + pgvector still live on
Supabase; the Space is stateless and just connects to it via `DATABASE_URL`.

**The one tradeoff:** free Spaces **sleep after ~48 h of inactivity** and
cold-start in ~1–2 min (the SciNCL weights are baked into the image, so it's a
reload, not a re-download). Background ingest only runs while the Space is
awake. For burst-usage evaluation by the team, that's fine.

---

## One-time setup

### 1. Create the Space
- Go to https://huggingface.co/new-space
- **Owner / name:** e.g. `your-username/weave`
- **SDK:** Docker → **Blank** template
- **Hardware:** CPU basic (free)
- Create it. You'll get a git URL like
  `https://huggingface.co/spaces/your-username/weave`.

### 2. Add the database secret
In the Space's **Settings → Variables and secrets → New secret**:
- Name: `DATABASE_URL`
- Value: the Supabase **Session pooler** connection string (see the ⚠️ below)

HF injects secrets as **runtime environment variables**, which is exactly what
`config.py` reads. (Optionally also add `GEMINI_API_KEY`, but the app is
designed to take the Gemini key from the browser, so it's not required.)

`POSTGRES_SSLMODE=require` is already baked into the `Dockerfile`, so Supabase
TLS works out of the box.

> ### ⚠️ Use the Session **pooler** URL, not the direct connection (learned the hard way)
> Supabase's **direct** host `db.<ref>.supabase.co` is **IPv6-only** on the free
> plan. Fly has IPv6 so it works there, but **HF Spaces have no IPv6** — the direct
> URL fails on HF with `could not translate host name ... Name or service not known`.
>
> Fix: in Supabase, click the green **Connect** button → **Session pooler** tab
> (port 5432) and use *that* string. The shared pooler host
> `aws-<n>-<region>.pooler.supabase.com` has an **IPv4** address and is free — do
> **not** buy the $4/mo "dedicated IPv4" add-on. Note the different username
> (`postgres.<ref>`) and host:
> ```
> postgresql://postgres.<ref>:<pw>@aws-1-us-west-1.pooler.supabase.com:5432/postgres
> ```
> Use this same pooler URL in your root `.env` too, so local dev / Fly / HF all
> work identically regardless of IPv6.
>
> **Also:** free Supabase projects **auto-pause after ~7 days idle** — the host
> stops resolving entirely (same DNS error). If you hit it, resume the project in
> the dashboard and wait for **Active** before restarting the Space.
>
> **Applying a changed secret:** if the Space already crashed, saving the secret
> may not auto-apply — hit **Settings → Factory reboot** to restart with it.

### 3. Push this repo to the Space
The Space is its own git repo. Add it as a second remote and push:

```bash
# from the repo root
git remote add space https://huggingface.co/spaces/your-username/weave
git push space main
```

If HF asks for credentials, use your HF username + an access token
(https://huggingface.co/settings/tokens, "write" scope) as the password.

That's it. HF builds the root `Dockerfile`, and the Space goes live at
`https://your-username-weave.hf.space`.

---

## How it works (why the root `Dockerfile` differs from `Dockerfile.api`)

| | `Dockerfile` (HF) | `Dockerfile.api` (Fly) |
|---|---|---|
| Runtime user | non-root, UID 1000 (`$HOME=/home/user`) | root |
| Port | 7860 (declared via `app_port` in `README.md`) | 8080 |
| Model cache | `/home/user/.model_cache` (writable) | `/app/.model_cache` |
| Secrets | HF Settings → runtime env | `fly secrets set` |

The HF front-matter that configures the Space lives at the **top of the repo
`README.md`** (`sdk: docker`, `app_port: 7860`). GitHub renders that YAML block
as a harmless metadata table.

Both Dockerfiles build the same app from the same source, so they won't drift
in behavior — only in host conventions. Once you're confident on HF, you can
delete `Dockerfile.api`, `fly.toml`, `deploy/deploy.sh`, and the
`.github/workflows/deploy-fly.yml` workflow to fully cut Fly loose.

---

## Fly.io status (migration complete)

Fly auto-deploy is already **off** — `.github/workflows/deploy-fly.yml` is
manual-only (`workflow_dispatch`), so pushes to `main` don't trigger paid
redeploys. As of the HF cutover, the Fly app `siip-armchair-akhil` was replaced
by a tiny **nginx 301 redirect** (`deploy/redirect/`, deployed via
`deploy-fly-redirect.yml`) so the old URL forwards to this Space at ≈$0. The
full-app `fly.toml` / `deploy-fly.yml` are kept only as a fallback. Destroy the
Fly app entirely once no one relies on the old `*.fly.dev` link.

(Pushing to the `space` remote does **not** touch GitHub Actions — only pushing
to the GitHub `origin` does.)

---

## Redeploying after code changes
```bash
git push space main   # rebuilds the Space
```
No script, no CLI login — HF rebuilds on push.

## Running maintenance scripts
Free Spaces don't offer an SSH console. To run one-off ops (e.g.
`scripts/restore_from_dump.py`), run them locally against the same Supabase
`DATABASE_URL` — the whole point of the single-pipeline design.
