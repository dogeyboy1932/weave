# Deploying ArmChair to Hugging Face Spaces (free, $0/mo)

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
- **Owner / name:** e.g. `your-username/armchair`
- **SDK:** Docker → **Blank** template
- **Hardware:** CPU basic (free)
- Create it. You'll get a git URL like
  `https://huggingface.co/spaces/your-username/armchair`.

### 2. Add the database secret
In the Space's **Settings → Variables and secrets → New secret**:
- Name: `DATABASE_URL`
- Value: the same Supabase connection string from your root `.env`

HF injects secrets as **runtime environment variables**, which is exactly what
`config.py` reads. (Optionally also add `GEMINI_API_KEY`, but the app is
designed to take the Gemini key from the browser, so it's not required.)

`POSTGRES_SSLMODE=require` is already baked into the `Dockerfile`, so Supabase
TLS works out of the box.

### 3. Push this repo to the Space
The Space is its own git repo. Add it as a second remote and push:

```bash
# from the repo root
git remote add space https://huggingface.co/spaces/your-username/armchair
git push space main
```

If HF asks for credentials, use your HF username + an access token
(https://huggingface.co/settings/tokens, "write" scope) as the password.

That's it. HF builds the root `Dockerfile`, and the Space goes live at
`https://your-username-armchair.hf.space`.

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

## ⚠️ Turn off the Fly auto-deploy (avoid surprise charges)

`.github/workflows/deploy-fly.yml` redeploys Fly on **every push to `main`** —
including the push you just made to update these files. While you're migrating,
either:

- **Disable the workflow:** GitHub → Actions → "Deploy to Fly.io" → ⋯ →
  *Disable workflow*, **or**
- Delete `.github/workflows/deploy-fly.yml`.

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
