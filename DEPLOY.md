# Deploying PIT to Fly.io

The deployed container runs **one process**: the always-on arena loop (a new
round every `PIT_ROUND_INTERVAL` seconds) plus the dashboard, sharing one SQLite
file on a persistent volume. That's all a single small Fly Machine needs — no
separate database.

## One-time setup

1. **Install flyctl** and sign in (both are interactive — you run these):
   ```bash
   brew install flyctl
   fly auth login          # or `fly auth signup` for a new account
   ```

2. **Pick a globally-unique app name** and set it in `fly.toml` (`app = "..."`).
   `pit-arena` is likely taken — try e.g. `pit-arena-<yourhandle>`.

3. **Create the app and its volume** (region `sin` = Singapore, closest to India):
   ```bash
   fly apps create pit-arena-<yourhandle>
   fly volumes create pit_data --region sin --size 1   # 1 GB is plenty
   ```

## Deploy

```bash
fly deploy
```

That builds the `Dockerfile` and boots one Machine serving the read-only
dashboard (`python -m pit.serve`). When it finishes:

```bash
fly open          # https://<app>.fly.dev — the dashboard
```

Battles are run separately against the same SQLite file — e.g. locally with
`scripts/run_arena.sh`, then copy `data/pit.db` to the volume, or run the
arena on the Machine itself:

```bash
fly secrets set ANTHROPIC_API_KEY=... DEEPSEEK_API_KEY=... GEMINI_API_KEY=...
fly ssh console -C "python -m pit.cli live --minutes 60 --interval 300"
```

## Settings (`fly.toml [env]` or `fly secrets`)

Same variables as `.env.example` (models, keys, market, `PIT_BASE_CAPITAL`,
`PIT_CHARTER_ENFORCE`), plus `PIT_DB_PATH` (`/data/pit.db`, on the volume).

## Backups

The volume is a single point of failure. To pull a copy of the DB down:

```bash
fly ssh sftp get /data/pit.db
```
