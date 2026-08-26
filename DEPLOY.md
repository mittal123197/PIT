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

That builds the `Dockerfile`, ships it, and boots one always-on Machine. When it
finishes:

```bash
fly open          # opens https://<app>.fly.dev — the live dashboard
fly logs          # watch the arena daemon run rounds
```

## Optional: LLM agents in the cloud

The default deploy runs the fast, free, deterministic agents. To let Groq drive
them, set the key as a secret and flip the flag:

```bash
fly secrets set GROQ_API_KEY=gsk_...      # stored encrypted, not in the image
fly deploy   # after uncommenting PIT_USE_LLM="1" in fly.toml [env]
```

Keep `PIT_ROUND_INTERVAL` generous (e.g. 3600) and `PIT_BARS` small (e.g. 24)
for LLM runs — every agent wake is an API call.

## Tuning knobs (`fly.toml [env]` or `fly secrets`)

| Var | Meaning | Default |
|-----|---------|---------|
| `PIT_ROUND_INTERVAL` | seconds between rounds | 300 |
| `PIT_BARS` | synthetic feed length per round | 120 |
| `PIT_USE_LLM` | `1` = Groq agents (needs `GROQ_API_KEY`) | 0 |
| `PIT_DAEMON` | `0` = dashboard only, no auto rounds | 1 |
| `PIT_DB_PATH` | SQLite path (points at the volume) | `/data/pit.db` |

## Backups

The volume is a single point of failure. To pull a copy of the DB down:

```bash
fly ssh console -C "cat /data/pit.db" > backup-$(date +%F).db   # small DBs
```

For anything bigger, `fly ssh sftp get /data/pit.db`.

## Notes

- `auto_stop_machines = false` keeps the Machine resident so the arena loop never
  pauses — this is the whole point vs. free tiers that sleep on idle.
- A local run uses the same code: `python3 -m pit.serve` (dashboard on :8080,
  daemon in a thread). Only the deploy target differs.
