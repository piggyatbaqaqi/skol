# Redis target switcher — testing skol against tsqali on puchpuchobs

Phase 3.5 of the skol→tsqali migration: validate that the skol app
stack (Django, bin/ scripts) talks correctly to tsqali's single-node
Redis cluster, before cutting prod over in Phase 4.  The mechanism is
two sourceable shell scripts that flip the operator's shell env (and
the local docker container's state) between two targets:

| target  | REDIS_HOST                  | REDIS_PORT | cluster | local docker state |
|---------|-----------------------------|------------|---------|--------------------|
| `local` | `localhost`                 | `6380`     | no      | started            |
| `tsqali`| `skol.synoptickeyof.life`   | `6381`     | yes     | stopped            |

When `tsqali` is active, puchpuchobs's local redis container is **stopped**
so any code path that accidentally hits `localhost:6380` fails fast with
`Connection refused` instead of silently reading stale local data.

## Prerequisites

These env vars must be set in `~/.bashrc` (or wherever your shell
loads its environment) before sourcing the tsqali switcher:

```
TSQALI_REDIS_USERNAME=admin
TSQALI_REDIS_PASSWORD=...
TSQALI_REDIS_CACERT=/etc/letsencrypt/live/synoptickeyof.life/fullchain.pem
# Optional — defaults shown:
# TSQALI_REDIS_HOST=skol.synoptickeyof.life
# TSQALI_REDIS_PORT=6381
```

The local switcher requires nothing extra — local creds come from
`/home/skol/.skol_env` via env_config.py's fallback chain.

## Usage

In any shell where you'll launch a skol process (Django dev server,
bin/ script, etc.):

```bash
source bin/skol-redis-env-tsqali.sh   # switch to tsqali, stop local
# ... launch Django, run scripts ...
source bin/skol-redis-env-local.sh    # switch back to local, start local
```

The switchers print one line confirming the new target.  Sanity-check
anytime with:

```bash
env | grep ^REDIS_   # see what's actually exported
```

Round-tripping (local → tsqali → local) restores the original
`REDIS_*` values via a snapshot stored in `SKOL_REDIS_ENV_SAVED_*`,
so the second `local` invocation puts you exactly back where you
started — no manual cleanup of overrides needed.

## Optional: show the active target in your shell prompt

Each switcher exports `SKOL_REDIS_TARGET=tsqali` or `=local`.  Add
this snippet to `~/.bashrc` to surface it as a `[skol:tsqali]` /
`[skol:local]` prefix on your shell prompt:

```bash
# Show the active SKOL Redis target in the prompt.  Place AFTER the
# conda init block (so $CONDA_PROMPT_MODIFIER is already exported)
# and AFTER any line that re-assigns PS1.  The guard prevents
# re-prepending if .bashrc is re-sourced (the indicator is already
# there).  The :+ expansion means: only show the prefix when
# SKOL_REDIS_TARGET is set — no indicator before either switcher
# has been sourced (avoiding misleading 'local' in a default shell).
if [[ "$PS1" != *SKOL_REDIS_TARGET* ]]; then
    PS1='${SKOL_REDIS_TARGET:+[skol:$SKOL_REDIS_TARGET] }'"$PS1"
fi
```

How this composes with the things already in your `PS1`:

- **Conda's `(envname)` prefix** — set via `$CONDA_PROMPT_MODIFIER`,
  which conda re-exports on each `activate`/`deactivate`.  We don't
  touch that variable; our prefix just sits in front of (or behind)
  it depending on where in PS1 you place this snippet.
- **Terminal title escapes** — the `\[\e]0;…\a\]` sequence inside PS1
  that updates the xterm/iTerm title bar stays intact.  Our prefix
  is plain text, doesn't break the escape.
- **PS1 re-assignment** — if a later line in `~/.bashrc` does
  `PS1='...'` (overwriting rather than appending), our prefix is
  lost.  Put the snippet AFTER any such line.  The guard makes
  it safe to source `.bashrc` multiple times.

After adding and re-sourcing `.bashrc`, the prompt looks something
like:

```
(skol3.14) [skol:tsqali] piggy@puchpuchobs:~/src/skol$
```

— at a glance you know which Redis the next bin/ script or Django
launch will hit.

## Test matrix

Run each step against `local` first to capture the baseline, then
re-run against `tsqali` to compare.  Tests are roughly ordered by
"if this works, I can sleep at night."

### 1. Wire-level smoke test

```bash
python -c "
import sys
sys.path.insert(0, 'bin')
from env_config import create_redis_client
r = create_redis_client()
print(type(r).__name__, '->', r.ping(), r.dbsize())
"
```

Expected:
- `local`: `Redis -> True <small number>`
- `tsqali`: `RedisCluster -> True 14562977` (or whatever the current count is)

If the type isn't right, env_config didn't see `REDIS_CLUSTER_MODE`.
If ping fails, network/TLS/auth issue.

### 2. Django HTTP

Start the dev server in the shell with the env active:

```bash
cd django && python manage.py runserver 0.0.0.0:8000
```

Hit the following URLs (browser or curl):

| Endpoint                  | What it exercises                                  |
|---------------------------|----------------------------------------------------|
| `/`                       | Basic page render, no redis needed                 |
| `/api/embeddings/`        | `EmbeddingListView` — uses `scan_iter`, cluster-safe |
| `/api/vocab-tree/versions/` | `VocabTreeVersionsView` — also `scan_iter`        |
| `/search/?q=Pleurotus`    | Full search path: embedding lookup + sort by similarity |

Expected: identical responses against both targets (give or take a
hundred ms of tunnel latency on tsqali).

### 3. Read-mostly bin/ script

```bash
bin/build_sources_stats.py --experiment production_v4 --verbosity 2
```

Reads many `skol:sources*` keys, computes per-source statistics,
writes one summary key.  Output should be identical against both
targets (modulo source data drift since the mirror).

### 4. Read+write bin/ script with embeddings

Pick one document, run embed_treatments on it:

```bash
bin/embed_treatments.py --experiment production_v4 --doc-id <some-doc-id> --force
```

Watch for:
- `redis.cluster.RedisCluster` mentioned in the connection log
- No `MOVED` redirect errors (cluster client should follow transparently)
- No `LOADING` errors (cluster bootstrap should be complete by now)
- Embedding key written successfully (verify via DBSIZE delta or a
  direct GET of the new key)

### 5. End-to-end pipeline (optional, high-confidence)

Kick off a small predict run:

```bash
bin/predict_v4.py --experiment production_v4 --source-db skol_dev --limit 5
```

Exercises:
- Classifier model load (large value GET)
- Per-line embedding lookups (many small GETs)
- Per-line embedding writes (many small SETs)
- The full v4 layout+treatment CRF pipeline

If this works on both targets, the rest of skol almost certainly
does too.

## Rollback

```bash
source bin/skol-redis-env-local.sh
```

That's it.  Local container starts (if stopped), env vars restored to
their pre-tsqali values.  Any long-running process launched in the
tsqali env needs to be restarted in the local-env shell to pick up
the change.

## What to watch for (expected differences, NOT bugs)

- **Latency.** Every tsqali op goes through the SSH reverse tunnel
  via skol.synoptickeyof.life and back.  Single ops are sub-100ms;
  tight loops over thousands of keys are noticeably slower.  If a
  bin/ script that took 30 sec now takes 5 min, that's tunnel
  overhead, not a bug.
- **`DBSIZE`** returns per-node size on a RedisCluster client.  With
  a single-node cluster owning all slots, this is the total.  When
  we eventually add nodes, this becomes per-node and you'd want
  `CLUSTER COUNTKEYSINSLOT` in a loop instead.
- **`KEYS *`** raises immediately on the cluster client.  All in-tree
  callers were converted to `scan_iter` in commit `16d117a`; if a
  new one shows up, it'll be loud rather than silent.

## Implementation notes

The switchers are intentionally sourced (not exec'd) so they can
modify the parent shell's env.  Side effect: they also call
`docker compose stop/start redis` on `/opt/skol/advanced-databases`,
which IS a host-wide change — every shell on the box loses access to
the local redis when one shell switches to tsqali.  That's the point
("unambiguous which server is getting hit"); if you need multiple
shells with different targets simultaneously, this design won't fit
and you'd want per-process env overrides instead.

When done with all testing, leave the shell on `local` so the docker
container is back up — the next session starts in a sane state.
