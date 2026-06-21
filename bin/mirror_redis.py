#!/usr/bin/env python3
"""One-shot mirror of one Redis server's contents to another.

Snapshots the source Redis (via ``redis-cli --rdb``, which uses the SYNC
protocol — no filesystem coordination needed on the source side), ships
the RDB file to the target via SSH, and atomically swaps it into place
on the target's Docker volume mount (stop the redis service → backup
the old dump → install the new dump → start the redis service → wait
for RDB load → convert to AOF → restart cleanly).

On Redis 7+ with ``appendonly yes`` (skol's config), a freshly-placed
dump.rdb is NOT loaded on startup unless we first flip appendonly to
no.  So the script edits ``redis.conf`` to ``appendonly no`` before
starting the container, waits for the RDB load to finish, then uses
``CONFIG SET appendonly yes`` to trigger an immediate AOF rewrite
that captures the in-memory state into a fresh AOF base file, and
finally restores ``appendonly yes`` in the conf so future restarts
load from the (now-populated) AOF.  An EXIT trap ensures the conf
restoration happens even if the script aborts partway.

All three skol hosts (puchpuchobs, tsqali, skol.synoptickeyof.life) run
Redis under ``docker compose``, so the swap goes through ``docker
compose stop redis`` / ``docker compose start redis`` against the
deployed compose file at ``/opt/skol/advanced-databases/docker-compose.yaml``.

The target Redis is **overwritten**.  Any keys present on the target
that aren't on the source are lost.  A timestamped backup of the
previous dump.rdb is left on the target box for rollback.

Usage::

    bin/mirror_redis.py                              # LOCAL -> SKOL
    bin/mirror_redis.py --from TSQALI --to SKOL
    bin/mirror_redis.py --insecure                   # skip TLS verification

Each ``--from`` / ``--to`` value names an env-var prefix.  The special
prefix ``LOCAL`` means "no prefix — use bare ``REDIS_*`` env vars",
which matches the convention on puchpuchobs.  Variables read per
prefix:

==================================  ==========================================
``${PREFIX}_REDIS_HOST``            Required.  Redis hostname.
``${PREFIX}_REDIS_PORT``            Default ``6379``.
``${PREFIX}_REDIS_USERNAME``        Required.  ACL user.
``${PREFIX}_REDIS_PASSWORD``        Required.  ACL password (passed via
                                    ``REDISCLI_AUTH`` env, not argv, so it
                                    stays out of ``ps``).
``${PREFIX}_REDIS_CACERT``          Path to CA bundle.  If set, TLS is used
                                    for that prefix.  For SKOL this is
                                    typically
                                    ``/etc/letsencrypt/live/synoptickeyof.life/fullchain.pem``
                                    (root-readable — run as root).
``${PREFIX}_SSH_HOST``              SSH endpoint for the file swap (target
                                    only).  Defaults to ``${PREFIX}_REDIS_HOST``.
``${PREFIX}_SSH_USER``              SSH user (target only).  Defaults to the
                                    current local user (``getpass.getuser()``)
                                    — root SSH is not required.
``${PREFIX}_COMPOSE_FILE``          Path to docker-compose.yaml on the target
                                    host.  Defaults to
                                    ``/opt/skol/advanced-databases/docker-compose.yaml``.
``${PREFIX}_COMPOSE_SERVICE``       Redis service name in the compose file.
                                    Defaults to ``redis``.
``${PREFIX}_REDIS_CONF``            Path to redis.conf on the target host (used
                                    to temporarily flip appendonly during the
                                    swap).  Defaults to
                                    ``/opt/skol/advanced-databases/redis.conf``.
==================================  ==========================================

For ``LOCAL`` (sentinel), drop the prefix from the variable name: just
``REDIS_HOST``, ``REDIS_PORT``, etc.

Privilege model on the SSH target:

  * rsync (over ssh, with ``--partial`` so interrupted transfers
    resume from where they stopped) ships the snapshot to
    ``/data/tmp`` as the SSH user — no root needed (the directory
    is sticky-world-writable, 1777, created by skol's postinst).
  * The swap script (docker, cp into the redis volume, chown to the
    redis container's UID, mv aside the AOF) needs root.  It runs
    under a single ``sudo bash <script>`` invocation via ``ssh -t``.
    Sudo prompts for the SSH user's password ONCE (interactive TTY)
    and the cached credentials cover the rest of the run.

For fully unattended operation (e.g. cron), add a NOPASSWD entry on
the target like::

    <ssh_user> ALL=(root) NOPASSWD: /usr/bin/bash /data/tmp/mirror_redis_swap.sh

Run locally as whoever can read ``${PREFIX}_REDIS_CACERT`` — LE certs
are typically root-only, so this script is usually invoked as root
even though the SSH side isn't.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Snapshot files live under /data/tmp (a real disk-backed tmpdir available
# on all three skol hosts), not /tmp.  /tmp is tmpfs on these boxes, and
# the SBERT-cache RDB is ~52 GB — landing a snapshot there would OOM the
# host long before the transfer completed.
LOCAL_RDB = Path("/data/tmp/mirror_redis.rdb")
REMOTE_RDB = "/data/tmp/mirror_redis.rdb"
DEFAULT_COMPOSE_FILE = "/opt/skol/advanced-databases/docker-compose.yaml"
DEFAULT_COMPOSE_SERVICE = "redis"
DEFAULT_REDIS_CONF = "/opt/skol/advanced-databases/redis.conf"


def env_var_name(prefix: str, suffix: str) -> str:
    """Resolve a (prefix, suffix) pair to an env var name.

    The sentinel prefix ``LOCAL`` means "no prefix — use the bare suffix",
    matching the convention on puchpuchobs where Redis creds are exported
    as ``REDIS_HOST`` / ``REDIS_PORT`` / etc.  Any other prefix is joined
    with ``_``.
    """
    return suffix if prefix == "LOCAL" else f"{prefix}_{suffix}"


def get_env(prefix: str, suffix: str, default: str | None = None) -> str | None:
    return os.environ.get(env_var_name(prefix, suffix), default)


def require_env(prefix: str, suffix: str) -> str:
    val = get_env(prefix, suffix)
    if not val:
        sys.exit(f"ERROR: ${env_var_name(prefix, suffix)} is not set")
    return val


def redis_cli_argv(prefix: str, insecure: bool) -> list[str]:
    """redis-cli connection flags (no password — that goes via REDISCLI_AUTH env).

    TLS is enabled if any of: ``${PREFIX}_REDIS_CACERT`` is set,
    ``${PREFIX}_REDIS_TLS`` is ``yes``/``true``/``1``, or the global
    ``--insecure`` flag is passed.  Verification uses the cacert if
    available and ``--insecure`` wasn't passed; otherwise ``--insecure``
    skips both cert and hostname checks (needed for e.g. connecting to
    ``localhost`` when the server cert is for ``synoptickeyof.life``).
    """
    host = require_env(prefix, "REDIS_HOST")
    port = get_env(prefix, "REDIS_PORT") or "6379"
    user = require_env(prefix, "REDIS_USERNAME")
    args = ["-h", host, "-p", port, "--user", user, "--no-auth-warning"]
    cacert = get_env(prefix, "REDIS_CACERT")
    tls_explicit = (get_env(prefix, "REDIS_TLS") or "").lower() in ("yes", "true", "1")
    if cacert or tls_explicit or insecure:
        args.append("--tls")
        if insecure or not cacert:
            args.append("--insecure")
        else:
            args += ["--cacert", cacert]
    return args


def run_redis(prefix: str, insecure: bool, *cmd: str, capture: bool = False) -> str:
    """Invoke redis-cli with the given prefix's credentials and TLS settings.

    Password flows via REDISCLI_AUTH env var so it doesn't show up in
    /proc/<pid>/cmdline (and thus ps / journalctl).

    On non-zero exit when capture is True, prints the captured stdout
    and stderr to the terminal before re-raising — otherwise the
    operator sees only a CalledProcessError traceback with no clue
    what redis-cli actually complained about.
    """
    env = os.environ.copy()
    env["REDISCLI_AUTH"] = require_env(prefix, "REDIS_PASSWORD")
    argv = ["redis-cli", *redis_cli_argv(prefix, insecure), *cmd]
    try:
        result = subprocess.run(
            argv,
            env=env,
            capture_output=capture,
            text=capture,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if capture:
            if exc.stdout:
                print(exc.stdout, end="", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, end="", file=sys.stderr)
        raise
    return result.stdout.strip() if capture else ""


def config_get(prefix: str, insecure: bool, key: str) -> str:
    """Read a CONFIG GET value.  redis-cli prints two lines (key, value)."""
    raw = run_redis(prefix, insecure, "CONFIG", "GET", key, capture=True)
    return raw.splitlines()[-1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="src", default="LOCAL",
                   help="Source env-var prefix (default: LOCAL — bare REDIS_*)")
    p.add_argument("--to", dest="dst", default="SKOL",
                   help="Target env-var prefix (default: SKOL)")
    p.add_argument("--insecure", action="store_true",
                   help="Skip TLS verification (use for self-signed certs, or "
                        "when the cert SAN doesn't match the connection hostname)")
    p.add_argument("--reuse-snapshot", action="store_true",
                   help=f"Skip step 1 if {LOCAL_RDB} already exists on disk.  "
                        "Useful for resuming after a step 2-4 failure without "
                        "burning the snapshot time again — assumes the source "
                        "hasn't drifted enough since the original snapshot to "
                        "matter for your use case.")
    args = p.parse_args()

    src, dst, insecure = args.src, args.dst, args.insecure

    if args.reuse_snapshot and LOCAL_RDB.exists():
        size = LOCAL_RDB.stat().st_size
        print(f"[1/4] Reusing existing snapshot at {LOCAL_RDB} ({size:,} bytes).")
        print(f"      Source {src} may have drifted; pass without --reuse-snapshot "
              "to re-snapshot.")
    else:
        print(f"[1/4] Snapshotting {src} Redis -> {LOCAL_RDB}...")
        if LOCAL_RDB.exists():
            LOCAL_RDB.unlink()
        run_redis(src, insecure, "--rdb", str(LOCAL_RDB))
        size = LOCAL_RDB.stat().st_size
        print(f"  {size:,} bytes")

    print(f"[2/4] Locating {dst} dump.rdb in-container path...")
    target_dir = config_get(dst, insecure, "dir").rstrip("/")
    target_file = config_get(dst, insecure, "dbfilename")
    print(f"  In-container path: {target_dir}/{target_file}")

    ssh_host = get_env(dst, "SSH_HOST") or require_env(dst, "REDIS_HOST")
    # Default SSH user is the operator's own account, not root.  /data/tmp
    # is sticky-world-writable (1777) so any user can scp into it; the
    # privileged swap operations get a single sudo wrapping the bash run
    # below.  Operators with NOPASSWD sudo for /usr/bin/bash (or for the
    # swap script path) run unattended.  Without NOPASSWD, sudo prompts
    # once via the ssh -t TTY and caches credentials for the rest of the
    # session.
    ssh_user = get_env(dst, "SSH_USER") or getpass.getuser()
    ssh_target = f"{ssh_user}@{ssh_host}"
    compose_file = get_env(dst, "COMPOSE_FILE") or DEFAULT_COMPOSE_FILE
    compose_service = get_env(dst, "COMPOSE_SERVICE") or DEFAULT_COMPOSE_SERVICE
    redis_conf = get_env(dst, "REDIS_CONF") or DEFAULT_REDIS_CONF

    print(f"[3/4] Shipping snapshot to {ssh_target} and swapping in place...")
    # rsync over ssh, not scp: --partial preserves whatever bytes
    # already made it on a previous run, so an interrupted 47G transfer
    # resumes from where it stopped instead of restarting at zero.
    # --progress shows live bytes/sec during the transfer.  -a is
    # archive mode (preserves perms/times); we don't strictly need it
    # for a single file but it's cheap and idiomatic.
    subprocess.run(
        ["rsync", "-av", "--partial", "--progress",
         str(LOCAL_RDB), f"{ssh_target}:{REMOTE_RDB}"],
        check=True,
    )

    # Go-template for docker inspect: print the host-side Source for the
    # bind mount whose Destination matches the in-container dump.rdb dir.
    # Building this outside the f-string keeps the {{ }} escapes legible.
    docker_format = (
        '{{ range .Mounts }}{{ if eq .Destination "' + target_dir + '" }}'
        '{{ .Source }}{{ end }}{{ end }}'
    )
    remote_script = f"""\
set -euo pipefail
COMPOSE_FILE={shlex.quote(compose_file)}
SERVICE={shlex.quote(compose_service)}
CONF={shlex.quote(redis_conf)}
DBFILENAME={shlex.quote(target_file)}

CID=$(docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE")
if [ -z "$CID" ]; then
    echo "ERROR: no container running for service '$SERVICE' in $COMPOSE_FILE" >&2
    exit 1
fi
HOST_DIR=$(docker inspect "$CID" --format {shlex.quote(docker_format)})
if [ -z "$HOST_DIR" ]; then
    echo "ERROR: no bind mount with destination {target_dir} on container $CID" >&2
    exit 1
fi
TARGET_RDB="$HOST_DIR/$DBFILENAME"
echo "  Target host path: $TARGET_RDB"
echo "  Redis conf path:  $CONF"

# Ensure 'appendonly yes' is restored in the conf on script exit, even
# if we abort partway.  Otherwise a half-finished mirror leaves the
# host in a state where every container restart loads dump.rdb and
# ignores subsequent AOF writes — silent data loss waiting to happen.
trap 'sed -i "s/^appendonly no$/appendonly yes/" "$CONF" 2>/dev/null || true' EXIT

docker compose -f "$COMPOSE_FILE" stop "$SERVICE"

# Single timestamp shared by every backup taken in this swap so they
# all line up when reading 'ls' later.
TS=$(date +%Y%m%d-%H%M%S)

# Back up the RDB.
BACKUP="${{TARGET_RDB}}.before-mirror-$TS"
if [ -f "$TARGET_RDB" ]; then
    mv "$TARGET_RDB" "$BACKUP"
    echo "  Backed up old rdb: $BACKUP"
fi

# Back up any AOF artifacts.  Move (don't rm) so a failed mirror is
# recoverable.  Layout differs by Redis version:
#   Redis 7.x: multi-part bundle under $HOST_DIR/appendonlydir/
#              (manifest + .base.rdb + .incr.aof files)
#   Redis 6.x: single $HOST_DIR/appendonly.aof file
# Handle both; mv -T avoids mv-into-dir if the backup name somehow
# already exists.
if [ -d "$HOST_DIR/appendonlydir" ]; then
    mv -T "$HOST_DIR/appendonlydir" "$HOST_DIR/appendonlydir.before-mirror-$TS"
    echo "  Backed up appendonlydir/ (Redis 7.x AOF bundle)"
fi
if [ -f "$HOST_DIR/appendonly.aof" ]; then
    mv "$HOST_DIR/appendonly.aof" "$HOST_DIR/appendonly.aof.before-mirror-$TS"
    echo "  Backed up appendonly.aof (Redis 6.x legacy)"
fi

cp {shlex.quote(REMOTE_RDB)} "$TARGET_RDB"
# Match ownership of the volume dir so the in-container redis user can
# read (and later rewrite) the file.
chown --reference="$HOST_DIR" "$TARGET_RDB"
chmod 660 "$TARGET_RDB"

# Disable appendonly in the conf BEFORE starting redis.  Reason:
# Redis 7+ with 'appendonly yes' and no AOF manifest on disk does NOT
# fall back to dump.rdb — it starts empty and creates a fresh empty
# AOF.  The dump.rdb we just placed would be ignored entirely.
# Flipping appendonly to no makes redis load from dump.rdb; we flip
# it back to yes via CONFIG SET (which triggers an immediate AOF
# rewrite of the just-loaded state) once the load completes.
sed -i 's/^appendonly yes$/appendonly no/' "$CONF"
echo "  Temporarily set appendonly=no in $CONF (for the RDB load)"

docker compose -f "$COMPOSE_FILE" start "$SERVICE"

# Wait for the RDB load to complete.  PING starts succeeding as soon
# as the listener is up; CONFIG, DBSIZE, etc. reply LOADING until
# the in-memory load finishes.  47G takes ~80 sec on nvme; allow
# up to an hour for slower disks or larger datasets.
#
# Auth: docker compose exec runs the redis-cli inside the container,
# where REDIS_PASSWORD is already in env via env_file.  The single
# quotes around the sh -c argument preserve the literal $REDIS_PASSWORD
# so it's expanded INSIDE the container (not in our outer bash, where
# the var isn't defined).
echo "  Waiting for redis to finish loading dataset..."
i=0
while true; do
    LOADING=$(docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" sh -c \\
        'redis-cli --tls --insecure -h 127.0.0.1 -p 6379 --user admin -a "$REDIS_PASSWORD" --no-auth-warning INFO persistence' \\
        2>/dev/null | grep ^loading: | tr -d '\\r' | cut -d: -f2)
    [ "${{LOADING:-1}}" = "0" ] && break
    i=$((i + 1))
    if [ "$i" -ge 7200 ]; then
        echo "ERROR: redis did not finish loading within 1 hour" >&2
        exit 1
    fi
    sleep 0.5
done
LOADED=$(docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" sh -c \\
    'redis-cli --tls --insecure -h 127.0.0.1 -p 6379 --user admin -a "$REDIS_PASSWORD" --no-auth-warning DBSIZE')
echo "  Loaded keys: $LOADED"

# Flip to AOF.  CONFIG SET appendonly yes triggers an immediate
# background AOF rewrite that writes the in-memory state to a fresh
# appendonlydir/appendonly.aof.X.base.rdb.  Canonical 'convert from
# RDB-only to AOF-enabled' procedure.
echo "  Flipping appendonly=yes (triggers AOF rewrite)..."
docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" sh -c \\
    'redis-cli --tls --insecure -h 127.0.0.1 -p 6379 --user admin -a "$REDIS_PASSWORD" --no-auth-warning CONFIG SET appendonly yes' \\
    > /dev/null

# Wait for the AOF rewrite to complete.  Comparable size to the RDB,
# so similar time budget.
echo "  Waiting for AOF rewrite to complete..."
i=0
while true; do
    REWRITING=$(docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" sh -c \\
        'redis-cli --tls --insecure -h 127.0.0.1 -p 6379 --user admin -a "$REDIS_PASSWORD" --no-auth-warning INFO persistence' \\
        2>/dev/null | grep ^aof_rewrite_in_progress: | tr -d '\\r' | cut -d: -f2)
    [ "${{REWRITING:-1}}" = "0" ] && break
    i=$((i + 1))
    if [ "$i" -ge 7200 ]; then
        echo "ERROR: AOF rewrite did not complete within 1 hour" >&2
        exit 1
    fi
    sleep 1
done
AOF_SIZE=$(docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" sh -c \\
    'redis-cli --tls --insecure -h 127.0.0.1 -p 6379 --user admin -a "$REDIS_PASSWORD" --no-auth-warning INFO persistence' \\
    2>/dev/null | grep ^aof_base_size: | tr -d '\\r' | cut -d: -f2)
echo "  AOF rewrite complete: aof_base_size=$AOF_SIZE bytes"

# Restore appendonly=yes in the conf so future container restarts
# load from the (now-populated) AOF.  The trap above also runs this
# on error paths; doing it explicitly here means the trap becomes a
# no-op (sed find-no-match) on normal exit.
sed -i 's/^appendonly no$/appendonly yes/' "$CONF"
echo "  Restored appendonly=yes in $CONF"

rm {shlex.quote(REMOTE_RDB)}
"""

    # Two-call SSH so we can do the privileged work via 'sudo bash' with
    # a TTY for the password prompt — 'ssh -t' conflicts with feeding the
    # script via stdin (-tt + stdin redirection garbles input on some
    # terminals).  So:
    #   1. First SSH (no TTY, stdin = script): write the script to a
    #      tempfile under /data/tmp (writable by any user, 1777).
    #   2. Second SSH (-t for TTY): sudo bash the tempfile.  sudo prompts
    #      for password if needed; one prompt covers the whole script
    #      since everything runs inside that one sudo invocation.
    #      Tempfile cleanup happens inside the SAME ssh -t call so
    #      stranding it on failure is rare.
    swap_path = "/data/tmp/mirror_redis_swap.sh"
    subprocess.run(
        ["ssh", ssh_target,
         f"cat > {swap_path} && chmod 700 {swap_path}"],
        input=remote_script,
        text=True,
        check=True,
    )
    subprocess.run(
        ["ssh", "-t", ssh_target,
         f"sudo bash {swap_path}; rc=$?; rm -f {swap_path}; exit $rc"],
        check=True,
    )

    print("[4/4] Comparing DBSIZE...")
    src_size = run_redis(src, insecure, "DBSIZE", capture=True)
    dst_size = run_redis(dst, insecure, "DBSIZE", capture=True)
    print(f"  {src} DBSIZE: {src_size}")
    print(f"  {dst} DBSIZE: {dst_size}")
    if src_size != dst_size:
        print("  WARNING: sizes differ — source may have moved keys since snapshot")
        print("           started (writes during the snapshot window are normal).")
        print("           For a frozen comparison, pause writes on the source first.")

    LOCAL_RDB.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
