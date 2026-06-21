# Source this file to point skol app code (Django, bin/ scripts) back
# at puchpuchobs's local Redis AND start the local redis container if
# it isn't running.
#
# Usage:    source bin/skol-redis-env-local.sh
# Reverse:  source bin/skol-redis-env-tsqali.sh
# Runbook:  docs/redis-target-switcher.md
#
# If the tsqali switcher previously snapshotted REDIS_* values into
# SKOL_REDIS_ENV_SAVED_*, those get restored.  Otherwise this script
# explicitly sets the local-pointing values (REDIS_HOST=localhost,
# REDIS_PORT=6380, REDIS_CLUSTER_MODE=no) but leaves
# REDIS_USERNAME/PASSWORD/CACERT untouched — env_config.py falls
# back to /home/skol/.skol_env for those.

_skol_die() {
    echo "ERROR: $1" >&2
    return 1 2>/dev/null || exit 1
}

# --- start the local docker redis if it's stopped ---
_compose_dir=/opt/skol/advanced-databases
if [ -d "$_compose_dir" ]; then
    if ! docker compose -f "$_compose_dir/docker-compose.yaml" ps --status running --services 2>/dev/null \
         | grep -qx redis; then
        echo "Starting local redis container..."
        (cd "$_compose_dir" && docker compose start redis) \
            || _skol_die "failed to start local redis container"
    fi
fi
unset _compose_dir

# --- restore snapshot if there is one; otherwise set local-pointing defaults ---
if [ "${SKOL_REDIS_ENV_SAVED:-}" = "1" ]; then
    # Restore exactly what was there before the tsqali switcher ran.
    # Use 'unset; export' rather than 'export X=$Y' so empty values
    # (from a shell that didn't have a given var set) become unset
    # rather than empty-string, which env_config.py treats as "not set".
    if [ -n "$SKOL_REDIS_ENV_SAVED_HOST" ]; then export REDIS_HOST="$SKOL_REDIS_ENV_SAVED_HOST"; else unset REDIS_HOST; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_PORT" ]; then export REDIS_PORT="$SKOL_REDIS_ENV_SAVED_PORT"; else unset REDIS_PORT; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_USERNAME" ]; then export REDIS_USERNAME="$SKOL_REDIS_ENV_SAVED_USERNAME"; else unset REDIS_USERNAME; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_PASSWORD" ]; then export REDIS_PASSWORD="$SKOL_REDIS_ENV_SAVED_PASSWORD"; else unset REDIS_PASSWORD; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_TLS" ]; then export REDIS_TLS="$SKOL_REDIS_ENV_SAVED_TLS"; else unset REDIS_TLS; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_CACERT" ]; then export REDIS_CACERT="$SKOL_REDIS_ENV_SAVED_CACERT"; else unset REDIS_CACERT; fi
    if [ -n "$SKOL_REDIS_ENV_SAVED_CLUSTER_MODE" ]; then export REDIS_CLUSTER_MODE="$SKOL_REDIS_ENV_SAVED_CLUSTER_MODE"; else unset REDIS_CLUSTER_MODE; fi
    unset SKOL_REDIS_ENV_SAVED \
          SKOL_REDIS_ENV_SAVED_HOST SKOL_REDIS_ENV_SAVED_PORT \
          SKOL_REDIS_ENV_SAVED_USERNAME SKOL_REDIS_ENV_SAVED_PASSWORD \
          SKOL_REDIS_ENV_SAVED_TLS SKOL_REDIS_ENV_SAVED_CACERT \
          SKOL_REDIS_ENV_SAVED_CLUSTER_MODE
    echo "skol app env: LOCAL (restored from snapshot)"
else
    # No snapshot to restore — set the basic local-pointing values
    # and let env_config.py pick up creds from /home/skol/.skol_env.
    export REDIS_HOST=localhost
    export REDIS_PORT=6380
    export REDIS_TLS=yes
    unset REDIS_CLUSTER_MODE
    echo "skol app env: LOCAL ($REDIS_HOST:$REDIS_PORT, single-node)"
fi
echo "Local redis container: running"
