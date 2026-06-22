# Source this file to point skol app code (Django, bin/ scripts) at
# tsqali's Redis cluster AND stop the local redis container so any
# accidental localhost:6380 connection fails fast with 'connection
# refused' instead of silently hitting the wrong server.
#
# Usage:    source bin/skol-redis-env-tsqali.sh
# Reverse:  source bin/skol-redis-env-local.sh
# Runbook:  docs/redis-target-switcher.md
#
# Reads (from current shell env — must be set in ~/.bashrc or similar):
#   TSQALI_REDIS_USERNAME       Required.  Usually 'admin'.
#   TSQALI_REDIS_PASSWORD       Required.
#   TSQALI_REDIS_CACERT         Required.  Path to LE CA bundle.
#   TSQALI_REDIS_HOST           Optional.  Defaults to skol.synoptickeyof.life
#                               (the public IP / SSH-tunnel endpoint).
#   TSQALI_REDIS_PORT           Optional.  Defaults to 6381.
#
# Sets / overrides:
#   REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD, REDIS_TLS,
#   REDIS_CACERT, REDIS_CLUSTER_MODE
#
# Snapshots any previous values into SKOL_REDIS_ENV_SAVED_* so the
# local switcher can restore them on round-trip.

_skol_die() {
    echo "ERROR: $1" >&2
    # 'return' if sourced, 'exit' if accidentally executed
    return 1 2>/dev/null || exit 1
}

# --- pre-flight: required env vars from caller's shell ---
for _v in TSQALI_REDIS_USERNAME TSQALI_REDIS_PASSWORD TSQALI_REDIS_CACERT; do
    if [ -z "${!_v:-}" ]; then
        _skol_die "\$$_v is not set; cannot configure tsqali env"
    fi
done

# --- snapshot existing REDIS_* (once, so repeated sourcing is idempotent) ---
if [ -z "${SKOL_REDIS_ENV_SAVED:-}" ]; then
    export SKOL_REDIS_ENV_SAVED_HOST="${REDIS_HOST:-}"
    export SKOL_REDIS_ENV_SAVED_PORT="${REDIS_PORT:-}"
    export SKOL_REDIS_ENV_SAVED_USERNAME="${REDIS_USERNAME:-}"
    export SKOL_REDIS_ENV_SAVED_PASSWORD="${REDIS_PASSWORD:-}"
    export SKOL_REDIS_ENV_SAVED_TLS="${REDIS_TLS:-}"
    export SKOL_REDIS_ENV_SAVED_CACERT="${REDIS_CACERT:-}"
    export SKOL_REDIS_ENV_SAVED_CLUSTER_MODE="${REDIS_CLUSTER_MODE:-}"
    export SKOL_REDIS_ENV_SAVED=1
fi

# --- stop the local docker redis if it's running ---
_compose_dir=/opt/skol/advanced-databases
if [ -d "$_compose_dir" ] && \
   docker compose -f "$_compose_dir/docker-compose.yaml" ps --status running --services 2>/dev/null \
   | grep -qx redis; then
    echo "Stopping local redis container..."
    (cd "$_compose_dir" && docker compose stop redis) \
        || _skol_die "failed to stop local redis container"
fi
unset _compose_dir

# --- set the tsqali-pointing env vars ---
export REDIS_HOST="${TSQALI_REDIS_HOST:-skol.synoptickeyof.life}"
export REDIS_PORT="${TSQALI_REDIS_PORT:-6381}"
export REDIS_USERNAME="$TSQALI_REDIS_USERNAME"
export REDIS_PASSWORD="$TSQALI_REDIS_PASSWORD"
export REDIS_TLS=yes
export REDIS_CACERT="$TSQALI_REDIS_CACERT"
export REDIS_CLUSTER_MODE=yes

# Surface the current target for prompt integration.  See
# docs/redis-target-switcher.md for the ~/.bashrc snippet that
# conditionally prepends [skol:tsqali] / [skol:local] to PS1.
export SKOL_REDIS_TARGET=tsqali

unset _v
echo "skol app env: TSQALI ($REDIS_HOST:$REDIS_PORT, cluster mode)"
echo "Local redis container: stopped"
