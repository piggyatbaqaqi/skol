#!/bin/sh
# Redis entrypoint script.
#
# Renders the config template by substituting placeholders with environment
# variables, then either execs redis-server (non-cluster mode) or runs it
# in the background long enough to bootstrap CLUSTER ADDSLOTSRANGE before
# waiting on it (cluster mode).

set -e

CONFIG_FILE="/usr/local/etc/redis/redis.conf"
CONFIG_TEMPLATE="/usr/local/etc/redis/redis.conf.template"

if [ -z "$REDIS_PASSWORD" ]; then
    echo "ERROR: REDIS_PASSWORD environment variable is not set" >&2
    exit 1
fi

# Cluster-related defaults.  When REDIS_CLUSTER_ENABLED=no (the default),
# the announce-* settings are written to the config but unused by Redis,
# so the values don't have to be meaningful in non-cluster mode.
REDIS_CLUSTER_ENABLED="${REDIS_CLUSTER_ENABLED:-no}"
REDIS_CLUSTER_ANNOUNCE_IP="${REDIS_CLUSTER_ANNOUNCE_IP:-127.0.0.1}"
REDIS_CLUSTER_ANNOUNCE_PORT="${REDIS_CLUSTER_ANNOUNCE_PORT:-6379}"
REDIS_CLUSTER_ANNOUNCE_BUS_PORT="${REDIS_CLUSTER_ANNOUNCE_BUS_PORT:-16379}"

# Render the template.
cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"
sed -i \
    -e "s|PLACEHOLDER_WILL_BE_SET_BY_ENTRYPOINT|${REDIS_PASSWORD}|g" \
    -e "s|PLACEHOLDER_CLUSTER_ENABLED|${REDIS_CLUSTER_ENABLED}|g" \
    -e "s|PLACEHOLDER_CLUSTER_ANNOUNCE_IP|${REDIS_CLUSTER_ANNOUNCE_IP}|g" \
    -e "s|PLACEHOLDER_CLUSTER_ANNOUNCE_PORT|${REDIS_CLUSTER_ANNOUNCE_PORT}|g" \
    -e "s|PLACEHOLDER_CLUSTER_ANNOUNCE_BUS_PORT|${REDIS_CLUSTER_ANNOUNCE_BUS_PORT}|g" \
    "$CONFIG_FILE"

echo "Redis configuration rendered (cluster=${REDIS_CLUSTER_ENABLED}, announce=${REDIS_CLUSTER_ANNOUNCE_IP}:${REDIS_CLUSTER_ANNOUNCE_PORT})"

# Non-cluster mode: nothing to bootstrap.  Exec so redis-server becomes PID 1.
if [ "$REDIS_CLUSTER_ENABLED" != "yes" ]; then
    exec redis-server "$CONFIG_FILE"
fi

# Cluster mode: start redis in the background, bootstrap slots, then wait on
# the process so signals propagate and the container exits with redis's
# status.
redis-server "$CONFIG_FILE" &
REDIS_PID=$!

# All admin operations go via TLS+localhost.  --insecure because the cert is
# for synoptickeyof.life, not 127.0.0.1, and hostname verification has no
# security value when we're staying entirely in-container.
RCLI="redis-cli --tls --insecure -h 127.0.0.1 -p 6379 -a ${REDIS_PASSWORD} --no-auth-warning"

echo "Waiting for Redis to start..."
i=0
while ! $RCLI ping >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "ERROR: Redis did not become ready within 30s" >&2
        kill -TERM "$REDIS_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 0.5
done

# Bootstrap all 16384 slots if none have been assigned yet.  Redis persists
# slot ownership in cluster-config-file (nodes.conf), so this is a one-time
# op per data dir — restarts pick up the existing assignment.
SLOTS_ASSIGNED=$($RCLI CLUSTER INFO 2>/dev/null | grep ^cluster_slots_assigned: | tr -d '\r' | cut -d: -f2)
if [ "${SLOTS_ASSIGNED:-0}" -lt 16384 ]; then
    echo "Assigning all 16384 hash slots to this node..."
    $RCLI CLUSTER ADDSLOTSRANGE 0 16383
    echo "Slot assignment complete"
else
    echo "All 16384 slots already assigned (preserved from nodes.conf)"
fi

# Forward signals to redis and exit when it does.
wait $REDIS_PID
