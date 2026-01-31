#!/bin/sh
# Redis entrypoint script
# Substitutes REDIS_PASSWORD environment variable into the config file

set -e

CONFIG_FILE="/usr/local/etc/redis/redis.conf"
CONFIG_TEMPLATE="/usr/local/etc/redis/redis.conf.template"

if [ -z "$REDIS_PASSWORD" ]; then
    echo "ERROR: REDIS_PASSWORD environment variable is not set" >&2
    exit 1
fi

# Copy template to working config
cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"

# Substitute password placeholder with actual password
# Using sed with different delimiter since password might contain special chars
sed -i "s|PLACEHOLDER_WILL_BE_SET_BY_ENTRYPOINT|${REDIS_PASSWORD}|g" "$CONFIG_FILE"

echo "Redis configuration initialized with authentication"

# Start Redis with the config file
exec redis-server "$CONFIG_FILE"
