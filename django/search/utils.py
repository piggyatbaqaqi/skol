"""
Utility functions for the search app.
"""
from typing import Optional

import redis
from django.conf import settings


def get_redis_client(
    decode_responses: bool = False,
    db: int = 0,
    cluster_mode: Optional[bool] = None,
):
    """
    Create a Redis client with proper TLS and authentication configuration.

    Uses Django settings for connection parameters:
    - REDIS_HOST, REDIS_PORT: Connection endpoint
    - REDIS_USERNAME, REDIS_PASSWORD: Authentication (optional)
    - REDIS_TLS: Enable TLS (optional)
    - REDIS_CLUSTER_MODE: Return RedisCluster instead of Redis (optional)

    Args:
        decode_responses: Whether to decode responses as strings
        db: Redis database number.  Ignored when cluster_mode is True
            (Redis Cluster only supports db 0).
        cluster_mode: When True, returns a redis.cluster.RedisCluster
            instance instead of redis.Redis.  Default: settings.REDIS_CLUSTER_MODE.

    Returns:
        Configured Redis client (redis.Redis or redis.cluster.RedisCluster).
    """
    use_cluster = (
        cluster_mode
        if cluster_mode is not None
        else getattr(settings, 'REDIS_CLUSTER_MODE', False)
    )

    kwargs = {
        'host': settings.REDIS_HOST,
        'port': settings.REDIS_PORT,
        'decode_responses': decode_responses,
    }

    # Add authentication if configured
    if hasattr(settings, 'REDIS_USERNAME') and settings.REDIS_USERNAME:
        kwargs['username'] = settings.REDIS_USERNAME
    if hasattr(settings, 'REDIS_PASSWORD') and settings.REDIS_PASSWORD:
        kwargs['password'] = settings.REDIS_PASSWORD

    # Configure TLS if enabled
    if getattr(settings, 'REDIS_TLS', False):
        kwargs['ssl'] = True
        kwargs['ssl_ca_certs'] = '/etc/ssl/certs/ca-certificates.crt'
        # Don't verify hostname (cert is for synoptickeyof.life but we connect to localhost)
        kwargs['ssl_check_hostname'] = False

    if use_cluster:
        # Redis Cluster has no multi-db concept (everything is on db 0).
        # If db != 0 here it's almost certainly a config mistake;
        # ignoring silently is cheaper than erroring at construction.
        from redis.cluster import RedisCluster
        return RedisCluster(**kwargs)

    kwargs['db'] = db
    return redis.Redis(**kwargs)
