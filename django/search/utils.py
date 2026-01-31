"""
Utility functions for the search app.
"""
import redis
from django.conf import settings


def get_redis_client(decode_responses: bool = False, db: int = 0) -> redis.Redis:
    """
    Create a Redis client with proper TLS and authentication configuration.

    Uses Django settings for connection parameters:
    - REDIS_HOST, REDIS_PORT: Connection endpoint
    - REDIS_USERNAME, REDIS_PASSWORD: Authentication (optional)
    - REDIS_TLS: Enable TLS (optional)

    Args:
        decode_responses: Whether to decode responses as strings
        db: Redis database number

    Returns:
        Configured Redis client
    """
    kwargs = {
        'host': settings.REDIS_HOST,
        'port': settings.REDIS_PORT,
        'db': db,
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

    return redis.Redis(**kwargs)
