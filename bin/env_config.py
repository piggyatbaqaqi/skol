"""
Centralized environment variable configuration for SKOL scripts.

This module provides a unified way to read configuration from environment
variables with sensible defaults, used by all bin scripts.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


def get_env_config() -> Dict[str, Any]:
    """
    Get complete environment configuration from environment variables or defaults.

    Returns:
        Dictionary of configuration values for all SKOL components
    """
    return {
        # CouchDB connection settings
        'couchdb_url': os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        'couchdb_host': os.environ.get('COUCHDB_HOST', '127.0.0.1:5984'),
        'couchdb_username': os.environ.get('COUCHDB_USER', 'admin'),
        'couchdb_password': os.environ.get('COUCHDB_PASSWORD', 'SU2orange!'),
        'couchdb_database': os.environ.get('COUCHDB_DATABASE', 'skol_dev'),

        # Ingest database settings (for extract_taxa_to_couchdb.py)
        'ingest_url': os.environ.get('INGEST_URL', os.environ.get('COUCHDB_URL', 'http://localhost:5984')),
        'ingest_database': os.environ.get('INGEST_DATABASE'),
        'ingest_username': os.environ.get('INGEST_USERNAME', os.environ.get('COUCHDB_USER')),
        'ingest_password': os.environ.get('INGEST_PASSWORD', os.environ.get('COUCHDB_PASSWORD')),
        'ingest_db_name': os.environ.get('INGEST_DB_NAME', 'skol_dev'),

        # Taxon database settings (for extract_taxa_to_couchdb.py)
        'taxon_url': os.environ.get('TAXON_URL'),
        'taxon_database': os.environ.get('TAXON_DATABASE'),
        'taxon_username': os.environ.get('TAXON_USERNAME'),
        'taxon_password': os.environ.get('TAXON_PASSWORD'),
        'taxon_db_name': os.environ.get('TAXON_DB_NAME', 'skol_taxa_dev'),

        # Training database settings
        'training_database': os.environ.get('TRAINING_DATABASE', 'skol_training'),

        # Redis settings
        'redis_host': os.environ.get('REDIS_HOST', 'localhost'),
        'redis_port': int(os.environ.get('REDIS_PORT', '6379')),
        'redis_url': os.environ.get('REDIS_URL', f"redis://{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}"),

        # Model settings
        'model_version': os.environ.get('MODEL_VERSION', 'v2.0'),
        'classifier_model_expire': os.environ.get('MODEL_EXPIRE', None),

        # Embedding settings
        'embedding_name': os.environ.get('EMBEDDING_NAME', 'skol:embedding:v1.1'),
        'embedding_expire': int(os.environ.get('EMBEDDING_EXPIRE', str(60 * 60 * 24 * 2))),  # 2 days default

        # Prediction settings
        'couchdb_pattern': os.environ.get('COUCHDB_PATTERN', '*.txt'),
        'pattern': os.environ.get('PATTERN', '*.txt.ann'),
        'prediction_batch_size': int(os.environ.get('PREDICTION_BATCH_SIZE', '24')),
        'num_workers': int(os.environ.get('NUM_WORKERS', '4')),

        # Data paths
        'annotated_path': Path(os.environ.get('ANNOTATED_PATH', Path.cwd().parent / "data" / "annotated")),

        # Spark settings
        'cores': int(os.environ.get('SPARK_CORES', '4')),
        'bahir_package': os.environ.get('BAHIR_PACKAGE', 'org.apache.bahir:spark-sql-cloudant_2.12:2.4.0'),
        'spark_driver_memory': os.environ.get('SPARK_DRIVER_MEMORY', '4g'),
        'spark_executor_memory': os.environ.get('SPARK_EXECUTOR_MEMORY', '4g'),
    }


def get_couchdb_config() -> Dict[str, Any]:
    """
    Get CouchDB-specific configuration.

    Returns:
        Dictionary with CouchDB connection settings
    """
    config = get_env_config()
    return {
        'url': config['couchdb_url'],
        'host': config['couchdb_host'],
        'username': config['couchdb_username'],
        'password': config['couchdb_password'],
        'database': config['couchdb_database'],
    }


def get_redis_config() -> Dict[str, Any]:
    """
    Get Redis-specific configuration.

    Returns:
        Dictionary with Redis connection settings
    """
    config = get_env_config()
    return {
        'host': config['redis_host'],
        'port': config['redis_port'],
        'url': config['redis_url'],
    }


def get_spark_config() -> Dict[str, Any]:
    """
    Get Spark-specific configuration.

    Returns:
        Dictionary with Spark settings
    """
    config = get_env_config()
    return {
        'cores': config['cores'],
        'bahir_package': config['bahir_package'],
        'driver_memory': config['spark_driver_memory'],
        'executor_memory': config['spark_executor_memory'],
    }


def build_couchdb_url(host: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> str:
    """
    Build a CouchDB URL from components.

    Args:
        host: CouchDB host (default: from environment)
        username: Username (default: from environment)
        password: Password (default: from environment)

    Returns:
        CouchDB URL string
    """
    config = get_env_config()
    host = host or config['couchdb_host']

    # If host already includes protocol, return as-is
    if host.startswith('http://') or host.startswith('https://'):
        return host

    return f"http://{host}"


def build_redis_url(host: Optional[str] = None, port: Optional[int] = None) -> str:
    """
    Build a Redis URL from components.

    Args:
        host: Redis host (default: from environment)
        port: Redis port (default: from environment)

    Returns:
        Redis URL string
    """
    config = get_env_config()
    host = host or config['redis_host']
    port = port or config['redis_port']
    return f"redis://{host}:{port}"
