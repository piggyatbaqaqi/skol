"""
Centralized environment variable configuration for SKOL scripts.

This module provides a unified way to read configuration from environment
variables with sensible defaults, used by all bin scripts.

Configuration priority (highest to lowest):
1. Command-line arguments
2. Experiment values (from skol_experiments CouchDB, when --experiment is set)
3. Environment variables
4. /home/skol/.skol_env file
5. Default values
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

# Cache for .skol_env file contents
_skol_env_cache: Optional[Dict[str, str]] = None


def _load_skol_env() -> Dict[str, str]:
    """
    Load configuration from /home/skol/.skol_env file.

    This file uses VAR=value syntax (compatible with shell source but not exported).
    We parse it to provide fallback values when environment variables aren't set.

    Returns:
        Dictionary of key-value pairs from the file
    """
    global _skol_env_cache
    if _skol_env_cache is not None:
        return _skol_env_cache

    _skol_env_cache = {}
    skol_env_path = Path('/home/skol/.skol_env')

    if skol_env_path.exists():
        try:
            with open(skol_env_path) as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse VAR=value (handle quoted values)
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip()
                        # Remove surrounding quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        _skol_env_cache[key] = value
        except (IOError, OSError):
            pass  # File not readable, use defaults

    return _skol_env_cache


def _get_env(key: str, default: str = '') -> str:
    """
    Get environment variable with fallback to .skol_env file.

    Args:
        key: Environment variable name
        default: Default value if not found anywhere

    Returns:
        Value from environment, .skol_env file, or default
    """
    # First check environment
    value = os.environ.get(key)
    if value is not None:
        return value

    # Then check .skol_env file
    skol_env = _load_skol_env()
    if key in skol_env:
        return skol_env[key]

    return default


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    """Parse an optional integer from string, returning None if empty or invalid."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_doc_ids(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated list of document IDs, returning None if empty."""
    if not value:
        return None
    return [doc_id.strip() for doc_id in value.split(',') if doc_id.strip()]


def _load_experiment(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """Load an experiment document from the skol_experiments CouchDB database.

    Uses lazy import of couchdb to avoid adding a hard dependency for scripts
    that don't use experiments.

    Args:
        config: Base configuration with CouchDB connection settings.
        experiment_name: The experiment document ID to load.

    Returns:
        The experiment document as a dict.

    Raises:
        SystemExit: If the experiment database or document is not found.
    """
    try:
        import couchdb as couchdb_lib
    except ImportError:
        print(
            "Error: python-couchdb is required for --experiment. "
            "Install it with: pip install couchdb",
            file=sys.stderr,
        )
        sys.exit(1)

    db_name = config.get('experiments_database', 'skol_experiments')
    try:
        server = couchdb_lib.Server(config['couchdb_url'])
        server.resource.credentials = (
            config['couchdb_username'],
            config['couchdb_password'],
        )
        if db_name not in server:
            print(
                f"Error: experiments database '{db_name}' not found.",
                file=sys.stderr,
            )
            sys.exit(1)
        db = server[db_name]
        doc = db.get(experiment_name)
        if doc is None:
            print(
                f"Error: experiment '{experiment_name}' not found in {db_name}.",
                file=sys.stderr,
            )
            sys.exit(1)
        return dict(doc)
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"Error: could not load experiment '{experiment_name}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _apply_experiment(
    config: Dict[str, Any],
    experiment_doc: Dict[str, Any],
    cli_explicit_keys: Set[str],
) -> None:
    """Apply experiment values to config, skipping CLI-explicit keys.

    Mapping from experiment document fields to config keys:
        experiment.databases.ingest    -> ingest_db_name
        experiment.databases.training  -> training_database
        experiment.databases.taxa      -> taxon_db_name, source_db
        experiment.databases.taxa_full -> dest_db
        experiment.redis_keys.classifier_model -> classifier_model_key
        experiment.redis_keys.embedding        -> embedding_name
        experiment.redis_keys.menus            -> menus_key
        experiment.model_name                  -> model_name

    Args:
        config: The config dict to modify in place.
        experiment_doc: The experiment document from CouchDB.
        cli_explicit_keys: Set of config keys explicitly set via CLI args.
    """
    databases = experiment_doc.get('databases', {})
    redis_keys = experiment_doc.get('redis_keys', {})

    db_mapping = [
        ('ingest', ['ingest_db_name']),
        ('training', ['training_database']),
        ('annotations', ['annotations_db_name']),
        ('taxa', ['taxon_db_name', 'source_db']),
        ('taxa_full', ['dest_db']),
    ]
    for exp_field, config_keys in db_mapping:
        value = databases.get(exp_field)
        if value:
            for config_key in config_keys:
                if config_key not in cli_explicit_keys:
                    config[config_key] = value

    redis_mapping = [
        ('classifier_model', 'classifier_model_key'),
        ('embedding', 'embedding_name'),
        ('menus', 'menus_key'),
    ]
    for exp_field, config_key in redis_mapping:
        value = redis_keys.get(exp_field)
        if value:
            if config_key not in cli_explicit_keys:
                config[config_key] = value

    if experiment_doc.get('model_name') and 'model_name' not in cli_explicit_keys:
        config['model_name'] = experiment_doc['model_name']


def get_env_config() -> Dict[str, Any]:
    """
    Get complete environment configuration from command-line args, environment variables, or defaults.

    Command-line arguments take priority over experiment values, which take
    priority over environment variables, which take priority over defaults.
    Arguments follow the pattern: --config-key for config['config_key']
    (underscores in keys become dashes in argument names)

    Returns:
        Dictionary of configuration values for all SKOL components
    """
    # First, get base configuration from environment variables, .skol_env file, or defaults
    base_config = {
        # CouchDB connection settings
        'couchdb_url': _get_env('COUCHDB_URL', 'http://localhost:5984'),
        'couchdb_host': _get_env('COUCHDB_HOST', '127.0.0.1:5984'),
        'couchdb_username': _get_env('COUCHDB_USER', 'admin'),
        'couchdb_password': _get_env('COUCHDB_PASSWORD', 'SU2orange!'),
        'couchdb_database': _get_env('COUCHDB_DATABASE', 'skol_dev'),

        # Ingest database settings (for extract_taxa_to_couchdb.py)
        'ingest_url': _get_env('INGEST_URL', _get_env('COUCHDB_URL', 'http://localhost:5984')),
        'ingest_database': _get_env('INGEST_DATABASE', ''),
        'ingest_username': _get_env('INGEST_USERNAME', _get_env('COUCHDB_USER', '')),
        'ingest_password': _get_env('INGEST_PASSWORD', _get_env('COUCHDB_PASSWORD', '')),
        'ingest_db_name': _get_env('INGEST_DB_NAME', 'skol_dev'),

        # Taxon database settings (for extract_taxa_to_couchdb.py)
        'taxon_url': _get_env('TAXON_URL', ''),
        'taxon_database': _get_env('TAXON_DATABASE', ''),
        'taxon_username': _get_env('TAXON_USERNAME', ''),
        'taxon_password': _get_env('TAXON_PASSWORD', ''),
        'taxon_db_name': _get_env('TAXON_DB_NAME', 'skol_taxa_dev'),

        # JSON translation settings (for taxa_to_json.py)
        'source_db': _get_env('SOURCE_DB', 'skol_taxa_dev'),
        'dest_db': _get_env('DEST_DB', 'skol_taxa_full_dev'),
        'checkpoint_path': _get_env('CHECKPOINT_PATH', ''),

        # Training database settings
        'training_database': _get_env('TRAINING_DATABASE', 'skol_training'),

        # Annotations output database (empty = write to ingest DB)
        'annotations_db_name': _get_env('ANNOTATIONS_DB_NAME', ''),

        # Redis settings
        'redis_host': _get_env('REDIS_HOST', 'localhost'),
        'redis_port': int(_get_env('REDIS_PORT', '6380')),
        'redis_username': _get_env('REDIS_USERNAME', 'admin'),
        'redis_password': _get_env('REDIS_PASSWORD', ''),
        'redis_tls': _get_env('REDIS_TLS', '').lower() in ('1', 'true', 'yes'),
        'redis_url': _get_env('REDIS_URL', ''),  # Built dynamically if not set

        # Model settings
        'model_version': _get_env('MODEL_VERSION', 'v2.0'),
        'classifier_model_key': '',  # Populated by experiment resolution
        'classifier_model_expire': _get_env('MODEL_EXPIRE', ''),

        # Embedding settings
        'embedding_name': _get_env('EMBEDDING_NAME', 'skol:embedding:v1.1'),
        'embedding_expire': int(_get_env('EMBEDDING_EXPIRE', str(60 * 60 * 24 * 2))),  # 2 days default

        # Prediction settings
        'couchdb_pattern': _get_env('COUCHDB_PATTERN', '*.txt'),
        'pattern': _get_env('PATTERN', '*.ann'),  # Matches both .txt.ann and .pdf.ann
        'prediction_batch_size': int(_get_env('PREDICTION_BATCH_SIZE', '24')),
        'num_workers': int(_get_env('NUM_WORKERS', '4')),
        'union_batch_size': int(_get_env('UNION_BATCH_SIZE', '1000')),
        'incremental_batch_size': int(_get_env('INCREMENTAL_BATCH_SIZE', '50')),

        # Taxonomy abbreviations for pre-filtering documents
        'taxonomy_abbrevs': _get_env(
            'TAXONOMY_ABBREVS',
            'comb.,fam.,gen.,nom.,ined.,var.,subg.,subsp.,sp.,f.,syn.,'
            'nov.,spec.,ssp.,spp.,sensu,s.l.,s.s.,s.str.,cf.,aff.,incertae,sed.'
        ).split(','),

        # Data paths
        'annotated_path': Path(_get_env('ANNOTATED_PATH', str(Path.cwd().parent / "data" / "annotated"))),

        # Spark settings
        'cores': int(_get_env('SPARK_CORES', '4')),
        'bahir_package': _get_env('BAHIR_PACKAGE', 'org.apache.bahir:spark-sql-cloudant_2.12:2.4.0'),
        'spark_driver_memory': _get_env('SPARK_DRIVER_MEMORY', '4g'),
        'spark_executor_memory': _get_env('SPARK_EXECUTOR_MEMORY', '4g'),

        # NCBI E-utilities API key (optional, increases rate limit from 3 to 10 rps)
        'ncbi_api_key': _get_env('NCBI_API_KEY', '') or None,

        # Experiment framework
        'experiment_name': _get_env('EXPERIMENT_NAME', ''),
        'experiments_database': _get_env('EXPERIMENTS_DATABASE', 'skol_experiments'),

        # General settings
        'verbosity': int(_get_env('VERBOSITY', '1')),

        # Work-skipping and partial computation options
        'dry_run': _get_env('DRY_RUN', '').lower() in ('1', 'true', 'yes'),
        'skip_existing': _get_env('SKIP_EXISTING', '').lower() in ('1', 'true', 'yes'),
        'force': _get_env('FORCE', '').lower() in ('1', 'true', 'yes'),
        'incremental': _get_env('INCREMENTAL', '').lower() in ('1', 'true', 'yes'),
        'taxonomy_filter': _get_env('TAXONOMY_FILTER', '').lower() in ('1', 'true', 'yes'),
        'skip_golden': _get_env('SKIP_GOLDEN', '').lower() in ('1', 'true', 'yes'),
        'limit': _parse_optional_int(_get_env('LIMIT', '')),
        'doc_ids': _parse_doc_ids(_get_env('DOC_IDS', '')),
    }

    # Parse command-line arguments to override config
    parser = argparse.ArgumentParser(add_help=False)  # Don't add -h/--help to avoid conflicts

    # Add arguments for each configuration key
    # String arguments
    for key in [
        'couchdb_url', 'couchdb_host', 'couchdb_username', 'couchdb_password', 'couchdb_database',
        'ingest_url', 'ingest_database', 'ingest_username', 'ingest_password', 'ingest_db_name',
        'taxon_url', 'taxon_database', 'taxon_username', 'taxon_password', 'taxon_db_name',
        'training_database', 'annotations_db_name',
        'redis_host', 'redis_url', 'redis_username', 'redis_password',
        'model_version', 'classifier_model_expire',
        'model_name', 'menus_key',
        'embedding_name',
        'couchdb_pattern', 'pattern',
        'ncbi_api_key',
        'bahir_package', 'spark_driver_memory', 'spark_executor_memory'
    ]:
        arg_name = '--' + key.replace('_', '-')
        parser.add_argument(arg_name, type=str, default=None, dest=key)

    # Integer arguments
    for key in ['redis_port', 'embedding_expire', 'prediction_batch_size', 'num_workers', 'cores', 'verbosity', 'union_batch_size', 'incremental_batch_size']:
        arg_name = '--' + key.replace('_', '-')
        parser.add_argument(arg_name, type=int, default=None, dest=key)

    # Path arguments
    parser.add_argument('--annotated-path', type=str, default=None, dest='annotated_path')

    # Work-skipping and partial computation options
    parser.add_argument('--dry-run', action='store_true', default=None, dest='dry_run',
                        help='Preview what would be done without making changes')
    parser.add_argument('--skip-existing', action='store_true', default=None, dest='skip_existing',
                        help='Skip records/documents that already have output')
    parser.add_argument('--force', action='store_true', default=None, dest='force',
                        help='Process even if output already exists (overrides --skip-existing)')
    parser.add_argument('--incremental', action='store_true', default=None, dest='incremental',
                        help='Save each record as it completes (crash-resistant)')
    parser.add_argument('--taxonomy-filter', action='store_true', default=None, dest='taxonomy_filter',
                        help='Only process documents with taxonomy abbreviations')
    parser.add_argument('--skip-golden', action='store_true', default=None, dest='skip_golden',
                        help='Skip documents marked as part of the golden dataset')
    parser.add_argument('--limit', type=int, default=None, dest='limit',
                        help='Process at most N records')
    parser.add_argument('--doc-id', '--doc-ids', type=str, default=None, dest='doc_ids',
                        help='Process only specific document ID(s), comma-separated')

    # Experiment flag (resolves databases and Redis keys from experiment record)
    parser.add_argument('--experiment', type=str, default=None, dest='experiment_name',
                        help='Experiment name — resolves databases and Redis keys from skol_experiments')

    # Parse known args (ignore unknown args to avoid breaking scripts with their own arguments)
    args, _ = parser.parse_known_args()

    # Override base config with command-line arguments (if provided)
    # Track which keys were explicitly set via CLI so experiment values don't override them
    cli_explicit_keys: Set[str] = set()
    for key, value in vars(args).items():
        if value is not None:
            cli_explicit_keys.add(key)
            if key == 'annotated_path':
                base_config[key] = Path(value)
            elif key == 'doc_ids':
                # Parse comma-separated doc IDs from command line
                base_config[key] = _parse_doc_ids(value)
            else:
                base_config[key] = value

    # Experiment resolution: load experiment doc and apply values for
    # any config keys that were not explicitly set via CLI
    experiment_name = base_config.get('experiment_name', '')
    if experiment_name:
        experiment_doc = _load_experiment(base_config, experiment_name)
        _apply_experiment(base_config, experiment_doc, cli_explicit_keys)

    return base_config


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
        Dictionary with Redis connection settings including TLS and auth
    """
    config = get_env_config()
    return {
        'host': config['redis_host'],
        'port': config['redis_port'],
        'username': config['redis_username'],
        'password': config['redis_password'],
        'tls': config['redis_tls'],
        'url': config['redis_url'] or build_redis_url(),
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


def build_redis_url(
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    tls: Optional[bool] = None
) -> str:
    """
    Build a Redis URL from components.

    Args:
        host: Redis host (default: from environment)
        port: Redis port (default: from environment)
        username: Redis username (default: from environment)
        password: Redis password (default: from environment)
        tls: Use TLS (rediss://) (default: from environment)

    Returns:
        Redis URL string (redis:// or rediss:// for TLS)
    """
    config = get_env_config()
    host = host or config['redis_host']
    port = port or config['redis_port']
    username = username if username is not None else config['redis_username']
    password = password if password is not None else config['redis_password']
    use_tls = tls if tls is not None else config['redis_tls']

    # Use rediss:// for TLS connections
    scheme = 'rediss' if use_tls else 'redis'

    # Build URL with optional authentication
    if username and password:
        return f"{scheme}://{username}:{password}@{host}:{port}"
    elif password:
        return f"{scheme}://:{password}@{host}:{port}"
    else:
        return f"{scheme}://{host}:{port}"


def create_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    tls: Optional[bool] = None,
    db: int = 0,
    decode_responses: bool = False
):
    """
    Create a Redis client with proper TLS and authentication configuration.

    This is the recommended way to create Redis connections in SKOL.
    It handles TLS certificates and authentication automatically based on
    environment configuration.

    Args:
        host: Redis host (default: from REDIS_HOST env var)
        port: Redis port (default: from REDIS_PORT env var)
        username: Redis username (default: from REDIS_USERNAME env var)
        password: Redis password (default: from REDIS_PASSWORD env var)
        tls: Use TLS (default: from REDIS_TLS env var)
        db: Redis database number (default: 0)
        decode_responses: Whether to decode responses as strings (default: False)

    Returns:
        redis.Redis: Configured Redis client

    Example:
        >>> from env_config import create_redis_client
        >>> r = create_redis_client()
        >>> r.set('key', 'value')
        >>> r.get('key')
    """
    import redis

    config = get_env_config()
    host = host or config['redis_host']
    port = port or config['redis_port']
    username = username if username is not None else config['redis_username']
    password = password if password is not None else config['redis_password']
    use_tls = tls if tls is not None else config['redis_tls']

    # Build connection kwargs
    kwargs: Dict[str, Any] = {
        'host': host,
        'port': port,
        'db': db,
        'decode_responses': decode_responses,
    }

    # Add authentication if configured
    if username:
        kwargs['username'] = username
    if password:
        kwargs['password'] = password

    # Configure TLS if enabled
    if use_tls:
        kwargs['ssl'] = True
        kwargs['ssl_ca_certs'] = '/etc/ssl/certs/ca-certificates.crt'
        # Don't verify hostname (cert is for synoptickeyof.life but we connect to localhost)
        kwargs['ssl_check_hostname'] = False

    return redis.Redis(**kwargs)
