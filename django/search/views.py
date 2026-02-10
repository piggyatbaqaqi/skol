"""
REST API views for SKOL semantic search.
"""
import logging
import math
import traceback

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.http import HttpResponse
import redis
import requests
from requests.auth import HTTPBasicAuth

from .utils import get_redis_client

logger = logging.getLogger(__name__)

# Note: dr_drafts_mycosearch and skol packages are installed via pip
# No sys.path manipulation needed in production

# Note: Experiment is imported lazily inside SearchView.post() to avoid
# loading heavy ML dependencies (TensorFlow, transformers, etc.) at Django startup


class EmbeddingListView(APIView):
    """
    API endpoint to list available embeddings from Redis.

    GET /api/embeddings/
    Returns: List of embedding names matching pattern 'skol:embedding:*'
    """

    def get(self, request):
        logger.info("EmbeddingListView.get() called")
        try:
            # Connect to Redis
            r = get_redis_client(decode_responses=True)

            # Get all keys matching the pattern
            keys = r.keys('skol:embedding:*')
            logger.info(f"Found {len(keys)} embeddings: {keys}")

            # Sort keys
            keys.sort()

            return Response({
                'embeddings': keys,
                'count': len(keys)
            })

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BuildEmbeddingView(APIView):
    """
    API endpoint to trigger building embedding model if it doesn't exist.

    GET /api/embeddings/build/
    Check if the configured embedding exists.

    POST /api/embeddings/build/
    Trigger building the embedding model if it doesn't already exist.

    Request body (optional):
        {
            "force": false,  // Set to true to rebuild even if exists
            "embedding_name": "skol:embedding:v1.1"  // Optional override
        }

    Returns:
        {
            "status": "exists" | "building" | "complete" | "error",
            "embedding_name": "skol:embedding:v1.1",
            "message": "...",
            "embedding_count": 1234  // Number of embeddings (if exists/complete)
        }
    """

    def get(self, request):
        """Check if embedding exists."""
        try:
            # Get embedding name from query param or use default
            embedding_name = request.GET.get(
                'embedding_name',
                getattr(settings, 'EMBEDDING_NAME', 'skol:embedding:v1.1')
            )

            r = get_redis_client(decode_responses=False)

            exists = r.exists(embedding_name)

            if exists:
                # Get embedding count by loading and checking size
                import pickle
                data = r.get(embedding_name)
                if data:
                    df = pickle.loads(data)
                    count = len(df)
                else:
                    count = 0

                return Response({
                    'status': 'exists',
                    'embedding_name': embedding_name,
                    'message': f'Embedding exists with {count} entries',
                    'embedding_count': count
                })
            else:
                return Response({
                    'status': 'not_found',
                    'embedding_name': embedding_name,
                    'message': 'Embedding does not exist. POST to this endpoint to build it.'
                })

        except Exception as e:
            logger.error(f"Error checking embedding status: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request):
        """Trigger embedding build if not exists."""
        logger.info("BuildEmbeddingView.post() called")
        try:
            # Check if user is admin (staff or superuser)
            is_admin = (
                request.user.is_authenticated and
                (request.user.is_staff or request.user.is_superuser)
            )

            # Get parameters - force and embedding_name only allowed for admins
            default_embedding = getattr(settings, 'EMBEDDING_NAME', 'skol:embedding:v1.1')

            if is_admin:
                force = request.data.get('force', False)
                embedding_name = request.data.get('embedding_name', default_embedding)
            else:
                # Non-admins can only trigger builds with defaults
                force = False
                embedding_name = default_embedding
                # Log if they tried to use restricted parameters
                if request.data.get('force') or request.data.get('embedding_name'):
                    logger.info(
                        f"Non-admin user attempted to use restricted parameters "
                        f"(force={request.data.get('force')}, "
                        f"embedding_name={request.data.get('embedding_name')})"
                    )

            r = get_redis_client(decode_responses=False)

            # Check if already exists
            exists = r.exists(embedding_name)

            if exists and not force:
                # Get embedding count
                import pickle
                data = r.get(embedding_name)
                count = len(pickle.loads(data)) if data else 0

                return Response({
                    'status': 'exists',
                    'embedding_name': embedding_name,
                    'message': f'Embedding already exists with {count} entries. '
                               f'Use force=true to rebuild.',
                    'embedding_count': count
                })

            # Trigger the embedding build as a background subprocess
            # Running in background avoids proxy timeout issues
            logger.info(f"Starting embedding build: {embedding_name} (force={force})")

            # Check if a build is already in progress (lock managed by subprocess)
            lock_key = 'skol:build:embedding:lock'

            if r.exists(lock_key):
                logger.info("Embedding build already in progress (lock exists)")
                return Response({
                    'status': 'building',
                    'embedding_name': embedding_name,
                    'message': 'Build in progress. Poll GET to check status.'
                }, status=status.HTTP_409_CONFLICT)

            # Start the subprocess - it will acquire its own lock
            try:
                import subprocess
                import os
                from pathlib import Path

                bin_path = Path(settings.SKOL_BIN_PATH)
                # Use the with_skol symlink which activates conda environment
                embed_script = bin_path / 'embed_taxa'

                if not embed_script.exists():
                    # Fall back to .py if symlink doesn't exist
                    embed_script = bin_path / 'embed_taxa.py'
                    if not embed_script.exists():
                        return Response({
                            'status': 'error',
                            'embedding_name': embedding_name,
                            'message': f'embed_taxa not found at {bin_path}'
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Build command
                cmd = [str(embed_script), '--verbosity=2']
                if force:
                    cmd.append('--force')

                # Set environment variables to pass configuration
                env = os.environ.copy()
                env['COUCHDB_HOST'] = f"{settings.COUCHDB_HOST}:{settings.COUCHDB_PORT}"
                env['COUCHDB_USER'] = settings.COUCHDB_USERNAME
                env['COUCHDB_PASSWORD'] = settings.COUCHDB_PASSWORD
                env['REDIS_HOST'] = settings.REDIS_HOST
                env['REDIS_PORT'] = str(settings.REDIS_PORT)
                env['EMBEDDING_NAME'] = embedding_name
                env['TAXON_DB_NAME'] = getattr(
                    settings, 'TAXON_DB_NAME', 'skol_taxa_dev'
                )

                # Log file for output
                log_file = '/var/log/skol/embed-taxa-api.log'

                logger.info(f"Running command: {' '.join(cmd)}")
                logger.info(f"Output will be logged to: {log_file}")

                # Run in background with output to log file
                # The lock will expire after TTL if process hangs
                with open(log_file, 'a') as log_f:
                    log_f.write(f"\n{'='*70}\n")
                    log_f.write(f"API-triggered build at {__import__('datetime').datetime.now()}\n")
                    log_f.write(f"Command: {' '.join(cmd)}\n")
                    log_f.write(f"{'='*70}\n")
                    log_f.flush()

                    # Start process in background (don't wait)
                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,  # Detach from parent
                    )

                logger.info(f"Started embed_taxa process with PID {process.pid}")

                # Return immediately - don't wait for completion
                # Lock will auto-expire via TTL when build finishes or times out
                return Response({
                    'status': 'building',
                    'embedding_name': embedding_name,
                    'message': f'Build started (PID {process.pid}). Poll GET to check.',
                    'log_file': log_file,
                }, status=status.HTTP_202_ACCEPTED)

            except Exception as e:
                logger.error(f"Failed to start embed_taxa: {e}")
                return Response({
                    'status': 'error',
                    'embedding_name': embedding_name,
                    'message': f'Failed to start embed_taxa: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Error building embedding: {e}")
            tb = traceback.format_exc()
            logger.error(tb)
            return Response({
                'status': 'error',
                'embedding_name': embedding_name if 'embedding_name' in locals() else 'unknown',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BuildVocabTreeView(APIView):
    """
    API endpoint to trigger building vocabulary tree if it doesn't exist.

    GET /api/vocab-tree/build/
    Check if the vocabulary tree exists.

    POST /api/vocab-tree/build/
    Trigger building the vocabulary tree if it doesn't already exist.

    Request body (optional):
        {
            "force": false,  // Set to true to rebuild even if exists
            "db_name": "skol_taxa_full_dev"  // Optional database override
        }

    Returns:
        {
            "status": "exists" | "complete" | "error",
            "redis_key": "skol:ui:menus_2026_01_29_...",
            "message": "...",
            "stats": {...}  // Tree statistics (if exists/complete)
        }
    """

    def get(self, request):
        """Check if vocabulary tree exists."""
        try:
            r = get_redis_client(decode_responses=True)

            # Check for latest pointer
            latest_key = r.get("skol:ui:menus_latest")

            if latest_key and r.exists(latest_key):
                # Get stats from the tree
                import json
                tree_json = r.get(latest_key)
                if tree_json:
                    data = json.loads(tree_json)
                    stats = data.get('stats', {})
                    version = data.get('version', 'unknown')

                    return Response({
                        'status': 'exists',
                        'redis_key': latest_key,
                        'version': version,
                        'message': f'Vocabulary tree exists with {stats.get("total_nodes", 0)} nodes',
                        'stats': stats
                    })

            return Response({
                'status': 'not_found',
                'message': 'Vocabulary tree does not exist. POST to this endpoint to build it.'
            })

        except Exception as e:
            logger.error(f"Error checking vocab tree status: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request):
        """Trigger vocabulary tree build if not exists."""
        try:
            # Check if user is admin (staff or superuser)
            is_admin = (
                request.user.is_authenticated and
                (request.user.is_staff or request.user.is_superuser)
            )

            # Get parameters - force and db_name only allowed for admins
            default_db = getattr(settings, 'VOCAB_TREE_DB', 'skol_taxa_full_dev')

            if is_admin:
                force = request.data.get('force', False)
                db_name = request.data.get('db_name', default_db)
            else:
                # Non-admins can only trigger builds with defaults
                force = False
                db_name = default_db
                # Log if they tried to use restricted parameters
                if request.data.get('force') or request.data.get('db_name'):
                    logger.info(
                        f"Non-admin user attempted to use restricted parameters "
                        f"(force={request.data.get('force')}, "
                        f"db_name={request.data.get('db_name')})"
                    )

            r = get_redis_client(decode_responses=True)

            # Check if already exists
            latest_key = r.get("skol:ui:menus_latest")
            exists = latest_key and r.exists(latest_key)

            if exists and not force:
                import json
                tree_json = r.get(latest_key)
                if tree_json:
                    data = json.loads(tree_json)
                    stats = data.get('stats', {})

                    return Response({
                        'status': 'exists',
                        'redis_key': latest_key,
                        'message': f'Vocabulary tree already exists with '
                                   f'{stats.get("total_nodes", 0)} nodes. '
                                   f'Use force=true to rebuild.',
                        'stats': stats
                    })

            # Trigger the vocab tree build as a background subprocess
            logger.info(f"Starting vocab tree build from {db_name} (force={force})")

            # Check if a build is already in progress (lock managed by subprocess)
            lock_key = 'skol:build:vocab_tree:lock'

            if r.exists(lock_key):
                logger.info("Vocab tree build already in progress (lock exists)")
                return Response({
                    'status': 'building',
                    'message': 'Build in progress. Poll GET to check status.'
                }, status=status.HTTP_409_CONFLICT)

            # Start the subprocess - it will acquire its own lock
            try:
                import subprocess
                import os
                from pathlib import Path

                bin_path = Path(settings.SKOL_BIN_PATH)
                # Use the with_skol symlink which activates conda environment
                build_script = bin_path / 'build_vocab_tree'

                if not build_script.exists():
                    # Fall back to .py if symlink doesn't exist
                    build_script = bin_path / 'build_vocab_tree.py'
                    if not build_script.exists():
                        return Response({
                            'status': 'error',
                            'message': f'build_vocab_tree not found at {bin_path}'
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Build command
                cmd = [str(build_script), '--db', db_name, '--verbosity=2']

                # Set environment variables
                env = os.environ.copy()
                env['COUCHDB_URL'] = settings.COUCHDB_URL
                env['COUCHDB_USER'] = settings.COUCHDB_USERNAME
                env['COUCHDB_PASSWORD'] = settings.COUCHDB_PASSWORD
                env['REDIS_HOST'] = settings.REDIS_HOST
                env['REDIS_PORT'] = str(settings.REDIS_PORT)

                # Log file for output
                log_file = '/var/log/skol/build-vocab-tree-api.log'

                logger.info(f"Running command: {' '.join(cmd)}")
                logger.info(f"Output will be logged to: {log_file}")

                # Run in background with output to log file
                with open(log_file, 'a') as log_f:
                    log_f.write(f"\n{'='*70}\n")
                    log_f.write(f"API-triggered build at {__import__('datetime').datetime.now()}\n")
                    log_f.write(f"Command: {' '.join(cmd)}\n")
                    log_f.write(f"{'='*70}\n")
                    log_f.flush()

                    process = subprocess.Popen(
                        cmd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

                logger.info(f"Started build_vocab_tree process with PID {process.pid}")

                # Return immediately - don't wait for completion
                # Lock will auto-expire via TTL when build finishes or times out
                return Response({
                    'status': 'building',
                    'message': f'Build started (PID {process.pid}). Poll GET to check.',
                    'log_file': log_file,
                }, status=status.HTTP_202_ACCEPTED)

            except Exception as e:
                logger.error(f"Failed to start build_vocab_tree: {e}")
                return Response({
                    'status': 'error',
                    'message': f'Failed to start build_vocab_tree: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.error(f"Error building vocab tree: {e}")
            tb = traceback.format_exc()
            logger.error(tb)
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def clean_float(value):
    """Convert a float value to JSON-safe format, replacing NaN/Inf with None."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def clean_value(value):
    """
    Convert any value to JSON-safe format.

    Handles numpy types, NaN, Inf, and converts them to JSON-serializable types.
    """
    import numpy as np
    import pandas as pd

    # Handle None
    if value is None:
        return None

    # Handle pandas NA
    if pd.isna(value):
        return None

    # Handle numpy types
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Handle strings (check for nan string representation)
    if isinstance(value, str):
        return value

    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}

    # Handle lists recursively
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    # Try to convert to native Python type
    try:
        # Check if it's a numpy scalar
        if hasattr(value, 'item'):
            return clean_value(value.item())
    except (ValueError, TypeError):
        pass

    return value


class SearchView(APIView):
    """
    API endpoint to perform semantic search using SKOL embeddings.

    POST /api/search/
    Request body:
        {
            "prompt": "description text to search for",
            "embedding_name": "skol:embedding:v1.1",
            "k": 3  (optional, default: 3)
        }

    Returns:
        {
            "results": [
                {
                    "Similarity": 0.95,
                    "Title": "Taxon name",
                    "Description": "...",
                    "Feed": "SKOL",
                    ...
                },
                ...
            ],
            "count": 3,
            "prompt": "original prompt"
        }
    """

    def post(self, request):
        # Validate request
        prompt = request.data.get('prompt')
        embedding_name = request.data.get('embedding_name')
        k = request.data.get('k', 3)

        if not prompt:
            return Response(
                {'error': 'prompt is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not embedding_name:
            return Response(
                {'error': 'embedding_name is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            k = int(k)
            if k < 1 or k > 100:
                return Response(
                    {'error': 'k must be between 1 and 100'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            return Response(
                {'error': 'k must be an integer'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Python 3.11+ compatibility: Apply formatargspec shim before importing ML libraries
            import skol_compat  # noqa: F401 (imported for side effects)

            # Lazy import to avoid loading heavy ML dependencies at Django startup
            from dr_drafts_mycosearch.sota_search import Experiment

            # Create experiment instance
            experiment = Experiment(
                prompt=prompt,
                redis_url=settings.REDIS_URL,
                embedding_name=embedding_name,
                k=k
            )

            # Run the search
            experiment.run()

            # Get results
            results = []
            for i in range(min(k, len(experiment.nearest_neighbors))):
                # Get the row index and similarity score
                idx = experiment.nearest_neighbors.index[i]
                similarity = experiment.nearest_neighbors.iloc[i]['similarity']
                row = experiment.embeddings.loc[idx]

                # Build result dictionary from the embedding row data
                # For SKOL_TAXA, all data is already in the embeddings DataFrame
                # Use clean_value() for all fields to ensure JSON serialization
                result_dict = {
                    'Similarity': clean_value(similarity),
                    'Title': clean_value(row.get('taxon', '')),
                    'Description': clean_value(row.get('description', '')),
                    'Feed': clean_value(row.get('source', '')),
                    'URL': clean_value(row.get('filename', '')),
                }

                # Add optional metadata fields if they exist
                if 'source_metadata' in row.index:
                    src_meta = row['source_metadata']
                    if isinstance(src_meta, dict):
                        result_dict['SourceMetadata'] = clean_value(src_meta)
                        # Extract PDF source info for direct PDF access
                        if 'db_name' in src_meta and 'doc_id' in src_meta:
                            result_dict['PDFDbName'] = clean_value(src_meta['db_name'])
                            result_dict['PDFDocId'] = clean_value(src_meta['doc_id'])
                if 'source' in row.index:
                    src = row['source']
                    if isinstance(src, dict):
                        result_dict['Source'] = clean_value(src)
                # Also include ingest field if present (new format)
                if 'ingest' in row.index:
                    ingest = row['ingest']
                    if isinstance(ingest, dict):
                        result_dict['Ingest'] = clean_value(ingest)
                if 'line_number' in row.index:
                    result_dict['LineNumber'] = clean_value(row['line_number'])
                if 'paragraph_number' in row.index:
                    result_dict['ParagraphNumber'] = clean_value(row['paragraph_number'])
                if 'page_number' in row.index:
                    result_dict['PageNumber'] = clean_value(row['page_number'])
                if 'pdf_page' in row.index:
                    result_dict['PDFPage'] = clean_value(row['pdf_page'])
                if 'pdf_label' in row.index:
                    result_dict['PDFLabel'] = clean_value(row['pdf_label'])
                if 'empirical_page_number' in row.index:
                    result_dict['EmpiricalPageNumber'] = clean_value(row['empirical_page_number'])
                if 'taxon_id' in row.index:
                    result_dict['taxon_id'] = clean_value(row['taxon_id'])

                    # Detect collection results vs taxa results
                    taxon_id = row['taxon_id']
                    if isinstance(taxon_id, str) and taxon_id.startswith('collection_'):
                        result_dict['ResultType'] = 'collection'
                        # Extract collection_id from taxon_id
                        try:
                            collection_id_str = taxon_id.replace('collection_', '')
                            result_dict['CollectionId'] = int(collection_id_str)
                        except (ValueError, AttributeError):
                            result_dict['CollectionId'] = None

                        # Add owner info if present
                        if 'owner' in row.index:
                            owner = row['owner']
                            if isinstance(owner, dict):
                                result_dict['Owner'] = clean_value(owner)
                    else:
                        result_dict['ResultType'] = 'taxon'

                results.append(result_dict)

            return Response({
                'results': results,
                'count': len(results),
                'prompt': prompt,
                'embedding_name': embedding_name,
                'k': k
            })

        except ValueError as e:
            tb = traceback.format_exc()
            logger.error(f"Embedding error for prompt='{prompt[:50]}...', embedding={embedding_name}: {e}\n{tb}")
            return Response(
                {'error': f'Embedding error: {str(e)}'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Search failed for prompt='{prompt[:50]}...', embedding={embedding_name}: {e}\n{tb}")
            return Response(
                {'error': f'Search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TaxaInfoView(APIView):
    """
    API endpoint to get taxa document information including source PDF details.

    GET /api/taxa/<taxa_id>/
    Returns: Taxa document with source information for PDF retrieval

    Response format matches search result format for use with TaxonResultWidget:
    {
        "taxon_id": "...",
        "Title": "Taxon name",
        "Description": "...",
        "PDFDbName": "...",
        "PDFDocId": "...",
        "PDFPage": 42,
        "LineNumber": 100,
        ...
    }
    """

    def get(self, request, taxa_id, taxa_db='skol_taxa_dev'):
        try:
            # Build CouchDB URL for the taxa document
            couchdb_url = settings.COUCHDB_URL
            auth = HTTPBasicAuth(settings.COUCHDB_USERNAME, settings.COUCHDB_PASSWORD)

            # Fetch the taxa document
            taxa_url = f"{couchdb_url}/{taxa_db}/{taxa_id}"
            response = requests.get(taxa_url, auth=auth, timeout=30)

            # If not found and taxa_id doesn't have prefix, try with taxon_ prefix
            # This handles legacy embeddings that don't include the prefix
            if response.status_code == 404 and not taxa_id.startswith('taxon_'):
                taxa_url_with_prefix = f"{couchdb_url}/{taxa_db}/taxon_{taxa_id}"
                response = requests.get(taxa_url_with_prefix, auth=auth, timeout=30)
                if response.status_code == 200:
                    # Update taxa_id to include the prefix for consistency
                    taxa_id = f"taxon_{taxa_id}"

            if response.status_code == 404:
                return Response(
                    {'error': f'Taxa document not found: {taxa_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            response.raise_for_status()
            taxa_doc = response.json()

            # Extract ingest metadata for PDF linking
            ingest = taxa_doc.get('ingest', {})
            pdf_db_name = None
            pdf_doc_id = None
            url = None

            if isinstance(ingest, dict):
                pdf_db_name = ingest.get('db_name')
                pdf_doc_id = ingest.get('_id')
                url = ingest.get('url', '')

            # Default PDFDbName to 'skol_dev' when not available in ingest
            if not pdf_db_name:
                pdf_db_name = 'skol_dev'

            # Return taxa info in search result format for widget compatibility
            result = {
                'taxon_id': taxa_id,
                'taxa_db': taxa_db,
                # Search result compatible fields
                'Title': taxa_doc.get('taxon', ''),
                'Description': taxa_doc.get('description', ''),
                'Feed': 'CouchDB Taxa',
                'URL': url,
                'Source': ingest,
                # PDF linking fields
                'PDFDbName': pdf_db_name,
                'PDFDocId': pdf_doc_id,
                'PDFPage': clean_value(taxa_doc.get('pdf_page')),
                'PDFLabel': clean_value(taxa_doc.get('pdf_label')),
                # Position metadata
                'LineNumber': clean_value(taxa_doc.get('line_number')),
                'ParagraphNumber': clean_value(taxa_doc.get('paragraph_number')),
                'PageNumber': clean_value(taxa_doc.get('page_number')),
                'EmpiricalPageNumber': clean_value(taxa_doc.get('empirical_page_number')),
            }

            return Response(result)

        except requests.exceptions.RequestException as e:
            logger.error(f"CouchDB request failed for taxa {taxa_id}: {e}")
            return Response(
                {'error': f'Failed to fetch taxa document: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PDFAttachmentView(APIView):
    """
    API endpoint to retrieve PDF attachments from CouchDB documents.

    GET /api/pdf/<db_name>/<doc_id>/
    GET /api/pdf/<db_name>/<doc_id>/<attachment_name>/

    Returns: PDF file content (application/pdf)

    This endpoint fetches the PDF attachment from the specified CouchDB document
    and returns it as a downloadable/viewable PDF file.
    """

    def get(self, request, db_name, doc_id, attachment_name='article.pdf'):
        try:
            # Build CouchDB URL for the attachment
            couchdb_url = settings.COUCHDB_URL
            auth = HTTPBasicAuth(settings.COUCHDB_USERNAME, settings.COUCHDB_PASSWORD)

            # Fetch the attachment
            attachment_url = f"{couchdb_url}/{db_name}/{doc_id}/{attachment_name}"
            response = requests.get(attachment_url, auth=auth, timeout=60, stream=True)

            if response.status_code == 404:
                return Response(
                    {'error': f'PDF attachment not found: {db_name}/{doc_id}/{attachment_name}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            response.raise_for_status()

            # Return the PDF content
            http_response = HttpResponse(
                response.content,
                content_type=response.headers.get('Content-Type', 'application/pdf')
            )

            # Set Content-Disposition for inline viewing or download
            disposition = request.GET.get('download', 'inline')
            if disposition == 'true':
                http_response['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            else:
                http_response['Content-Disposition'] = f'inline; filename="{attachment_name}"'

            return http_response

        except requests.exceptions.RequestException as e:
            logger.error(f"CouchDB request failed for {db_name}/{doc_id}/{attachment_name}: {e}")
            return Response(
                {'error': f'Failed to fetch PDF: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PDFFromTaxaView(APIView):
    """
    API endpoint to retrieve PDF from a taxa document's source reference.

    GET /api/taxa/<taxa_id>/pdf/
    GET /api/taxa/<taxa_id>/pdf/?taxa_db=<taxa_db_name>

    This is a convenience endpoint that:
    1. Looks up the taxa document
    2. Extracts the source db_name and doc_id
    3. Returns the PDF attachment from the source document
    """

    def get(self, request, taxa_id):
        try:
            taxa_db = request.GET.get('taxa_db', 'skol_taxa_dev')

            # Build CouchDB URL
            couchdb_url = settings.COUCHDB_URL
            auth = HTTPBasicAuth(settings.COUCHDB_USERNAME, settings.COUCHDB_PASSWORD)

            # Fetch the taxa document to get source info
            taxa_url = f"{couchdb_url}/{taxa_db}/{taxa_id}"
            taxa_response = requests.get(taxa_url, auth=auth, timeout=30)

            if taxa_response.status_code == 404:
                return Response(
                    {'error': f'Taxa document not found: {taxa_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            taxa_response.raise_for_status()
            taxa_doc = taxa_response.json()

            # Get ingest information
            ingest = taxa_doc.get('ingest', {})
            ingest_db = ingest.get('db_name')
            ingest_doc_id = ingest.get('_id')

            if not ingest_db or not ingest_doc_id:
                return Response(
                    {'error': 'Taxa document does not have ingest information'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Fetch the PDF attachment from the ingest document
            attachment_name = 'article.pdf'
            attachment_url = f"{couchdb_url}/{ingest_db}/{ingest_doc_id}/{attachment_name}"
            pdf_response = requests.get(attachment_url, auth=auth, timeout=60, stream=True)

            if pdf_response.status_code == 404:
                return Response(
                    {'error': f'PDF attachment not found in ingest document: {ingest_db}/{ingest_doc_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            pdf_response.raise_for_status()

            # Return the PDF content
            http_response = HttpResponse(
                pdf_response.content,
                content_type=pdf_response.headers.get('Content-Type', 'application/pdf')
            )

            # Include page number in filename if available
            pdf_page = taxa_doc.get('pdf_page')
            if pdf_page:
                filename = f"{taxa_id}_page{pdf_page}.pdf"
            else:
                filename = f"{taxa_id}.pdf"

            disposition = request.GET.get('download', 'inline')
            if disposition == 'true':
                http_response['Content-Disposition'] = f'attachment; filename="{filename}"'
            else:
                http_response['Content-Disposition'] = f'inline; filename="{filename}"'

            # Add custom headers for client-side page navigation
            if pdf_page:
                http_response['X-PDF-Page'] = str(pdf_page)

            return http_response

        except requests.exceptions.RequestException as e:
            logger.error(f"CouchDB request failed for taxa {taxa_id}: {e}")
            return Response(
                {'error': f'Failed to fetch PDF: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================================
# Collection Views
# ============================================================================

from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Collection, SearchHistory, ExternalIdentifier, IdentifierType
from .serializers import (
    CollectionListSerializer,
    CollectionDetailSerializer,
    CollectionCreateSerializer,
    CollectionUpdateSerializer,
    SearchHistorySerializer,
    ExternalIdentifierSerializer,
    ExternalIdentifierCreateSerializer,
    IdentifierTypeSerializer,
)


class IdentifierTypeListView(APIView):
    """
    GET /api/identifier-types/
    List all available identifier types.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        identifier_types = IdentifierType.objects.all()
        serializer = IdentifierTypeSerializer(identifier_types, many=True)
        return Response({
            'identifier_types': serializer.data,
            'count': len(serializer.data)
        })


class FungariaListView(APIView):
    """
    GET /api/fungaria/
    List all fungaria/herbaria from Redis (Index Herbariorum registry).

    Returns simplified list for dropdown selection with code, organization,
    and URL information for building links.

    Query parameters:
        - search: Filter by code or organization name (case-insensitive)
        - limit: Maximum number of results (default: 0 = no limit)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            r = get_redis_client(decode_responses=True)

            raw = r.get('skol:fungaria')
            if not raw:
                return Response({
                    'fungaria': [],
                    'count': 0,
                    'error': 'Fungaria registry not loaded. Run manage_fungaria download.'
                })

            import json
            data = json.loads(raw)
            institutions = data.get('institutions', {})

            fungaria = []
            search_query = request.GET.get('search', '').lower()
            limit = int(request.GET.get('limit', 0))

            for code, inst in institutions.items():
                # Apply search filter
                org = inst.get('organization', '')
                if search_query:
                    if search_query not in code.lower() and search_query not in org.lower():
                        continue

                # Get fungi count if available
                collections_summary = inst.get('collectionsSummary', {})
                num_fungi = 0
                if isinstance(collections_summary, dict):
                    num_fungi = collections_summary.get('numFungi', 0) or 0

                # Build URL info
                contact = inst.get('contact', {})
                collection_url = ''
                web_url = ''
                if isinstance(contact, dict):
                    collection_url = contact.get('collectionUrl', '')
                    web_url = contact.get('webUrl', '')

                # Get location
                addr = inst.get('address', {})
                location = ''
                if isinstance(addr, dict):
                    city = addr.get('physicalCity') or addr.get('city', '')
                    state = addr.get('physicalState') or addr.get('state', '')
                    country = addr.get('physicalCountry') or addr.get('country', '')
                    location = ', '.join(filter(None, [city, state, country]))

                fungaria.append({
                    'code': code,
                    'organization': org,
                    'num_fungi': num_fungi,
                    'location': location,
                    'collection_url': collection_url,  # f-string with {id}
                    'web_url': web_url,  # fallback URL
                })

            # Sort by code
            fungaria.sort(key=lambda x: x['code'].upper())

            # Apply limit if specified
            if limit > 0 and len(fungaria) > limit:
                fungaria = fungaria[:limit]

            return Response({
                'fungaria': fungaria,
                'count': len(fungaria),
                'total_in_registry': len(institutions)
            })

        except Exception as e:
            logger.error(f"Failed to fetch fungaria: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CollectionListCreateView(APIView):
    """
    GET /api/collections/
    List all collections for the logged-in user.

    POST /api/collections/
    Create a new collection for the logged-in user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        collections = Collection.objects.filter(owner=request.user)
        serializer = CollectionListSerializer(collections, many=True)
        return Response({
            'collections': serializer.data,
            'count': len(serializer.data)
        })

    def post(self, request):
        serializer = CollectionCreateSerializer(data=request.data)
        if serializer.is_valid():
            collection = serializer.save(owner=request.user)
            return Response(
                CollectionDetailSerializer(collection).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CollectionDetailView(APIView):
    """
    GET /api/collections/<collection_id>/
    Retrieve a collection (any authenticated user can view).

    PUT /api/collections/<collection_id>/
    Update collection name/description (owner only).

    DELETE /api/collections/<collection_id>/
    Delete a collection (owner only).
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, collection_id):
        return get_object_or_404(Collection, collection_id=collection_id)

    def get(self, request, collection_id):
        collection = self.get_object(collection_id)
        serializer = CollectionDetailSerializer(collection)
        return Response(serializer.data)

    def put(self, request, collection_id):
        collection = self.get_object(collection_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to edit this collection'},
                status=status.HTTP_403_FORBIDDEN
            )
        serializer = CollectionUpdateSerializer(collection, data=request.data, partial=True)
        if serializer.is_valid():
            updated_collection = serializer.save()
            return Response(CollectionDetailSerializer(updated_collection).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, collection_id):
        collection = self.get_object(collection_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to delete this collection'},
                status=status.HTTP_403_FORBIDDEN
            )
        collection.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class CollectionByUserView(APIView):
    """
    GET /api/collections/user/<username>/
    List all collections for a specific user (public viewing).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, username):
        collections = Collection.objects.filter(owner__username=username)
        serializer = CollectionListSerializer(collections, many=True)
        return Response({
            'collections': serializer.data,
            'count': len(serializer.data),
            'username': username
        })


class SearchHistoryListCreateView(APIView):
    """
    GET /api/collections/<collection_id>/searches/
    List search history for a collection.

    POST /api/collections/<collection_id>/searches/
    Add a search to the collection's history (owner only).
    """
    permission_classes = [IsAuthenticated]

    def get_collection(self, collection_id):
        return get_object_or_404(Collection, collection_id=collection_id)

    def get(self, request, collection_id):
        collection = self.get_collection(collection_id)
        searches = collection.search_history.all()
        serializer = SearchHistorySerializer(searches, many=True)
        return Response({
            'searches': serializer.data,
            'count': len(serializer.data),
            'collection_id': collection_id
        })

    def post(self, request, collection_id):
        collection = self.get_collection(collection_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to add searches to this collection'},
                status=status.HTTP_403_FORBIDDEN
            )
        serializer = SearchHistorySerializer(data=request.data)
        if serializer.is_valid():
            search = serializer.save(
                collection=collection,
                result_count=len(request.data.get('result_references', []))
            )
            return Response(
                SearchHistorySerializer(search).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SearchHistoryDetailView(APIView):
    """
    GET /api/collections/<collection_id>/searches/<search_id>/
    Retrieve a specific search history entry.

    DELETE /api/collections/<collection_id>/searches/<search_id>/
    Delete a search history entry (owner only).
    """
    permission_classes = [IsAuthenticated]

    def get_objects(self, collection_id, search_id):
        collection = get_object_or_404(Collection, collection_id=collection_id)
        search = get_object_or_404(SearchHistory, id=search_id, collection=collection)
        return collection, search

    def get(self, request, collection_id, search_id):
        collection, search = self.get_objects(collection_id, search_id)
        return Response(SearchHistorySerializer(search).data)

    def delete(self, request, collection_id, search_id):
        collection, search = self.get_objects(collection_id, search_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to delete this search'},
                status=status.HTTP_403_FORBIDDEN
            )
        search.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ExternalIdentifierListCreateView(APIView):
    """
    GET /api/collections/<collection_id>/identifiers/
    List external identifiers for a collection.

    POST /api/collections/<collection_id>/identifiers/
    Add an external identifier to the collection (owner only).
    """
    permission_classes = [IsAuthenticated]

    def get_collection(self, collection_id):
        return get_object_or_404(Collection, collection_id=collection_id)

    def get(self, request, collection_id):
        collection = self.get_collection(collection_id)
        identifiers = collection.external_identifiers.all()
        serializer = ExternalIdentifierSerializer(identifiers, many=True)
        return Response({
            'identifiers': serializer.data,
            'count': len(serializer.data),
            'collection_id': collection_id
        })

    def post(self, request, collection_id):
        collection = self.get_collection(collection_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to add identifiers to this collection'},
                status=status.HTTP_403_FORBIDDEN
            )
        serializer = ExternalIdentifierCreateSerializer(data=request.data)
        if serializer.is_valid():
            identifier = serializer.save(collection=collection)
            return Response(
                ExternalIdentifierSerializer(identifier).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ExternalIdentifierDetailView(APIView):
    """
    GET /api/collections/<collection_id>/identifiers/<identifier_id>/
    Retrieve a specific external identifier.

    DELETE /api/collections/<collection_id>/identifiers/<identifier_id>/
    Delete an external identifier (owner only).
    """
    permission_classes = [IsAuthenticated]

    def get_objects(self, collection_id, identifier_id):
        collection = get_object_or_404(Collection, collection_id=collection_id)
        identifier = get_object_or_404(
            ExternalIdentifier, id=identifier_id, collection=collection
        )
        return collection, identifier

    def get(self, request, collection_id, identifier_id):
        collection, identifier = self.get_objects(collection_id, identifier_id)
        return Response(ExternalIdentifierSerializer(identifier).data)

    def delete(self, request, collection_id, identifier_id):
        collection, identifier = self.get_objects(collection_id, identifier_id)
        if collection.owner != request.user:
            return Response(
                {'error': 'You do not have permission to delete this identifier'},
                status=status.HTTP_403_FORBIDDEN
            )
        identifier.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class GuestCollectionImportView(APIView):
    """
    POST /api/collections/import-guest/
    Import a guest collection from localStorage into a real collection.

    This endpoint is called after login/registration to migrate any
    collection data that was stored in the browser while the user
    was not logged in.

    Request body:
    {
        "name": "Collection Name",
        "description": "Working description",
        "notes": "Optional notes",
        "search_history": [
            {
                "prompt": "search text",
                "embedding_name": "embed_name",
                "k": 3,
                "result_references": [...],
                "created_at": "2024-01-01T00:00:00Z"
            }
        ],
        "external_identifiers": [
            {
                "identifier_type_code": "inat",
                "value": "12345",
                "fungarium_code": "",
                "notes": ""
            }
        ]
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data

        # Validate basic structure
        if not isinstance(data, dict):
            return Response(
                {'error': 'Invalid data format'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Create the collection
        collection = Collection.objects.create(
            owner=request.user,
            name=data.get('name', 'Guest Collection'),
            description=data.get('description', ''),
            notes=data.get('notes', '')
        )

        # Import search history
        search_history = data.get('search_history', [])
        imported_searches = 0
        for search in search_history:
            if not isinstance(search, dict):
                continue
            try:
                SearchHistory.objects.create(
                    collection=collection,
                    prompt=search.get('prompt', ''),
                    embedding_name=search.get('embedding_name', 'unknown'),
                    k=search.get('k', 3),
                    result_references=search.get('result_references', []),
                    result_count=len(search.get('result_references', []))
                )
                imported_searches += 1
            except Exception as e:
                logger.warning(f"Failed to import search history entry: {e}")

        # Import external identifiers
        external_ids = data.get('external_identifiers', [])
        imported_identifiers = 0
        for identifier in external_ids:
            if not isinstance(identifier, dict):
                continue
            try:
                identifier_type_code = identifier.get('identifier_type_code', '')
                if not identifier_type_code:
                    continue

                # Look up identifier type
                try:
                    identifier_type = IdentifierType.objects.get(code=identifier_type_code)
                except IdentifierType.DoesNotExist:
                    logger.warning(f"Unknown identifier type: {identifier_type_code}")
                    continue

                ExternalIdentifier.objects.create(
                    collection=collection,
                    identifier_type=identifier_type,
                    value=identifier.get('value', ''),
                    fungarium_code=identifier.get('fungarium_code', ''),
                    notes=identifier.get('notes', '')
                )
                imported_identifiers += 1
            except Exception as e:
                logger.warning(f"Failed to import external identifier: {e}")

        return Response({
            'success': True,
            'collection_id': collection.collection_id,
            'name': collection.name,
            'imported_searches': imported_searches,
            'imported_identifiers': imported_identifiers
        }, status=status.HTTP_201_CREATED)


# ============================================================================
# Vocabulary Tree Views
# ============================================================================

import json


class VocabTreeView(APIView):
    """
    API endpoint to retrieve vocabulary trees from Redis.

    GET /api/vocab-tree/
    Returns the latest vocabulary tree.

    Query parameters:
        - version: Specific version string (e.g., "2026_01_26_10_30")
        - path: Dot-separated path to get children at a specific location
                (e.g., "pileus.shape" returns children under pileus > shape)
        - depth: Maximum depth to return (default: unlimited)

    Response format:
        {
            "version": "2026_01_26_10_30",
            "created_at": "2026-01-26T10:30:00",
            "tree": { ... },  // Full tree or subtree based on path
            "stats": { ... }  // Tree statistics
        }
    """

    def get(self, request):
        try:
            r = get_redis_client(decode_responses=True)

            version = request.GET.get('version')
            path = request.GET.get('path', '')
            max_depth = request.GET.get('depth')

            # Determine which key to fetch
            if version:
                key = f"skol:ui:menus_{version}"
            else:
                # Get the latest version
                latest_key = r.get("skol:ui:menus_latest")
                if not latest_key:
                    return Response(
                        {'error': 'No vocabulary tree available'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                key = latest_key

            # Fetch the tree data
            tree_json = r.get(key)
            if not tree_json:
                return Response(
                    {'error': f'Vocabulary tree not found: {key}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            data = json.loads(tree_json)
            tree = data.get('tree', {})
            stats = data.get('stats', {})

            # Navigate to specific path if requested
            if path:
                path_parts = path.split('.')
                current = tree
                for part in path_parts:
                    part_lower = part.lower()
                    if isinstance(current, dict) and part_lower in current:
                        current = current[part_lower]
                    else:
                        return Response(
                            {'error': f'Path not found: {path}'},
                            status=status.HTTP_404_NOT_FOUND
                        )
                tree = current

            # Limit depth if requested
            if max_depth:
                try:
                    max_depth = int(max_depth)
                    tree = self._limit_depth(tree, max_depth)
                except ValueError:
                    pass

            return Response({
                'version': data.get('version'),
                'created_at': data.get('created_at'),
                'path': path or None,
                'tree': tree,
                'stats': stats
            })

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vocab tree JSON: {e}")
            return Response(
                {'error': 'Invalid vocabulary tree data'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Failed to fetch vocab tree: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _limit_depth(self, tree, max_depth, current_depth=0):
        """Recursively limit tree depth."""
        if current_depth >= max_depth:
            if isinstance(tree, dict) and tree:
                return {"...": f"{len(tree)} children"}
            return tree

        if isinstance(tree, dict):
            return {
                k: self._limit_depth(v, max_depth, current_depth + 1)
                for k, v in tree.items()
            }
        return tree


class VocabTreeVersionsView(APIView):
    """
    API endpoint to list available vocabulary tree versions.

    GET /api/vocab-tree/versions/
    Returns list of available versions with their metadata.
    """

    def get(self, request):
        try:
            r = get_redis_client(decode_responses=True)

            # Find all vocab tree keys
            keys = r.keys('skol:ui:menus_*')

            # Filter out the "latest" pointer
            version_keys = [k for k in keys if k != 'skol:ui:menus_latest']

            # Get metadata for each version
            versions = []
            for key in sorted(version_keys, reverse=True):  # Newest first
                try:
                    tree_json = r.get(key)
                    if tree_json:
                        data = json.loads(tree_json)
                        versions.append({
                            'key': key,
                            'version': data.get('version'),
                            'created_at': data.get('created_at'),
                            'stats': data.get('stats', {})
                        })
                except (json.JSONDecodeError, TypeError):
                    # Skip malformed entries
                    versions.append({
                        'key': key,
                        'version': key.replace('skol:ui:menus_', ''),
                        'created_at': None,
                        'stats': {}
                    })

            # Get the current "latest" pointer
            latest = r.get('skol:ui:menus_latest')

            return Response({
                'versions': versions,
                'count': len(versions),
                'latest': latest
            })

        except Exception as e:
            logger.error(f"Failed to list vocab tree versions: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class VocabTreeChildrenView(APIView):
    """
    API endpoint to get children at a specific path in the vocabulary tree.

    GET /api/vocab-tree/children/
    Returns list of child keys at the specified path.

    Query parameters:
        - path: Dot-separated path (e.g., "pileus.shape")
                If omitted, returns top-level keys.
        - version: Specific version (default: latest)

    Response format:
        {
            "path": "pileus.shape",
            "children": ["convex", "flat", "umbonate", ...],
            "count": 15,
            "has_grandchildren": {"convex": true, "flat": false, ...}
        }
    """

    def get(self, request):
        try:
            r = get_redis_client(decode_responses=True)

            version = request.GET.get('version')
            path = request.GET.get('path', '')

            # Determine which key to fetch
            if version:
                key = f"skol:ui:menus_{version}"
            else:
                latest_key = r.get("skol:ui:menus_latest")
                if not latest_key:
                    return Response(
                        {'error': 'No vocabulary tree available'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                key = latest_key

            # Fetch the tree
            tree_json = r.get(key)
            if not tree_json:
                return Response(
                    {'error': f'Vocabulary tree not found: {key}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            data = json.loads(tree_json)
            tree = data.get('tree', {})

            # Navigate to path
            current = tree
            if path:
                path_parts = path.split('.')
                for part in path_parts:
                    part_lower = part.lower()
                    if isinstance(current, dict) and part_lower in current:
                        current = current[part_lower]
                    else:
                        return Response(
                            {'error': f'Path not found: {path}'},
                            status=status.HTTP_404_NOT_FOUND
                        )

            # Get children
            if not isinstance(current, dict):
                return Response({
                    'path': path or None,
                    'children': [],
                    'count': 0,
                    'has_grandchildren': {},
                    'is_leaf': True
                })

            children = sorted(current.keys())
            has_grandchildren = {
                child: isinstance(current[child], dict) and bool(current[child])
                for child in children
            }

            return Response({
                'path': path or None,
                'children': children,
                'count': len(children),
                'has_grandchildren': has_grandchildren,
                'is_leaf': False
            })

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vocab tree JSON: {e}")
            return Response(
                {'error': 'Invalid vocabulary tree data'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Failed to get vocab tree children: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TextClassifierView(APIView):
    """REST interface for TaxaDecisionTreeClassifier (description text features).

    POST /api/classifier/text/
    Body: { taxa_ids: [...], top_n: 30, max_depth: 10, min_df: 1, max_df: 1.0 }
    Response: {
        features: [{name, importance, display_text}, ...],
        metadata: {n_classes, n_features, tree_depth, taxa_count},
        tree_json: {...}
    }
    """

    def post(self, request):
        taxa_ids = request.data.get('taxa_ids', [])
        top_n = request.data.get('top_n', 30)
        max_depth = request.data.get('max_depth', 10)
        min_df = request.data.get('min_df', 1)
        max_df = request.data.get('max_df', 1.0)

        if not taxa_ids or not isinstance(taxa_ids, list):
            return Response(
                {'error': 'taxa_ids must be a non-empty list'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Lazy import to avoid loading ML dependencies at startup
            import sys
            skol_root = settings.SKOL_ROOT_PATH
            if skol_root not in sys.path:
                sys.path.insert(0, skol_root)
            from taxa_classifier.taxa_decision_tree import TaxaDecisionTreeClassifier

            classifier = TaxaDecisionTreeClassifier(
                couchdb_url=settings.COUCHDB_URL,
                database='skol_taxa_dev',
                username=settings.COUCHDB_USERNAME,
                password=settings.COUCHDB_PASSWORD,
                max_depth=max_depth,
                min_df=min_df,
                max_df=max_df,
                verbosity=0,
            )

            stats = classifier.fit(taxa_ids=taxa_ids, test_size=0.0)
            importances = classifier.get_feature_importances(top_n=top_n)
            tree_json = classifier.tree_to_json(max_depth=max_depth)

            features = [
                {
                    'name': name,
                    'importance': float(importance),
                    'display_text': name,
                }
                for name, importance in importances
            ]

            return Response({
                'features': features,
                'metadata': {
                    'n_classes': stats['n_classes'],
                    'n_features': stats['n_features'],
                    'tree_depth': stats['tree_depth'],
                    'taxa_count': len(taxa_ids),
                },
                'tree_json': tree_json,
            })

        except ValueError as e:
            logger.warning(f"Text classifier validation error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Text classifier error: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class JsonClassifierView(APIView):
    """REST interface for TaxaJsonClassifier (structured JSON annotation features).

    POST /api/classifier/json/
    Body: { taxa_ids: [...], top_n: 30, max_depth: 10, min_df: 1, max_df: 1.0 }
    Response: {
        features: [{name, importance, display_text}, ...],
        metadata: {n_classes, n_features, tree_depth, taxa_count},
        tree_json: {...}
    }
    """

    def post(self, request):
        taxa_ids = request.data.get('taxa_ids', [])
        top_n = request.data.get('top_n', 30)
        max_depth = request.data.get('max_depth', 10)
        min_df = request.data.get('min_df', 1)
        max_df = request.data.get('max_df', 1.0)

        if not taxa_ids or not isinstance(taxa_ids, list):
            return Response(
                {'error': 'taxa_ids must be a non-empty list'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Lazy import to avoid loading ML dependencies at startup
            import sys
            skol_root = settings.SKOL_ROOT_PATH
            if skol_root not in sys.path:
                sys.path.insert(0, skol_root)
            from taxa_classifier.taxa_json_classifier import TaxaJsonClassifier

            classifier = TaxaJsonClassifier(
                couchdb_url=settings.COUCHDB_URL,
                database='skol_taxa_full_dev',
                username=settings.COUCHDB_USERNAME,
                password=settings.COUCHDB_PASSWORD,
                max_depth=max_depth,
                min_df=min_df,
                max_df=max_df,
                verbosity=0,
            )

            stats = classifier.fit(taxa_ids=taxa_ids, test_size=0.0)
            importances = classifier.get_feature_importances(top_n=top_n)
            tree_json = classifier.tree_to_json(max_depth=max_depth)

            features = [
                {
                    'name': name,
                    'importance': float(importance),
                    # Convert key=value to "key value" with _ replaced by spaces
                    'display_text': name.replace('=', ' ').replace('_', ' '),
                }
                for name, importance in importances
            ]

            return Response({
                'features': features,
                'metadata': {
                    'n_classes': stats['n_classes'],
                    'n_features': stats['n_features'],
                    'tree_depth': stats['tree_depth'],
                    'taxa_count': len(taxa_ids),
                },
                'tree_json': tree_json,
            })

        except ValueError as e:
            logger.warning(f"JSON classifier validation error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"JSON classifier error: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


def parse_span(span):
    """
    Parse a span that may be stored as either a dict or an array.

    Array format (from Spark StructType serialization):
      [paragraph_number, start_line, end_line, start_char, end_char,
       pdf_page, pdf_label, empirical_page]

    Returns a dict with named fields.
    """
    if isinstance(span, dict):
        return span
    elif isinstance(span, (list, tuple)) and len(span) >= 5:
        return {
            'paragraph_number': span[0],
            'start_line': span[1],
            'end_line': span[2],
            'start_char': span[3],
            'end_char': span[4],
            'pdf_page': span[5] if len(span) > 5 else None,
            'pdf_label': span[6] if len(span) > 6 else None,
            'empirical_page': span[7] if len(span) > 7 else None,
        }
    else:
        # Unknown format, return empty dict
        return {}


class SourceContextView(APIView):
    """
    Retrieve windowed source text with highlight markers for the Source Context Viewer.

    Shows all spans for a field in a single window, from the start of the first span
    to the end of the last span, with all spans highlighted.

    GET /api/taxa/<taxa_id>/context/
    Query params:
      - field: 'nomenclature' or 'description' (default: 'description')
      - context_chars: characters of context before/after span range (default: 500)
      - taxa_db: database name (default: 'skol_taxa_dev')

    Response:
      {
        "source_text": "...text with <mark>highlighted</mark> regions...",
        "total_spans": 2,
        "pdf_page": 35,
        "pdf_label": "35",
        "empirical_page": "127"
      }
    """

    def get(self, request, taxa_id):
        try:
            # Parse query parameters
            field = request.GET.get('field', 'description')
            context_chars = int(request.GET.get('context_chars', 500))
            taxa_db = request.GET.get('taxa_db', 'skol_taxa_dev')

            if field not in ('nomenclature', 'description'):
                return Response(
                    {'error': f'Invalid field: {field}. Must be "nomenclature" or "description"'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Build CouchDB URL
            couchdb_url = settings.COUCHDB_URL
            auth = HTTPBasicAuth(settings.COUCHDB_USERNAME, settings.COUCHDB_PASSWORD)

            # Fetch the taxa document
            taxa_url = f"{couchdb_url}/{taxa_db}/{taxa_id}"
            taxa_response = requests.get(taxa_url, auth=auth, timeout=30)

            if taxa_response.status_code == 404:
                return Response(
                    {'error': f'Taxa document not found: {taxa_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            taxa_response.raise_for_status()
            taxa_doc = taxa_response.json()

            # Get spans for the requested field
            spans_key = f'{field}_spans'
            raw_spans = taxa_doc.get(spans_key, [])

            if not raw_spans:
                return Response(
                    {'error': f'No {field} spans found in taxa document'},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Parse all spans
            spans = [parse_span(s) for s in raw_spans]

            # Get ingest information
            ingest = taxa_doc.get('ingest', {})
            ingest_db = ingest.get('db_name') or 'skol_dev'  # Default to skol_dev
            ingest_doc_id = ingest.get('_id')

            if not ingest_doc_id:
                return Response(
                    {'error': 'Taxa document does not have ingest._id information'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Fetch annotated file from the ingest document
            # Spans are computed from .ann files, so we must read from .ann for correct offsets
            # Use stored attachment_name if available, otherwise guess
            stored_attachment = taxa_doc.get('attachment_name')
            text_response = None
            attachment_name = None

            if stored_attachment:
                # Use the stored attachment name directly
                attachment_url = f"{couchdb_url}/{ingest_db}/{ingest_doc_id}/{stored_attachment}"
                text_response = requests.get(attachment_url, auth=auth, timeout=60)
                if text_response.status_code == 200:
                    attachment_name = stored_attachment

            if attachment_name is None:
                # Fall back to guessing: try article.pdf.ann first, then article.txt.ann
                for ann_name in ['article.pdf.ann', 'article.txt.ann']:
                    attachment_url = f"{couchdb_url}/{ingest_db}/{ingest_doc_id}/{ann_name}"
                    text_response = requests.get(attachment_url, auth=auth, timeout=60)
                    if text_response.status_code == 200:
                        attachment_name = ann_name
                        break

            if text_response is None or text_response.status_code == 404:
                return Response(
                    {'error': f'No .ann file found in ingest document: {ingest_db}/{ingest_doc_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            text_response.raise_for_status()
            article_text = text_response.text

            # Find the overall range: from start of first span to end of last span
            first_span_start = min(s.get('start_char', 0) for s in spans)
            last_span_end = max(s.get('end_char', len(article_text)) for s in spans)

            # Calculate window boundaries with context
            window_start = max(0, first_span_start - context_chars)
            window_end = min(len(article_text), last_span_end + context_chars)

            # Extract the window text
            window_text = article_text[window_start:window_end]

            # Build list of highlight regions (relative to window)
            # Sort spans by start_char to process in order
            sorted_spans = sorted(spans, key=lambda s: s.get('start_char', 0))

            # Insert highlight markers for all spans
            # Work backwards to avoid offset issues when inserting tags
            highlights = []
            for span in sorted_spans:
                start = span.get('start_char', 0) - window_start
                end = span.get('end_char', 0) - window_start
                # Clamp to window boundaries
                start = max(0, min(start, len(window_text)))
                end = max(0, min(end, len(window_text)))
                if start < end:
                    highlights.append((start, end))

            # Merge overlapping highlights
            merged_highlights = []
            for start, end in highlights:
                if merged_highlights and start <= merged_highlights[-1][1]:
                    # Overlapping or adjacent - merge
                    merged_highlights[-1] = (merged_highlights[-1][0], max(end, merged_highlights[-1][1]))
                else:
                    merged_highlights.append((start, end))

            # Insert <mark> tags working backwards to preserve offsets
            highlighted_text = window_text
            for start, end in reversed(merged_highlights):
                highlighted_text = (
                    highlighted_text[:start]
                    + '<mark>'
                    + highlighted_text[start:end]
                    + '</mark>'
                    + highlighted_text[end:]
                )

            # Get metadata from first span (for page info)
            first_span = spans[0]

            return Response({
                'source_text': highlighted_text,
                'total_spans': len(spans),
                'pdf_page': first_span.get('pdf_page'),
                'pdf_label': first_span.get('pdf_label'),
                'empirical_page': first_span.get('empirical_page'),
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"CouchDB request failed for taxa {taxa_id} context: {e}")
            return Response(
                {'error': f'Failed to fetch context: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            logger.error(f"Source context error: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class UserSettingsView(APIView):
    """
    GET/PUT user settings for search and embargo preferences.

    GET /api/user-settings/
    Returns the user's settings or creates defaults if none exist.

    PUT /api/user-settings/
    Updates user settings (partial update supported).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get user settings."""
        from .models import UserSettings
        from .serializers import UserSettingsSerializer

        settings, created = UserSettings.objects.get_or_create(user=request.user)
        serializer = UserSettingsSerializer(settings)
        return Response(serializer.data)

    def put(self, request):
        """Update user settings."""
        from .models import UserSettings
        from .serializers import UserSettingsSerializer

        settings, created = UserSettings.objects.get_or_create(user=request.user)
        serializer = UserSettingsSerializer(settings, data=request.data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
