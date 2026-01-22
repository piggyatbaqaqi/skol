"""
REST API views for SKOL semantic search.
"""
import logging
import traceback

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.http import HttpResponse
import redis
import requests
from requests.auth import HTTPBasicAuth

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
        try:
            # Connect to Redis
            r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )

            # Get all keys matching the pattern
            keys = r.keys('skol:embedding:*')

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
                result_dict = {
                    'Similarity': float(similarity),
                    'Title': row.get('taxon', ''),
                    'Description': row.get('description', ''),
                    'Feed': row.get('source', ''),
                    'URL': row.get('filename', ''),
                }

                # Add optional metadata fields if they exist
                if 'source_metadata' in row.index:
                    src_meta = row['source_metadata']
                    if isinstance(src_meta, dict):
                        result_dict['SourceMetadata'] = src_meta
                        # Extract PDF source info for direct PDF access
                        if 'db_name' in src_meta and 'doc_id' in src_meta:
                            result_dict['PDFDbName'] = src_meta['db_name']
                            result_dict['PDFDocId'] = src_meta['doc_id']
                if 'source' in row.index:
                    src = row['source']
                    if isinstance(src, dict):
                        result_dict['Source'] = src
                if 'line_number' in row.index:
                    result_dict['LineNumber'] = row['line_number']
                if 'paragraph_number' in row.index:
                    result_dict['ParagraphNumber'] = row['paragraph_number']
                if 'page_number' in row.index:
                    result_dict['PageNumber'] = row['page_number']
                if 'pdf_page' in row.index:
                    result_dict['PDFPage'] = row['pdf_page']
                if 'pdf_label' in row.index:
                    result_dict['PDFLabel'] = row['pdf_label']

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
    """

    def get(self, request, taxa_id, taxa_db='skol_taxa_dev'):
        try:
            # Build CouchDB URL for the taxa document
            couchdb_url = settings.COUCHDB_URL
            auth = HTTPBasicAuth(settings.COUCHDB_USERNAME, settings.COUCHDB_PASSWORD)

            # Fetch the taxa document
            taxa_url = f"{couchdb_url}/{taxa_db}/{taxa_id}"
            response = requests.get(taxa_url, auth=auth, timeout=30)

            if response.status_code == 404:
                return Response(
                    {'error': f'Taxa document not found: {taxa_id}'},
                    status=status.HTTP_404_NOT_FOUND
                )

            response.raise_for_status()
            taxa_doc = response.json()

            # Return taxa info with source details
            return Response({
                'taxa_id': taxa_id,
                'taxa_db': taxa_db,
                'taxon': taxa_doc.get('taxon', ''),
                'description': taxa_doc.get('description', ''),
                'source': taxa_doc.get('source', {}),
                'line_number': taxa_doc.get('line_number'),
                'paragraph_number': taxa_doc.get('paragraph_number'),
                'page_number': taxa_doc.get('page_number'),
                'pdf_page': taxa_doc.get('pdf_page'),
            })

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

            # Get source information
            source = taxa_doc.get('source', {})
            source_db = source.get('db_name')
            source_doc_id = source.get('doc_id')

            if not source_db or not source_doc_id:
                return Response(
                    {'error': 'Taxa document does not have source information'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Fetch the PDF attachment from the source document
            attachment_name = 'article.pdf'
            attachment_url = f"{couchdb_url}/{source_db}/{source_doc_id}/{attachment_name}"
            pdf_response = requests.get(attachment_url, auth=auth, timeout=60, stream=True)

            if pdf_response.status_code == 404:
                return Response(
                    {'error': f'PDF attachment not found in source document: {source_db}/{source_doc_id}'},
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
