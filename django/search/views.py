"""
REST API views for SKOL semantic search.
"""
import sys
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import redis

# Add dr-drafts-mycosearch to path (but don't import yet)
# Note: We add the parent directory, not src/, because sota_search.py does "from src import data"
dr_drafts_path = Path(__file__).resolve().parent.parent.parent.parent / 'dr-drafts-mycosearch'
if str(dr_drafts_path) not in sys.path:
    sys.path.insert(0, str(dr_drafts_path))

# Add skol to path for skol_compat module
skol_path = Path(__file__).resolve().parent.parent.parent.parent / 'skol'
if str(skol_path) not in sys.path:
    sys.path.insert(0, str(skol_path))

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
            from src.sota_search import Experiment

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
                    'URL': row.get('filename', '')
                }

                # Add optional metadata fields if they exist
                if 'source_metadata' in row.index and isinstance(row['source_metadata'], dict):
                    result_dict['SourceMetadata'] = row['source_metadata']
                if 'line_number' in row.index:
                    result_dict['LineNumber'] = row['line_number']
                if 'paragraph_number' in row.index:
                    result_dict['ParagraphNumber'] = row['paragraph_number']
                if 'page_number' in row.index:
                    result_dict['PageNumber'] = row['page_number']

                results.append(result_dict)

            return Response({
                'results': results,
                'count': len(results),
                'prompt': prompt,
                'embedding_name': embedding_name,
                'k': k
            })

        except ValueError as e:
            import traceback
            tb = traceback.format_exc()
            return Response(
                {'error': f'Embedding error: {str(e)}', 'traceback': tb},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return Response(
                {'error': f'Search failed: {str(e)}', 'traceback': tb},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
