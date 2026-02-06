"""
URL configuration for skolweb project.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.shortcuts import render
from django.conf import settings
import sys
from pathlib import Path


def pdf_viewer(request, taxa_id=None):
    """
    Render the PDF viewer page.

    Query params:
        - db: Database name (default: skol_dev)
        - doc_id: Document ID (if not using taxa_id)
        - page: Initial page number
        - taxa_db: Taxa database (default: skol_taxa_dev)
    """
    context = {}

    if taxa_id:
        # Use the taxa endpoint to get the PDF
        taxa_db = request.GET.get('taxa_db', 'skol_taxa_dev')
        context['pdf_url'] = f"{settings.FORCE_SCRIPT_NAME or ''}/api/taxa/{taxa_id}/pdf/?taxa_db={taxa_db}"
        context['title'] = f"Taxa: {taxa_id}"
        context['initial_page'] = request.GET.get('page', 1)
    else:
        # Direct database/document access
        db_name = request.GET.get('db', 'skol_dev')
        doc_id = request.GET.get('doc_id')
        attachment = request.GET.get('attachment', 'article.pdf')

        if doc_id:
            context['pdf_url'] = f"{settings.FORCE_SCRIPT_NAME or ''}/api/pdf/{db_name}/{doc_id}/{attachment}"
            context['title'] = f"Document: {doc_id}"
        else:
            context['error'] = "No document ID provided"
            context['pdf_url'] = ''
            context['title'] = 'PDF Viewer'

        context['initial_page'] = request.GET.get('page', 1)

    return render(request, 'pdf_viewer.html', context)


def collections_view(request):
    """
    Display the user's collections page.
    Requires authentication.
    """
    if not request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect(f"{settings.LOGIN_URL}?next={request.path}")
    return render(request, 'collections.html', {})


def collection_detail_view(request, collection_id):
    """
    Display a specific collection's detail page.
    Requires authentication.
    """
    if not request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect(f"{settings.LOGIN_URL}?next={request.path}")
    return render(request, 'collection_detail.html', {
        'collection_id': collection_id,
    })


def sources_view(request):
    """
    Display ingestion source statistics.

    Shows information about each publication source including:
    - Publication name
    - Publisher/source website
    - Number of ingested records
    - Number of records with taxonomy

    Statistics are read from Redis (pre-computed by bin/build_sources_stats.py).
    Falls back to on-the-fly calculation if Redis data is not available.
    """
    import json
    import redis

    context = {
        'sources': [],
        'error': None,
        'total_records': 0,
        'total_taxonomy_documents': 0,
        'total_taxa_records': 0,
        'cached': False,
        'cached_at': None,
    }

    # Try to read from Redis first
    try:
        from search.utils import get_redis_client
        r = get_redis_client(decode_responses=True)

        cached_data = r.get('skol:sources:stats')
        if cached_data:
            data = json.loads(cached_data)
            context['sources'] = data.get('sources', [])
            context['total_records'] = data.get('total_records', 0)
            context['total_taxonomy_documents'] = data.get('total_taxonomy_documents', 0)
            context['total_taxa_records'] = data.get('total_taxa_records', 0)
            context['cached'] = True
            context['cached_at'] = data.get('created_at')
            return render(request, 'sources.html', context)

    except (redis.RedisError, json.JSONDecodeError, ImportError) as e:
        # Redis not available or data invalid, fall back to on-the-fly calculation
        pass

    # Fall back to on-the-fly calculation (slow)
    import couchdb

    # Add ingestors path for PublicationRegistry
    ingestors_path = str(Path(__file__).resolve().parent.parent.parent / 'ingestors')
    if ingestors_path not in sys.path:
        sys.path.insert(0, ingestors_path)

    try:
        from ingestors.publications import PublicationRegistry
    except ImportError:
        PublicationRegistry = None

    couchdb_url = getattr(settings, 'COUCHDB_URL', 'http://127.0.0.1:5984')
    db_name = getattr(settings, 'INGESTION_DB_NAME', 'skol_dev')
    db_user = getattr(settings, 'COUCHDB_USERNAME', 'admin')
    db_password = getattr(settings, 'COUCHDB_PASSWORD', '')

    try:
        server = couchdb.Server(couchdb_url)
        if db_user and db_password:
            server.resource.credentials = (db_user, db_password)

        if db_name not in server:
            context['error'] = f"Database '{db_name}' not found"
            return render(request, 'sources.html', context)

        db = server[db_name]
        source_stats = {}
        doc_to_journal = {}

        for doc_id in db:
            if doc_id.startswith('_design/'):
                continue

            try:
                doc = db[doc_id]
                journal_name = doc.get('journal') or 'Unknown'
                doc_to_journal[doc_id] = journal_name

                if journal_name not in source_stats:
                    source_stats[journal_name] = {'total': 0, 'taxonomy': 0, 'taxa': 0}

                source_stats[journal_name]['total'] += 1
                if doc.get('taxonomy') is True:
                    source_stats[journal_name]['taxonomy'] += 1

            except Exception:
                continue

        taxa_db_name = getattr(settings, 'TAXA_DB_NAME', 'skol_taxa_dev')
        if taxa_db_name in server:
            taxa_db = server[taxa_db_name]
            for taxa_doc_id in taxa_db:
                if taxa_doc_id.startswith('_design/'):
                    continue
                try:
                    taxa_doc = taxa_db[taxa_doc_id]
                    ingest = taxa_doc.get('ingest', {})
                    ingest_doc_id = ingest.get('_id')
                    if ingest_doc_id and ingest_doc_id in doc_to_journal:
                        journal_name = doc_to_journal[ingest_doc_id]
                        source_stats[journal_name]['taxa'] += 1
                except Exception:
                    continue

        for journal_name, stats in source_stats.items():
            source_info = {
                'key': journal_name,
                'name': journal_name,
                'publisher': 'Unknown',
                'website': None,
                'total_records': stats['total'],
                'taxonomy_records': stats['taxonomy'],
                'taxonomy_percentage': round(
                    (stats['taxonomy'] / stats['total'] * 100) if stats['total'] > 0 else 0, 1
                ),
                'taxa_records': stats['taxa'],
            }

            if PublicationRegistry:
                pub_config = PublicationRegistry.get_by_journal(journal_name)
                if pub_config:
                    source_info['name'] = pub_config.get('name', journal_name)
                    source_info['publisher'] = pub_config.get('source', 'Unknown')
                    if pub_config.get('address'):
                        source_info['website'] = pub_config['address']
                    else:
                        for url_field in ['rss_url', 'index_url', 'archives_url', 'issues_url', 'url_prefix']:
                            if url_field in pub_config and pub_config[url_field]:
                                from urllib.parse import urlparse
                                parsed = urlparse(pub_config[url_field])
                                if parsed.netloc:
                                    source_info['website'] = f"{parsed.scheme}://{parsed.netloc}"
                                    break

            context['sources'].append(source_info)
            context['total_records'] += stats['total']
            context['total_taxonomy_documents'] += stats['taxonomy']
            context['total_taxa_records'] += stats['taxa']

        context['sources'].sort(key=lambda x: x['name'].lower())

    except Exception as e:
        context['error'] = f"Error connecting to CouchDB: {str(e)}"

    return render(request, 'sources.html', context)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),
    path('accounts/', include('accounts.urls')),  # Custom accounts (login, register, etc.)
    path('accounts/', include('allauth.urls')),   # allauth OAuth callbacks (github/, google/, orcid/)
    path('contact/', include('contact.urls')),
    path('about/', TemplateView.as_view(template_name='about.html'), name='about'),
    path('sources/', sources_view, name='sources'),
    path('pdf/', pdf_viewer, name='pdf-viewer'),
    path('pdf/taxa/<str:taxa_id>/', pdf_viewer, name='pdf-viewer-taxa'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('collections/', collections_view, name='collections'),
    path('collections/<int:collection_id>/', collection_detail_view, name='collection-detail-page'),
]
