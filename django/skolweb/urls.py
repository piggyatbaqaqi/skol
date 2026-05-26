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


def pdf_viewer(request, treatment_id=None):
    """
    Render the PDF viewer page.

    Query params:
        - db: Database name (default: skol_dev)
        - doc_id: Document ID (if not using treatment_id)
        - page: Initial page number
        - treatments_db: Taxa database (default: skol_taxa_dev)
    """
    context = {}

    if treatment_id:
        # Use the taxa endpoint to get the PDF
        treatments_db = request.GET.get('treatments_db', 'skol_taxa_dev')
        context['pdf_url'] = f"{settings.FORCE_SCRIPT_NAME or ''}/api/treatments/{treatment_id}/pdf/?treatments_db={treatments_db}"
        context['title'] = f"Taxa: {treatment_id}"
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


def project_detail_view(request, username, slug):
    """Display the project detail page (public)."""
    import json
    return render(request, 'project_detail.html', {
        'project_username': json.dumps(username),
        'project_slug': json.dumps(slug),
        'project_name': f'{username}/{slug}',
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
        'total_treatments_records': 0,
        'cached': False,
        'cached_at': None,
    }

    # Resolve the experiment-scoped Redis key.  Authenticated users
    # with an active experiment read ``skol:sources:stats:<exp>``
    # (populated by ``build_sources_stats.py --experiment <exp>``);
    # anonymous users + the v1 cron job stay on the legacy
    # ``skol:sources:stats`` key.  Mirrors the helper in
    # bin/build_sources_stats.py so both sides stay in sync.
    experiment_name = ''
    if request.user.is_authenticated:
        try:
            from search.views import get_user_experiment
            _, exp = get_user_experiment(request)
            if exp:
                experiment_name = exp.get('_id', '')
        except Exception:
            pass
    redis_key = (
        f'skol:sources:stats:{experiment_name}'
        if experiment_name else 'skol:sources:stats'
    )

    # Try to read from Redis first
    try:
        from search.utils import get_redis_client
        r = get_redis_client(decode_responses=True)

        cached_data = r.get(redis_key)
        if cached_data:
            data = json.loads(cached_data)
            context['sources'] = data.get('sources', [])
            context['total_records'] = data.get('total_records', 0)
            context['total_taxonomy_documents'] = data.get('total_taxonomy_documents', 0)
            context['total_treatments_records'] = data.get('total_treatments_records', 0)
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
    db_user = getattr(settings, 'COUCHDB_USERNAME', 'admin')
    db_password = getattr(settings, 'COUCHDB_PASSWORD', '')

    # Use the active experiment's ingest + treatments databases when
    # available (matches what build_sources_stats.py --experiment <X>
    # would compute).  Falls back to the legacy settings defaults for
    # anonymous users.  ``get_user_experiment`` was already called
    # above to resolve the Redis key — we re-fetch the doc here so the
    # slow-path stats also stay experiment-scoped.
    db_name = getattr(settings, 'INGESTION_DB_NAME', 'skol_dev')
    treatments_db_name = getattr(
        settings, 'TREATMENTS_DB_NAME', 'skol_treatments_dev',
    )
    if request.user.is_authenticated:
        try:
            from search.views import get_user_experiment
            _, exp = get_user_experiment(request)
            if exp:
                databases = exp.get('databases', {}) or {}
                db_name = databases.get('ingest', db_name)
                treatments_db_name = databases.get(
                    'treatments', treatments_db_name,
                )
        except Exception:
            pass

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
                    source_stats[journal_name] = {'total': 0, 'taxonomy': 0, 'treatments': 0}

                source_stats[journal_name]['total'] += 1
                if doc.get('taxonomy') is True:
                    source_stats[journal_name]['taxonomy'] += 1

            except Exception:
                continue

        # Use the experiment-scoped treatments DB resolved above; do NOT
        # re-read settings here.  The stats dict uses the canonical
        # ``treatments`` key (matches bin/build_sources_stats.py and the
        # source_stats initialiser a few lines up at line 213).
        if treatments_db_name in server:
            treatments_db = server[treatments_db_name]
            for treatment_doc_id in treatments_db:
                if treatment_doc_id.startswith('_design/'):
                    continue
                try:
                    treatment_doc = treatments_db[treatment_doc_id]
                    ingest = treatment_doc.get('ingest', {})
                    ingest_doc_id = ingest.get('_id')
                    if ingest_doc_id and ingest_doc_id in doc_to_journal:
                        journal_name = doc_to_journal[ingest_doc_id]
                        source_stats[journal_name]['treatments'] += 1
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
                'treatments_records': stats['treatments'],
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
            context['total_treatments_records'] += stats['treatments']

        context['sources'].sort(key=lambda x: x['name'].lower())

    except Exception as e:
        context['error'] = f"Error connecting to CouchDB: {str(e)}"

    return render(request, 'sources.html', context)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),
    path('accounts/', include('accounts.urls')),  # Custom accounts (login, register, etc.)
    path('accounts/', include('allauth.urls')),   # allauth OAuth callbacks (github/, google/, orcid/)
    path('accounts/', include('inaturalist_provider.urls')),  # iNaturalist OAuth2
    path('contact/', include('contact.urls')),
    path('about/', TemplateView.as_view(template_name='about.html'), name='about'),
    path('help/', TemplateView.as_view(template_name='help.html'), name='help'),
    path('sources/', sources_view, name='sources'),
    path('pdf/', pdf_viewer, name='pdf-viewer'),
    path('pdf/treatments/<str:treatment_id>/', pdf_viewer, name='pdf-viewer-treatments'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('collections/', collections_view, name='collections'),
    path('collections/<int:collection_id>/', collection_detail_view, name='collection-detail-page'),
    path('projects/<str:username>/<slug:slug>/', project_detail_view, name='project-detail-page'),
]
