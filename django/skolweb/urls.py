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


def sources_view(request):
    """
    Display ingestion source statistics.

    Shows information about each publication source including:
    - Publication name
    - Publisher/source website
    - Number of ingested records
    - Number of records with taxonomy
    """
    import couchdb

    # Add ingestors path for PublicationRegistry
    ingestors_path = str(Path(__file__).resolve().parent.parent.parent / 'ingestors')
    if ingestors_path not in sys.path:
        sys.path.insert(0, ingestors_path)

    try:
        from ingestors.publications import PublicationRegistry
    except ImportError:
        # Fallback if module isn't available
        PublicationRegistry = None

    # Get database settings
    db_host = getattr(settings, 'COUCHDB_HOST', '127.0.0.1:5984')
    db_name = getattr(settings, 'INGESTION_DB_NAME', 'skol_dev')
    db_user = getattr(settings, 'COUCHDB_USER', 'admin')
    db_password = getattr(settings, 'COUCHDB_PASSWORD', '')

    context = {
        'sources': [],
        'error': None,
        'total_records': 0,
        'total_taxonomy_records': 0,
    }

    try:
        # Connect to CouchDB
        couchdb_url = f"http://{db_host}"
        server = couchdb.Server(couchdb_url)
        if db_user and db_password:
            server.resource.credentials = (db_user, db_password)

        # Access the database
        if db_name not in server:
            context['error'] = f"Database '{db_name}' not found"
            return render(request, 'sources.html', context)

        db = server[db_name]

        # Collect statistics by source
        source_stats = {}

        for doc_id in db:
            # Skip design documents
            if doc_id.startswith('_design/'):
                continue

            try:
                doc = db[doc_id]

                # Get the publication source from the document
                pub_source = doc.get('publication')
                if not pub_source:
                    pub_source = 'unknown'

                # Initialize stats for this source if not seen before
                if pub_source not in source_stats:
                    source_stats[pub_source] = {
                        'total': 0,
                        'taxonomy': 0,
                    }

                # Increment total count
                source_stats[pub_source]['total'] += 1

                # Check if document has taxonomy
                if doc.get('taxonomy') is True:
                    source_stats[pub_source]['taxonomy'] += 1

            except Exception as e:
                # Skip documents we can't read
                continue

        # Build display list with publication details
        for pub_key, stats in source_stats.items():
            source_info = {
                'key': pub_key,
                'name': pub_key,  # Default to key
                'publisher': 'Unknown',
                'website': None,
                'total_records': stats['total'],
                'taxonomy_records': stats['taxonomy'],
                'taxonomy_percentage': round((stats['taxonomy'] / stats['total'] * 100) if stats['total'] > 0 else 0, 1),
            }

            # Try to get additional information from PublicationRegistry
            if PublicationRegistry:
                pub_config = PublicationRegistry.get(pub_key)
                if pub_config:
                    source_info['name'] = pub_config.get('name', pub_key)
                    source_info['publisher'] = pub_config.get('source', 'Unknown')

                    # Try to extract website from various URL fields
                    for url_field in ['rss_url', 'index_url', 'archives_url', 'issues_url', 'url_prefix']:
                        if url_field in pub_config and pub_config[url_field]:
                            url = pub_config[url_field]
                            # Extract domain from URL
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            if parsed.netloc:
                                source_info['website'] = f"{parsed.scheme}://{parsed.netloc}"
                                break

            context['sources'].append(source_info)
            context['total_records'] += stats['total']
            context['total_taxonomy_records'] += stats['taxonomy']

        # Sort sources by name
        context['sources'].sort(key=lambda x: x['name'].lower())

    except Exception as e:
        context['error'] = f"Error connecting to CouchDB: {str(e)}"

    return render(request, 'sources.html', context)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),
    path('accounts/', include('accounts.urls')),
    path('contact/', include('contact.urls')),
    path('about/', TemplateView.as_view(template_name='about.html'), name='about'),
    path('sources/', sources_view, name='sources'),
    path('pdf/', pdf_viewer, name='pdf-viewer'),
    path('pdf/taxa/<str:taxa_id>/', pdf_viewer, name='pdf-viewer-taxa'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
