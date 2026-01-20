"""
URL configuration for skolweb project.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.shortcuts import render
from django.conf import settings


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


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),
    path('accounts/', include('accounts.urls')),
    path('contact/', include('contact.urls')),
    path('about/', TemplateView.as_view(template_name='about.html'), name='about'),
    path('pdf/', pdf_viewer, name='pdf-viewer'),
    path('pdf/taxa/<str:taxa_id>/', pdf_viewer, name='pdf-viewer-taxa'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
