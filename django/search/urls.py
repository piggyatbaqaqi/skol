"""
URL configuration for search app.
"""
from django.urls import path
from .views import (
    SearchView,
    EmbeddingListView,
    TaxaInfoView,
    PDFAttachmentView,
    PDFFromTaxaView,
)

app_name = 'search'

urlpatterns = [
    path('embeddings/', EmbeddingListView.as_view(), name='embeddings'),
    path('search/', SearchView.as_view(), name='search'),
    # Taxa document info
    path('taxa/<str:taxa_id>/', TaxaInfoView.as_view(), name='taxa-info'),
    # PDF endpoints
    path('pdf/<str:db_name>/<str:doc_id>/',
         PDFAttachmentView.as_view(), name='pdf-attachment'),
    path('pdf/<str:db_name>/<str:doc_id>/<str:attachment_name>/',
         PDFAttachmentView.as_view(), name='pdf-attachment-named'),
    path('taxa/<str:taxa_id>/pdf/',
         PDFFromTaxaView.as_view(), name='taxa-pdf'),
]
