"""
URL configuration for search app.
"""
from django.urls import path
from .views import (
    # Existing views
    SearchView,
    EmbeddingListView,
    TaxaInfoView,
    PDFAttachmentView,
    PDFFromTaxaView,
    # Collection views
    IdentifierTypeListView,
    FungariaListView,
    CollectionListCreateView,
    CollectionDetailView,
    CollectionByUserView,
    SearchHistoryListCreateView,
    SearchHistoryDetailView,
    ExternalIdentifierListCreateView,
    ExternalIdentifierDetailView,
    # Vocabulary tree views
    VocabTreeView,
    VocabTreeVersionsView,
    VocabTreeChildrenView,
)

app_name = 'search'

urlpatterns = [
    # Existing endpoints
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

    # Identifier types (reference data)
    path('identifier-types/', IdentifierTypeListView.as_view(), name='identifier-types'),

    # Fungaria list (for fungarium identifier dropdown)
    path('fungaria/', FungariaListView.as_view(), name='fungaria-list'),

    # Collections
    path('collections/', CollectionListCreateView.as_view(), name='collection-list-create'),
    path('collections/<int:collection_id>/', CollectionDetailView.as_view(), name='collection-detail'),
    path('collections/user/<str:username>/', CollectionByUserView.as_view(), name='collection-by-user'),

    # Search history within collections
    path('collections/<int:collection_id>/searches/',
         SearchHistoryListCreateView.as_view(), name='search-history-list-create'),
    path('collections/<int:collection_id>/searches/<int:search_id>/',
         SearchHistoryDetailView.as_view(), name='search-history-detail'),

    # External identifiers within collections
    path('collections/<int:collection_id>/identifiers/',
         ExternalIdentifierListCreateView.as_view(), name='identifier-list-create'),
    path('collections/<int:collection_id>/identifiers/<int:identifier_id>/',
         ExternalIdentifierDetailView.as_view(), name='identifier-detail'),

    # Vocabulary tree endpoints
    path('vocab-tree/', VocabTreeView.as_view(), name='vocab-tree'),
    path('vocab-tree/versions/', VocabTreeVersionsView.as_view(), name='vocab-tree-versions'),
    path('vocab-tree/children/', VocabTreeChildrenView.as_view(), name='vocab-tree-children'),
]
