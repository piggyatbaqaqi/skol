"""
URL configuration for search app.
"""
from django.urls import path
from .views import (
    # Existing views
    SearchView,
    NomenclatureSearchView,
    EmbeddingListView,
    BuildEmbeddingView,
    TaxaInfoView,
    PDFAttachmentView,
    PDFFromTaxaView,
    # Collection views
    IdentifierTypeListView,
    FungariaListView,
    CollectionListCreateView,
    CollectionDetailView,
    CollectionByUserView,
    CollectionByUserIdView,
    SearchHistoryListCreateView,
    SearchHistoryDetailView,
    NomenclatureChangeView,
    ExternalIdentifierListCreateView,
    ExternalIdentifierDetailView,
    GuestCollectionImportView,
    # Vocabulary tree views
    VocabTreeView,
    VocabTreeVersionsView,
    VocabTreeChildrenView,
    BuildVocabTreeView,
    # Classifier views
    TextClassifierView,
    JsonClassifierView,
    # Source context viewer
    SourceContextView,
    # User settings
    UserSettingsView,
    # Comment/Discussion views
    CommentListCreateView,
    CommentCountView,
    CommentDetailView,
    CommentFlagView,
    CommentHideView,
    CommentCopyNomenclatureView,
)

app_name = 'search'

urlpatterns = [
    # Existing endpoints
    path('embeddings/', EmbeddingListView.as_view(), name='embeddings'),
    path('embeddings/build/', BuildEmbeddingView.as_view(), name='embeddings-build'),
    path('search/nomenclature/', NomenclatureSearchView.as_view(), name='nomenclature-search'),
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
    path('collections/user-id/<int:user_id>/', CollectionByUserIdView.as_view(), name='collection-by-user-id'),
    path('collections/import-guest/', GuestCollectionImportView.as_view(), name='collection-import-guest'),

    # Search history within collections
    path('collections/<int:collection_id>/searches/',
         SearchHistoryListCreateView.as_view(), name='search-history-list-create'),
    path('collections/<int:collection_id>/searches/<int:search_id>/',
         SearchHistoryDetailView.as_view(), name='search-history-detail'),

    # Nomenclature change events
    path('collections/<int:collection_id>/nomenclature-changes/',
         NomenclatureChangeView.as_view(), name='nomenclature-change'),

    # External identifiers within collections
    path('collections/<int:collection_id>/identifiers/',
         ExternalIdentifierListCreateView.as_view(), name='identifier-list-create'),
    path('collections/<int:collection_id>/identifiers/<int:identifier_id>/',
         ExternalIdentifierDetailView.as_view(), name='identifier-detail'),

    # Vocabulary tree endpoints
    path('vocab-tree/', VocabTreeView.as_view(), name='vocab-tree'),
    path('vocab-tree/versions/', VocabTreeVersionsView.as_view(), name='vocab-tree-versions'),
    path('vocab-tree/children/', VocabTreeChildrenView.as_view(), name='vocab-tree-children'),
    path('vocab-tree/build/', BuildVocabTreeView.as_view(), name='vocab-tree-build'),

    # Classifier endpoints
    path('classifier/text/', TextClassifierView.as_view(), name='classifier-text'),
    path('classifier/json/', JsonClassifierView.as_view(), name='classifier-json'),

    # Source context viewer endpoint
    path('taxa/<str:taxa_id>/context/', SourceContextView.as_view(), name='taxa-context'),

    # User settings
    path('user-settings/', UserSettingsView.as_view(), name='user-settings'),

    # Comments/Discussion
    path(
        'collections/<int:collection_id>/comments/',
        CommentListCreateView.as_view(),
        name='comment-list-create',
    ),
    path(
        'collections/<int:collection_id>/comments/count/',
        CommentCountView.as_view(),
        name='comment-count',
    ),
    path(
        'collections/<int:collection_id>/comments/'
        '<str:comment_id>/',
        CommentDetailView.as_view(),
        name='comment-detail',
    ),
    path(
        'collections/<int:collection_id>/comments/'
        '<str:comment_id>/flag/',
        CommentFlagView.as_view(),
        name='comment-flag',
    ),
    path(
        'collections/<int:collection_id>/comments/'
        '<str:comment_id>/hide/',
        CommentHideView.as_view(),
        name='comment-hide',
    ),
    path(
        'collections/<int:collection_id>/comments/'
        '<str:comment_id>/copy-nomenclature/',
        CommentCopyNomenclatureView.as_view(),
        name='comment-copy-nomenclature',
    ),
]
