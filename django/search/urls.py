"""
URL configuration for search app.
"""
from django.urls import path
from .views import SearchView, EmbeddingListView

app_name = 'search'

urlpatterns = [
    path('embeddings/', EmbeddingListView.as_view(), name='embeddings'),
    path('search/', SearchView.as_view(), name='search'),
]
