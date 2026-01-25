"""
Django admin configuration for search app models.
"""
from django.contrib import admin
from .models import IdentifierType, Collection, SearchHistory, ExternalIdentifier


@admin.register(IdentifierType)
class IdentifierTypeAdmin(admin.ModelAdmin):
    """Admin configuration for IdentifierType model."""
    list_display = ['code', 'name', 'url_pattern', 'created_at']
    list_filter = ['created_at']
    search_fields = ['code', 'name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['name']


@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    """Admin configuration for Collection model."""
    list_display = ['collection_id', 'name', 'owner', 'created_at', 'updated_at']
    list_filter = ['created_at', 'owner']
    search_fields = ['collection_id', 'name', 'description', 'owner__username']
    readonly_fields = ['collection_id', 'created_at', 'updated_at']
    raw_id_fields = ['owner']
    ordering = ['-updated_at']


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    """Admin configuration for SearchHistory model."""
    list_display = ['id', 'collection', 'prompt_preview', 'embedding_name', 'k', 'result_count', 'created_at']
    list_filter = ['created_at', 'embedding_name']
    search_fields = ['prompt', 'collection__name', 'collection__owner__username']
    readonly_fields = ['created_at']
    raw_id_fields = ['collection']
    ordering = ['-created_at']

    def prompt_preview(self, obj):
        """Return truncated prompt for display."""
        return f"{obj.prompt[:50]}..." if len(obj.prompt) > 50 else obj.prompt
    prompt_preview.short_description = 'Prompt'


@admin.register(ExternalIdentifier)
class ExternalIdentifierAdmin(admin.ModelAdmin):
    """Admin configuration for ExternalIdentifier model."""
    list_display = ['id', 'collection', 'identifier_type', 'value', 'created_at']
    list_filter = ['identifier_type', 'created_at']
    search_fields = ['value', 'notes', 'collection__name', 'collection__owner__username']
    readonly_fields = ['created_at']
    raw_id_fields = ['collection']
    ordering = ['-created_at']
