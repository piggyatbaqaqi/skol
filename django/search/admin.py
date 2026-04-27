"""
Django admin configuration for search app models.
"""
from django.contrib import admin
from .models import (
    IdentifierType,
    Collection,
    SearchHistory,
    ExternalIdentifier,
    MeasurementUnit,
    Project,
    CollectionProject,
    CollectionProjectRemoval,
)


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


@admin.register(MeasurementUnit)
class MeasurementUnitAdmin(admin.ModelAdmin):
    """Admin configuration for MeasurementUnit model."""
    list_display = ['symbol', 'sort_order']
    list_editable = ['sort_order']
    ordering = ['sort_order', 'symbol']


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    """Admin configuration for Project model.

    Only admins can delete projects (enforced by the democratic governance
    model: regular users add/remove collections but cannot delete projects).
    """
    list_display = ['slug', 'name', 'creator', 'collection_count', 'created_at']
    list_filter = ['created_at', 'creator']
    search_fields = ['name', 'slug', 'creator__username', 'description']
    readonly_fields = ['slug', 'creator', 'created_at']
    ordering = ['creator__username', 'slug']

    def collection_count(self, obj: Project) -> int:
        return obj.collections.count()
    collection_count.short_description = 'Collections'

    def has_delete_permission(self, request, obj=None):
        """Only admins (staff/superusers) may delete projects."""
        return request.user.is_staff or request.user.is_superuser


@admin.register(CollectionProject)
class CollectionProjectAdmin(admin.ModelAdmin):
    """Admin view of current collection–project memberships."""
    list_display = ['collection', 'project', 'added_by', 'added_at']
    list_filter = ['project', 'added_at']
    search_fields = [
        'collection__name', 'project__name',
        'added_by__username',
    ]
    readonly_fields = ['added_at']
    raw_id_fields = ['collection', 'project', 'added_by']
    ordering = ['-added_at']


@admin.register(CollectionProjectRemoval)
class CollectionProjectRemovalAdmin(admin.ModelAdmin):
    """Admin view of collection–project removal audit log (read-only)."""
    list_display = [
        'collection', 'project', 'removed_by', 'removed_at',
        'original_added_by', 'original_added_at',
    ]
    list_filter = ['project', 'removed_at']
    search_fields = [
        'collection__name', 'project__name',
        'removed_by__username', 'original_added_by__username',
    ]
    readonly_fields = [
        'collection', 'project', 'removed_by', 'removed_at',
        'original_added_by', 'original_added_at',
    ]
    raw_id_fields = ['collection', 'project', 'removed_by', 'original_added_by']
    ordering = ['-removed_at']

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
