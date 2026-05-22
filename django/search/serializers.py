"""
Django REST Framework serializers for the Collections feature.
"""
from rest_framework import serializers
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from .models import (
    Collection,
    SearchHistory,
    ExternalIdentifier,
    IdentifierType,
    UserSettings,
    MeasurementSet,
    Project,
    CollectionProject,
    ProjectNotesLog,
)


class IdentifierTypeSerializer(serializers.ModelSerializer):
    """Serializer for identifier types."""

    class Meta:
        model = IdentifierType
        fields = ['id', 'code', 'name', 'url_pattern', 'description', 'actions']
        read_only_fields = ['id']


class ExternalIdentifierSerializer(serializers.ModelSerializer):
    """Serializer for external identifiers."""

    identifier_type_code = serializers.CharField(
        source='identifier_type.code',
        read_only=True
    )
    identifier_type_name = serializers.CharField(
        source='identifier_type.name',
        read_only=True
    )
    # Inline the parent type's actions so the frontend can dispatch
    # per-type click behavior + buttons without a separate IdentifierType
    # fetch.
    identifier_type_actions = serializers.JSONField(
        source='identifier_type.actions',
        read_only=True,
    )
    url = serializers.CharField(read_only=True)
    # For fungarium identifiers, include the fungarium organization name
    fungarium_name = serializers.SerializerMethodField()

    class Meta:
        model = ExternalIdentifier
        fields = [
            'id', 'identifier_type', 'identifier_type_code',
            'identifier_type_name', 'identifier_type_actions',
            'value', 'fungarium_code',
            'fungarium_name', 'notes', 'url', 'created_at'
        ]
        read_only_fields = [
            'id', 'created_at', 'url', 'fungarium_name',
            'identifier_type_actions',
        ]

    def get_fungarium_name(self, obj) -> str:
        """Get the fungarium organization name from Redis."""
        if not obj.fungarium_code:
            return ''
        try:
            import json
            from .utils import get_redis_client

            r = get_redis_client(decode_responses=True)
            raw = r.get('skol:fungaria')
            if not raw:
                return ''

            data = json.loads(raw)
            institutions = data.get('institutions', {})
            inst = institutions.get(obj.fungarium_code)
            if inst:
                return inst.get('organization', '')
            return ''
        except Exception:
            return ''


class ExternalIdentifierCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating external identifiers."""

    identifier_type_code = serializers.SlugRelatedField(
        slug_field='code',
        queryset=IdentifierType.objects.all(),
        source='identifier_type'
    )

    class Meta:
        model = ExternalIdentifier
        fields = ['identifier_type_code', 'value', 'fungarium_code', 'notes']

    def validate(self, data):
        """Validate that fungarium_code is provided for fungarium identifiers."""
        identifier_type = data.get('identifier_type')
        fungarium_code = data.get('fungarium_code', '')

        if identifier_type and identifier_type.code == 'fungarium':
            if not fungarium_code:
                raise serializers.ValidationError({
                    'fungarium_code': 'Fungarium code is required for fungarium identifiers.'
                })
        return data


class SearchHistorySerializer(serializers.ModelSerializer):
    """Serializer for search history entries."""

    class Meta:
        model = SearchHistory
        fields = [
            'id', 'event_type', 'prompt', 'embedding_name', 'k',
            'result_references', 'result_count', 'nomenclature', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class NomenclatureChangeSerializer(serializers.ModelSerializer):
    """Serializer for creating nomenclature change events."""

    class Meta:
        model = SearchHistory
        fields = ['nomenclature']

    def create(self, validated_data):
        validated_data['event_type'] = 'nomenclature_change'
        validated_data['prompt'] = ''
        validated_data['embedding_name'] = ''
        validated_data['result_references'] = []
        validated_data['result_count'] = 0
        return super().create(validated_data)


class CollectionListSerializer(serializers.ModelSerializer):
    """Serializer for collection list view (minimal fields)."""

    owner_username = serializers.CharField(source='owner.username', read_only=True)
    search_count = serializers.SerializerMethodField()
    identifier_count = serializers.SerializerMethodField()
    is_embargoed = serializers.BooleanField(read_only=True)
    is_public = serializers.BooleanField(read_only=True)

    class Meta:
        model = Collection
        fields = [
            'collection_id', 'name', 'description', 'notes', 'nomenclature',
            'embargo_until', 'is_embargoed', 'is_public', 'owner_username',
            'search_count', 'identifier_count', 'flagged_by',
            'hidden', 'created_at', 'updated_at'
        ]
        read_only_fields = ['collection_id', 'created_at', 'updated_at',
                           'is_embargoed', 'is_public', 'flagged_by',
                           'hidden']

    def get_search_count(self, obj: Collection) -> int:
        """Get the count of search history entries."""
        return obj.search_history.count()

    def get_identifier_count(self, obj: Collection) -> int:
        """Get the count of external identifiers."""
        return obj.external_identifiers.count()


class CollectionDetailSerializer(serializers.ModelSerializer):
    """Serializer for collection detail view (full data)."""

    owner_username = serializers.CharField(source='owner.username', read_only=True)
    search_history = SearchHistorySerializer(many=True, read_only=True)
    external_identifiers = ExternalIdentifierSerializer(many=True, read_only=True)
    is_embargoed = serializers.BooleanField(read_only=True)
    is_public = serializers.BooleanField(read_only=True)

    class Meta:
        model = Collection
        fields = [
            'collection_id', 'name', 'description', 'notes', 'nomenclature',
            'embargo_until', 'is_embargoed', 'is_public', 'owner_username',
            'search_history', 'external_identifiers', 'flagged_by',
            'hidden', 'hidden_by', 'created_at', 'updated_at'
        ]
        read_only_fields = ['collection_id', 'created_at', 'updated_at',
                           'is_embargoed', 'is_public', 'flagged_by',
                           'hidden', 'hidden_by']


class CollectionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating collections."""

    class Meta:
        model = Collection
        fields = ['name', 'description', 'notes', 'nomenclature', 'embargo_until']

    def validate_embargo_until(self, value):
        """Validate embargo_until is null or within 1 year."""
        if value is None:
            return value
        max_embargo = timezone.now() + timedelta(days=365)
        if value > max_embargo:
            raise serializers.ValidationError(
                "Embargo date cannot be more than 1 year in the future."
            )
        return value


class CollectionUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating collections."""

    class Meta:
        model = Collection
        fields = ['name', 'description', 'notes', 'nomenclature', 'embargo_until']

    def validate_embargo_until(self, value):
        """Validate embargo_until is null or within 1 year."""
        if value is None:
            return value
        max_embargo = timezone.now() + timedelta(days=365)
        if value > max_embargo:
            raise serializers.ValidationError(
                "Embargo date cannot be more than 1 year in the future."
            )
        return value


class UserSettingsSerializer(serializers.ModelSerializer):
    """Serializer for user settings."""

    class Meta:
        model = UserSettings
        fields = [
            'default_embargo_days', 'default_embedding', 'default_k',
            'results_per_page', 'nomenclature_limit',
            'feature_taxa_count', 'feature_top_n',
            'feature_max_tree_depth', 'feature_min_df', 'feature_max_df',
            'default_experiment',
            'receive_admin_summary',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']

    def validate_default_embargo_days(self, value):
        """Validate embargo days is within 1 year."""
        if value > 365:
            raise serializers.ValidationError(
                "Default embargo cannot be more than 365 days."
            )
        return value


class MeasurementSetSerializer(serializers.ModelSerializer):
    """Serializer for measurement sets (spore dimensions, etc.)."""

    class Meta:
        model = MeasurementSet
        fields = [
            'id', 'feature', 'is_2d', 'report_q', 'unit', 'measurements',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def validate_measurements(self, value):
        """Validate measurements array structure."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Measurements must be a list.")
        for i, m in enumerate(value):
            if not isinstance(m, dict):
                raise serializers.ValidationError(
                    f"Measurement {i} must be an object."
                )
            if 'length' not in m:
                raise serializers.ValidationError(
                    f"Measurement {i} must have a 'length' field."
                )
            if not isinstance(m['length'], (int, float)) or m['length'] <= 0:
                raise serializers.ValidationError(
                    f"Measurement {i} 'length' must be a positive number."
                )
            if ('width' in m and m['width'] is not None
                    and (not isinstance(m['width'], (int, float)) or m['width'] <= 0)):
                raise serializers.ValidationError(
                    f"Measurement {i} 'width' must be a positive number."
                )
        return value


# ===========================================================================
# Project serializers
# ===========================================================================

class ProjectSerializer(serializers.ModelSerializer):
    """Read serializer for Project (list and detail)."""

    creator_username = serializers.CharField(
        source='creator.username', read_only=True
    )
    collection_count = serializers.SerializerMethodField()
    namespaced_slug = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id', 'name', 'slug', 'namespaced_slug',
            'creator_username', 'description', 'notes',
            'collection_count', 'created_at',
        ]
        read_only_fields = fields

    def get_collection_count(self, obj: Project) -> int:
        return obj.collections.count()

    def get_namespaced_slug(self, obj: Project) -> str:
        return f"{obj.creator.username}/{obj.slug}"


class ProjectNotesLogSerializer(serializers.ModelSerializer):
    """Read serializer for ProjectNotesLog entries."""

    changed_by_username = serializers.CharField(
        source='changed_by.username', read_only=True
    )

    class Meta:
        model = ProjectNotesLog
        fields = ['id', 'changed_by_username', 'changed_at', 'diff']
        read_only_fields = fields


class ProjectCreateSerializer(serializers.ModelSerializer):
    """Write serializer for Project creation."""

    class Meta:
        model = Project
        fields = ['name', 'description']

    def validate_name(self, value: str) -> str:
        """Ensure name produces a valid slug."""
        try:
            Project.generate_slug(value)
        except ValueError as exc:
            raise serializers.ValidationError(str(exc)) from exc
        return value


class CollectionProjectSerializer(serializers.ModelSerializer):
    """Serializer for membership records returned on project detail."""

    added_by_username = serializers.CharField(
        source='added_by.username', read_only=True
    )
    collection_name = serializers.CharField(
        source='collection.name', read_only=True
    )
    collection_id = serializers.IntegerField(
        source='collection.collection_id', read_only=True
    )

    class Meta:
        model = CollectionProject
        fields = [
            'collection_id', 'collection_name',
            'added_by_username', 'added_at',
        ]
        read_only_fields = fields


class UserSettingsWithProjectsSerializer(serializers.ModelSerializer):
    """Serializer for user settings including default_projects list."""

    default_project_slugs = serializers.SerializerMethodField()

    class Meta:
        model = UserSettings
        fields = [
            'default_embargo_days', 'default_embedding', 'default_k',
            'results_per_page', 'nomenclature_limit',
            'feature_taxa_count', 'feature_top_n',
            'feature_max_tree_depth', 'feature_min_df', 'feature_max_df',
            'default_experiment',
            'receive_admin_summary',
            'default_project_slugs',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'default_project_slugs']

    def get_default_project_slugs(self, obj: UserSettings):
        return [
            f"{p.creator.username}/{p.slug}"
            for p in obj.default_projects.select_related('creator').all()
        ]

    def validate_default_embargo_days(self, value: int) -> int:
        if value > 365:
            raise serializers.ValidationError(
                "Default embargo cannot be more than 365 days."
            )
        return value
