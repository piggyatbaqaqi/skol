"""
Django REST Framework serializers for the Collections feature.
"""
from rest_framework import serializers
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from .models import Collection, SearchHistory, ExternalIdentifier, IdentifierType, UserSettings


class IdentifierTypeSerializer(serializers.ModelSerializer):
    """Serializer for identifier types."""

    class Meta:
        model = IdentifierType
        fields = ['id', 'code', 'name', 'url_pattern', 'description']
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
    url = serializers.CharField(read_only=True)
    # For fungarium identifiers, include the fungarium organization name
    fungarium_name = serializers.SerializerMethodField()

    class Meta:
        model = ExternalIdentifier
        fields = [
            'id', 'identifier_type', 'identifier_type_code',
            'identifier_type_name', 'value', 'fungarium_code',
            'fungarium_name', 'notes', 'url', 'created_at'
        ]
        read_only_fields = ['id', 'created_at', 'url', 'fungarium_name']

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
            'feature_taxa_count', 'feature_max_tree_depth',
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
