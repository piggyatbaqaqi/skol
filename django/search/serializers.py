"""
Django REST Framework serializers for the Collections feature.
"""
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Collection, SearchHistory, ExternalIdentifier, IdentifierType


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
            'id', 'prompt', 'embedding_name', 'k',
            'result_references', 'result_count', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class CollectionListSerializer(serializers.ModelSerializer):
    """Serializer for collection list view (minimal fields)."""

    owner_username = serializers.CharField(source='owner.username', read_only=True)
    search_count = serializers.SerializerMethodField()
    identifier_count = serializers.SerializerMethodField()

    class Meta:
        model = Collection
        fields = [
            'collection_id', 'name', 'description', 'notes', 'owner_username',
            'search_count', 'identifier_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['collection_id', 'created_at', 'updated_at']

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

    class Meta:
        model = Collection
        fields = [
            'collection_id', 'name', 'description', 'notes', 'owner_username',
            'search_history', 'external_identifiers',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['collection_id', 'created_at', 'updated_at']


class CollectionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating collections."""

    class Meta:
        model = Collection
        fields = ['name', 'description', 'notes']


class CollectionUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating collections (name/description/notes)."""

    class Meta:
        model = Collection
        fields = ['name', 'description', 'notes']
