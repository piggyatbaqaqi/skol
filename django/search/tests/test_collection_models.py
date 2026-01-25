"""
Tests for Collection models.

Run with: pytest search/tests/test_collection_models.py -v
"""
import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from django.db import IntegrityError
from search.models import (
    Collection,
    SearchHistory,
    ExternalIdentifier,
    IdentifierType,
    generate_collection_id,
)


class TestIdentifierType(TestCase):
    """Tests for IdentifierType model."""

    def test_create_identifier_type(self) -> None:
        """Test creating an identifier type."""
        it = IdentifierType.objects.create(
            code='test',
            name='Test Type',
            url_pattern='https://example.com/{id}'
        )
        assert it.code == 'test'
        assert it.name == 'Test Type'

    def test_build_url(self) -> None:
        """Test URL building from pattern."""
        it = IdentifierType.objects.create(
            code='test_url',
            name='Test URL',
            url_pattern='https://www.example.org/records/{id}'
        )
        assert it.build_url('12345') == 'https://www.example.org/records/12345'

    def test_unique_code(self) -> None:
        """Test that identifier type codes must be unique."""
        IdentifierType.objects.create(
            code='unique_test',
            name='Test',
            url_pattern='https://example.com/{id}'
        )
        with pytest.raises(IntegrityError):
            IdentifierType.objects.create(
                code='unique_test',
                name='Test 2',
                url_pattern='https://other.com/{id}'
            )

    def test_str_representation(self) -> None:
        """Test string representation."""
        it = IdentifierType.objects.create(
            code='str_test',
            name='String Test',
            url_pattern='https://example.com/{id}'
        )
        assert str(it) == 'String Test (str_test)'


class TestGenerateCollectionId(TestCase):
    """Tests for 9-digit ID generation."""

    def test_id_is_9_digits(self) -> None:
        """Test that generated ID has exactly 9 digits."""
        collection_id = generate_collection_id()
        assert 100000000 <= collection_id <= 999999999

    def test_id_uniqueness(self) -> None:
        """Test that generated IDs are unique across multiple collections."""
        user = User.objects.create_user('testuser_gen', 'test@example.com', 'password')
        ids: set[int] = set()
        for _ in range(50):
            collection = Collection.objects.create(owner=user, name='Test')
            ids.add(collection.collection_id)
        assert len(ids) == 50


class TestCollection(TestCase):
    """Tests for Collection model."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('testuser_coll', 'test@example.com', 'password')

    def test_create_collection(self) -> None:
        """Test creating a collection."""
        collection = Collection.objects.create(
            owner=self.user,
            name='My Research'
        )
        assert collection.name == 'My Research'
        assert collection.owner == self.user
        assert 100000000 <= collection.collection_id <= 999999999

    def test_default_name(self) -> None:
        """Test default collection name."""
        collection = Collection.objects.create(owner=self.user)
        assert collection.name == 'Untitled Collection'

    def test_cascade_delete(self) -> None:
        """Test that deleting user deletes collections."""
        Collection.objects.create(owner=self.user, name='Test')
        assert Collection.objects.filter(owner=self.user).count() == 1
        self.user.delete()
        assert Collection.objects.filter(owner=self.user).count() == 0

    def test_str_representation(self) -> None:
        """Test string representation."""
        collection = Collection.objects.create(owner=self.user, name='Test Collection')
        expected = f"Test Collection ({collection.collection_id})"
        assert str(collection) == expected

    def test_ordering(self) -> None:
        """Test collections are ordered by updated_at descending."""
        c1 = Collection.objects.create(owner=self.user, name='First')
        c2 = Collection.objects.create(owner=self.user, name='Second')
        # Second created collection should appear first
        collections = list(Collection.objects.filter(owner=self.user))
        assert collections[0].name == 'Second'
        assert collections[1].name == 'First'


class TestSearchHistory(TestCase):
    """Tests for SearchHistory model."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('testuser_sh', 'test@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test')

    def test_create_search_history(self) -> None:
        """Test creating a search history entry."""
        search = SearchHistory.objects.create(
            collection=self.collection,
            prompt='red mushroom',
            embedding_name='skol:embedding:v1.1',
            k=5,
            result_references=[{'title': 'Test', 'similarity': 0.9}],
            result_count=1
        )
        assert search.prompt == 'red mushroom'
        assert search.collection == self.collection
        assert search.k == 5
        assert search.result_count == 1

    def test_ordering(self) -> None:
        """Test that searches are ordered by created_at descending."""
        SearchHistory.objects.create(
            collection=self.collection,
            prompt='first',
            embedding_name='test',
            k=3
        )
        SearchHistory.objects.create(
            collection=self.collection,
            prompt='second',
            embedding_name='test',
            k=3
        )
        searches = list(self.collection.search_history.all())
        assert searches[0].prompt == 'second'
        assert searches[1].prompt == 'first'

    def test_cascade_delete_with_collection(self) -> None:
        """Test that deleting collection deletes search history."""
        SearchHistory.objects.create(
            collection=self.collection,
            prompt='test',
            embedding_name='test',
            k=3
        )
        assert SearchHistory.objects.filter(collection=self.collection).count() == 1
        self.collection.delete()
        assert SearchHistory.objects.filter(collection=self.collection).count() == 0

    def test_str_representation_short_prompt(self) -> None:
        """Test string representation with short prompt."""
        search = SearchHistory.objects.create(
            collection=self.collection,
            prompt='short prompt',
            embedding_name='test',
            k=3
        )
        assert 'short prompt' in str(search)

    def test_str_representation_long_prompt(self) -> None:
        """Test string representation with long prompt (truncated)."""
        long_prompt = 'x' * 100
        search = SearchHistory.objects.create(
            collection=self.collection,
            prompt=long_prompt,
            embedding_name='test',
            k=3
        )
        assert '...' in str(search)
        assert len(str(search).split('...')[0].split(':')[1].strip()) <= 50


class TestExternalIdentifier(TestCase):
    """Tests for ExternalIdentifier model."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('testuser_ei', 'test@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test')
        self.identifier_type = IdentifierType.objects.create(
            code='test_type',
            name='Test Type',
            url_pattern='https://www.example.org/records/{id}'
        )

    def test_create_identifier(self) -> None:
        """Test creating an external identifier."""
        identifier = ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        assert identifier.value == '12345'
        assert identifier.url == 'https://www.example.org/records/12345'

    def test_unique_constraint(self) -> None:
        """Test that same identifier cannot be added twice to same collection."""
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        with pytest.raises(IntegrityError):
            ExternalIdentifier.objects.create(
                collection=self.collection,
                identifier_type=self.identifier_type,
                value='12345'
            )

    def test_same_value_different_type_allowed(self) -> None:
        """Test that same value with different type is allowed."""
        another_type = IdentifierType.objects.create(
            code='another_type',
            name='Another Type',
            url_pattern='https://other.com/{id}'
        )
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        # Should not raise
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=another_type,
            value='12345'
        )
        assert ExternalIdentifier.objects.filter(collection=self.collection).count() == 2

    def test_cascade_delete_with_collection(self) -> None:
        """Test that deleting collection deletes identifiers."""
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        assert ExternalIdentifier.objects.filter(collection=self.collection).count() == 1
        self.collection.delete()
        assert ExternalIdentifier.objects.filter(collection=self.collection).count() == 0

    def test_protect_identifier_type(self) -> None:
        """Test that identifier type cannot be deleted if in use."""
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        from django.db.models import ProtectedError
        with pytest.raises(ProtectedError):
            self.identifier_type.delete()

    def test_str_representation(self) -> None:
        """Test string representation."""
        identifier = ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.identifier_type,
            value='12345'
        )
        assert str(identifier) == 'test_type: 12345'


class TestSeededIdentifierTypes(TestCase):
    """Tests for seeded identifier types from migration."""

    def test_inat_exists(self) -> None:
        """Test iNaturalist identifier type was seeded."""
        it = IdentifierType.objects.get(code='inat')
        assert it.name == 'iNaturalist'
        assert '{id}' in it.url_pattern

    def test_mo_exists(self) -> None:
        """Test Mushroom Observer identifier type was seeded."""
        it = IdentifierType.objects.get(code='mo')
        assert it.name == 'Mushroom Observer'

    def test_genbank_exists(self) -> None:
        """Test GenBank identifier type was seeded."""
        it = IdentifierType.objects.get(code='genbank')
        assert it.name == 'GenBank'

    def test_all_seeded_types(self) -> None:
        """Test all expected identifier types exist."""
        codes = ['inat', 'mo', 'genbank', 'mycobank', 'indexfungorum']
        for code in codes:
            assert IdentifierType.objects.filter(code=code).exists(), f"Missing: {code}"
