"""
Database models for the search app.

Includes models for user collections, search history, and external identifiers.
"""
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from datetime import timedelta
import random
from typing import Optional, List


class IdentifierType(models.Model):
    """
    Configurable external identifier types (MO, iNat, GenBank, etc.).

    Administrators can define URL patterns for external systems.
    The {id} placeholder in url_pattern will be replaced with the actual identifier.
    """
    code = models.CharField(max_length=20, unique=True)  # e.g., "inat", "mo", "genbank"
    name = models.CharField(max_length=100)  # e.g., "iNaturalist"
    url_pattern = models.CharField(
        max_length=500,
        help_text="URL pattern with {id} placeholder, e.g., https://www.inaturalist.org/observations/{id}"
    )
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        verbose_name = "Identifier Type"
        verbose_name_plural = "Identifier Types"

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"

    def build_url(self, identifier_value: str) -> str:
        """Build the full URL for a given identifier value."""
        return self.url_pattern.replace("{id}", identifier_value)


def generate_collection_id() -> int:
    """Generate a unique 9-digit collection ID."""
    # Import here to avoid circular imports during migrations
    while True:
        # Generate random 9-digit number (100000000 to 999999999)
        collection_id = random.randint(100000000, 999999999)
        if not Collection.objects.filter(collection_id=collection_id).exists():
            return collection_id


class Collection(models.Model):
    """
    A user's collection of search research.

    Collections have a unique 9-digit ID for easy reference/sharing,
    belong to a user, and can be named for organization.

    Fields:
        - name: User-friendly collection name
        - description: The working taxonomic description being searched/refined
        - notes: Collection-level notes/metadata (what this collection is about)
    """
    collection_id = models.BigIntegerField(
        unique=True,
        validators=[
            MinValueValidator(100000000),
            MaxValueValidator(999999999)
        ],
        help_text="Unique 9-digit collection identifier"
    )
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='collections'
    )
    name = models.CharField(max_length=255, default="Untitled Collection")
    description = models.TextField(
        blank=True,
        default="",
        help_text="Working taxonomic description text for searching"
    )
    notes = models.TextField(
        blank=True,
        default="",
        help_text="Collection-level notes/metadata"
    )
    nomenclature = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Best guess taxon name for this collection"
    )
    embargo_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Collection is private until this date (null=public)"
    )
    flagged_by = models.JSONField(
        default=list,
        blank=True,
        help_text="List of user IDs who flagged this collection as inappropriate"
    )
    hidden = models.BooleanField(
        default=False,
        help_text="Admin-hidden collection: excluded from indexing, read-only for owner"
    )
    hidden_by = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='hidden_collections',
        help_text="Admin who hid this collection"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['collection_id']),
            models.Index(fields=['owner', '-updated_at']),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.collection_id})"

    def save(self, *args, **kwargs) -> None:
        """Override save to generate collection_id if not set."""
        if not self.collection_id:
            self.collection_id = generate_collection_id()
        super().save(*args, **kwargs)

    @property
    def is_embargoed(self) -> bool:
        """Check if collection is currently embargoed (private)."""
        if self.embargo_until is None:
            return False
        return self.embargo_until > timezone.now()

    @property
    def is_public(self) -> bool:
        """Check if collection is publicly visible."""
        return not self.is_embargoed


class SearchHistory(models.Model):
    """
    Records events within a collection: searches and nomenclature changes.

    Stores the search prompt, timestamp, and serialized result references
    for later review. Results are stored as JSON references to allow
    retrieval without duplicating CouchDB data.

    For nomenclature changes, stores the new nomenclature value.
    """
    EVENT_TYPE_CHOICES = [
        ('search', 'Search'),
        ('nomenclature_change', 'Nomenclature Change'),
    ]

    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='search_history'
    )
    event_type = models.CharField(
        max_length=30,
        choices=EVENT_TYPE_CHOICES,
        default='search',
        help_text="Type of history event"
    )
    # Search-specific fields (nullable for non-search events)
    prompt = models.TextField(blank=True, default='')
    embedding_name = models.CharField(max_length=255, blank=True, default='')
    k = models.PositiveIntegerField(default=3)
    # Store result references as JSON (not full results, to save space)
    # Format: [{"similarity": 0.95, "taxa_id": "...", "title": "..."}, ...]
    result_references = models.JSONField(default=list)
    result_count = models.PositiveIntegerField(default=0)
    # Nomenclature change field
    nomenclature = models.CharField(
        max_length=500,
        blank=True,
        default='',
        help_text="Nomenclature value at time of change (for nomenclature_change events)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Search History Entry"
        verbose_name_plural = "Search History Entries"
        indexes = [
            models.Index(fields=['collection', '-created_at']),
        ]

    def __str__(self) -> str:
        if self.event_type == 'nomenclature_change':
            return f"Nomenclature: {self.nomenclature[:50]} ({self.created_at})"
        prompt_preview = f"{self.prompt[:50]}..." if len(self.prompt) > 50 else self.prompt
        return f"Search: {prompt_preview} ({self.created_at})"


class ExternalIdentifier(models.Model):
    """
    External system identifiers associated with a collection.

    Allows users to link their collection to observations or records
    in external systems like iNaturalist, Mushroom Observer, or GenBank.

    For Fungarium identifiers, the fungarium_code field stores the Index
    Herbariorum code (e.g., 'NY', 'K', 'BPI') and value stores the accession number.
    """
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='external_identifiers'
    )
    identifier_type = models.ForeignKey(
        IdentifierType,
        on_delete=models.PROTECT,
        related_name='identifiers'
    )
    value = models.CharField(max_length=255)  # The actual ID value
    fungarium_code = models.CharField(
        max_length=20,
        blank=True,
        default='',
        help_text='Index Herbariorum code for fungarium identifiers'
    )
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['identifier_type__name', 'value']
        # Prevent duplicate identifiers of same type in same collection
        unique_together = ['collection', 'identifier_type', 'value', 'fungarium_code']

    def __str__(self) -> str:
        if self.fungarium_code:
            return f"{self.fungarium_code}: {self.value}"
        return f"{self.identifier_type.code}: {self.value}"

    @property
    def url(self) -> str:
        """Build the full URL for this identifier.

        For fungarium identifiers, URL is built from Redis data.
        Returns empty string if URL cannot be determined.
        """
        if self.identifier_type.code == 'fungarium' and self.fungarium_code:
            return self._build_fungarium_url()
        return self.identifier_type.build_url(self.value)

    def _build_fungarium_url(self) -> str:
        """Build URL for fungarium identifier from Redis data."""
        import json
        from .utils import get_redis_client

        try:
            r = get_redis_client(decode_responses=True)
            raw = r.get('skol:fungaria')
            if not raw:
                return ''

            data = json.loads(raw)
            institutions = data.get('institutions', {})
            inst = institutions.get(self.fungarium_code)

            if not inst:
                return ''

            contact = inst.get('contact', {})
            if isinstance(contact, dict):
                # Check for collectionUrl (f-string with {id})
                collection_url = contact.get('collectionUrl', '')
                if collection_url and '{id}' in collection_url:
                    return collection_url.replace('{id}', self.value)

                # Fall back to webUrl (no substitution)
                web_url = contact.get('webUrl', '')
                if web_url:
                    return web_url

            return ''
        except Exception:
            return ''


class UserSettings(models.Model):
    """
    User-specific settings for search and collection behavior.

    Settings include:
    - Default embargo period for new collections
    - Search preferences (embedding model, result count)
    - Feature display settings
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='settings'
    )

    # Embargo settings
    default_embargo_days = models.PositiveIntegerField(
        default=0,
        help_text="Default embargo period in days (0=public immediately)"
    )

    # Search settings
    default_embedding = models.CharField(
        max_length=255,
        blank=True,
        default='',
        help_text="Preferred embedding model for search"
    )
    default_k = models.PositiveIntegerField(
        default=3,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="Default number of search results"
    )

    # Feature settings
    feature_taxa_count = models.PositiveIntegerField(
        default=6,
        validators=[MinValueValidator(2), MaxValueValidator(50)],
        help_text="Number of taxa to retrieve for feature lists"
    )
    feature_max_tree_depth = models.PositiveIntegerField(
        default=10,
        validators=[MinValueValidator(1), MaxValueValidator(20)],
        help_text="Maximum depth for feature tree display"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Settings"
        verbose_name_plural = "User Settings"

    def __str__(self) -> str:
        return f"Settings for {self.user.username}"

    def get_embargo_date(self):
        """Calculate embargo date based on default_embargo_days."""
        if self.default_embargo_days == 0:
            return None
        return timezone.now() + timedelta(days=self.default_embargo_days)
