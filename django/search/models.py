"""
Database models for the search app.

Includes models for user collections, search history, and external identifiers.
"""
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
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


class SearchHistory(models.Model):
    """
    Records each search performed within a collection.

    Stores the search prompt, timestamp, and serialized result references
    for later review. Results are stored as JSON references to allow
    retrieval without duplicating CouchDB data.
    """
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='search_history'
    )
    prompt = models.TextField()
    embedding_name = models.CharField(max_length=255)
    k = models.PositiveIntegerField(default=3)
    # Store result references as JSON (not full results, to save space)
    # Format: [{"similarity": 0.95, "taxa_id": "...", "title": "..."}, ...]
    result_references = models.JSONField(default=list)
    result_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Search History Entry"
        verbose_name_plural = "Search History Entries"
        indexes = [
            models.Index(fields=['collection', '-created_at']),
        ]

    def __str__(self) -> str:
        prompt_preview = f"{self.prompt[:50]}..." if len(self.prompt) > 50 else self.prompt
        return f"Search: {prompt_preview} ({self.created_at})"


class ExternalIdentifier(models.Model):
    """
    External system identifiers associated with a collection.

    Allows users to link their collection to observations or records
    in external systems like iNaturalist, Mushroom Observer, or GenBank.
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
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['identifier_type__name', 'value']
        # Prevent duplicate identifiers of same type in same collection
        unique_together = ['collection', 'identifier_type', 'value']

    def __str__(self) -> str:
        return f"{self.identifier_type.code}: {self.value}"

    @property
    def url(self) -> str:
        """Build the full URL for this identifier."""
        return self.identifier_type.build_url(self.value)
