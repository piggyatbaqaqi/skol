"""
Database models for the search app.

Includes models for user collections, search history, external identifiers,
and projects (named groups of collections).
"""
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from datetime import timedelta
import random
import re
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
        default=20,
        validators=[MinValueValidator(1), MaxValueValidator(100)],
        help_text="Default number of search results"
    )
    results_per_page = models.PositiveIntegerField(
        default=10,
        validators=[MinValueValidator(5), MaxValueValidator(50)],
        help_text="Number of results to display per page"
    )
    nomenclature_limit = models.PositiveIntegerField(
        default=20,
        validators=[MinValueValidator(1), MaxValueValidator(200)],
        help_text="Maximum results for nomenclature pattern search"
    )

    # Feature settings
    feature_taxa_count = models.PositiveIntegerField(
        default=6,
        validators=[MinValueValidator(2), MaxValueValidator(50)],
        help_text="Number of taxa to retrieve for feature lists"
    )
    feature_top_n = models.PositiveIntegerField(
        default=30,
        validators=[MinValueValidator(5), MaxValueValidator(100)],
        help_text="Number of top features to return from the classifier"
    )
    feature_max_tree_depth = models.PositiveIntegerField(
        default=10,
        validators=[MinValueValidator(1), MaxValueValidator(50)],
        help_text="Maximum depth for feature tree display"
    )
    feature_min_df = models.PositiveIntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="Minimum document frequency for feature terms"
    )
    feature_max_df = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.1), MaxValueValidator(1.0)],
        help_text="Maximum document frequency fraction for feature terms"
    )

    # Experiment settings
    default_experiment = models.CharField(
        max_length=100,
        default='production',
        help_text="Active experiment name from skol_experiments"
    )

    # Email preferences
    receive_admin_summary = models.BooleanField(
        default=False,
        help_text="Receive daily admin summary email"
    )

    # Project defaults: new collections are automatically added to these projects.
    # Applies only to collections created *after* this setting is saved.
    default_projects = models.ManyToManyField(
        'Project',
        blank=True,
        related_name='default_for_users',
        help_text=(
            "Projects that new collections are automatically added to. "
            "Does not apply retroactively to existing collections."
        ),
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


class MeasurementUnit(models.Model):
    """
    Admin-configurable unit of measurement for the MetricsWidget.

    Controls which units appear in the unit dropdown. Each measurement set
    records its own unit; changing the unit does not rescale values.
    """
    symbol = models.CharField(
        max_length=20,
        unique=True,
        help_text='Display symbol (e.g., µm, mm, nm)'
    )
    sort_order = models.IntegerField(
        default=0,
        help_text='Display order in the dropdown (ascending)'
    )

    class Meta:
        ordering = ['sort_order', 'symbol']
        verbose_name = "Measurement Unit"
        verbose_name_plural = "Measurement Units"

    def __str__(self) -> str:
        return self.symbol


class MeasurementSet(models.Model):
    """
    A set of measurements for a specific feature of a collection.

    Canonical use: spore dimensions (length x width in µm).
    Stores raw measurements as JSON for client-side statistics computation.
    Each collection can have multiple measurement sets keyed by feature name
    (e.g., "spores", "basidia", "cystidia").
    """
    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='measurement_sets'
    )
    feature = models.CharField(
        max_length=100,
        default='spores',
        help_text='Feature name (e.g., spores, basidia, cystidia)'
    )
    is_2d = models.BooleanField(
        default=True,
        help_text='True for 2D measurements (length x width), False for 1D (length only)'
    )
    report_q = models.BooleanField(
        default=True,
        help_text='Whether to include Q (length/width ratio) in the formatted output'
    )
    unit = models.CharField(
        max_length=20,
        default='\u00b5m',
        help_text='Unit of measurement (e.g., µm, mm, nm)'
    )
    measurements = models.JSONField(
        default=list,
        help_text='Raw measurements: [{"length": 8.5, "width": 6.5}, ...]'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['collection', 'feature']
        ordering = ['feature']
        verbose_name = "Measurement Set"
        verbose_name_plural = "Measurement Sets"

    def __str__(self) -> str:
        n = len(self.measurements) if isinstance(self.measurements, list) else 0
        return f"{self.feature} ({n} samples) - {self.collection}"


# ===========================================================================
# Projects
# ===========================================================================

def _generate_slug(name: str) -> str:
    """Convert a project name to a URL-safe slug.

    Lowercases, collapses non-alphanumeric runs to ``-``, strips leading/
    trailing ``-``.  Raises ``ValueError`` if the result contains no
    alphanumeric character.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not any(c.isalnum() for c in slug):
        raise ValueError(
            f"Project name {name!r} produces an empty slug. "
            "The name must contain at least one alphanumeric character."
        )
    return slug


class Project(models.Model):
    """A named group of collections (e.g. all collections for a field guide).

    Governance rules
    ----------------
    * Anyone can create a project.
    * Anyone can add or remove any collection from any project.
    * All add/remove events are audited in CollectionProject /
      CollectionProjectRemoval.
    * The creator is permanently recorded but has no special operational powers.
    * Only admins can delete a project.
    * All projects are public.

    Namespacing
    -----------
    Slugs are unique per creator (``unique_together = ['creator', 'slug']``).
    Two different users may each have a project whose slug is
    ``french-guiana-fungi``.  In URLs the project is identified by
    ``username/slug``, e.g. ``?project=jsmith/french-guiana-fungi``.
    """

    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, allow_unicode=False)
    creator = models.ForeignKey(
        User,
        on_delete=models.PROTECT,
        related_name='created_projects',
    )
    description = models.TextField(blank=True, default='')
    notes = models.TextField(blank=True, default='')
    collections = models.ManyToManyField(
        'Collection',
        through='CollectionProject',
        related_name='projects',
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['creator', 'slug']]
        ordering = ['creator__username', 'slug']
        verbose_name = 'Project'
        verbose_name_plural = 'Projects'

    def __str__(self) -> str:
        return f"{self.creator.username}/{self.slug}"

    @staticmethod
    def generate_slug(name: str) -> str:
        """Public wrapper around ``_generate_slug``; raises ValueError on bad names."""
        return _generate_slug(name)

    def save(self, *args, **kwargs) -> None:
        """Auto-generate slug from name if not provided; resolve collisions."""
        if not self.slug:
            base = _generate_slug(self.name)
            slug = base
            counter = 2
            qs = Project.objects.filter(creator=self.creator)
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            while qs.filter(slug=slug).exists():
                slug = f"{base}-{counter}"
                counter += 1
            self.slug = slug
        super().save(*args, **kwargs)


class CollectionProject(models.Model):
    """Through-table linking a Collection to a Project.

    Records who added the collection and when.  Removed memberships are
    archived in ``CollectionProjectRemoval`` before the row is deleted.
    """

    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='collection_projects',
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='collection_projects',
    )
    added_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='project_additions',
    )
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['collection', 'project']]
        ordering = ['-added_at']
        verbose_name = 'Collection–Project membership'
        verbose_name_plural = 'Collection–Project memberships'

    def __str__(self) -> str:
        return f"{self.collection} in {self.project}"


class ProjectNotesLog(models.Model):
    """Audit log of changes to a project's notes field.

    A row is created every time the ``notes`` field changes value via the PATCH
    API.  The diff is stored in unified-diff format (output of
    ``difflib.unified_diff``).
    """

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='notes_log',
    )
    changed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='project_notes_changes',
    )
    changed_at = models.DateTimeField(auto_now_add=True)
    diff = models.TextField(
        help_text='Unified diff of the notes change (old → new).'
    )

    class Meta:
        ordering = ['-changed_at']
        verbose_name = 'Project notes log entry'
        verbose_name_plural = 'Project notes log entries'

    def __str__(self) -> str:
        return (
            f"{self.project} notes changed by {self.changed_by} "
            f"at {self.changed_at}"
        )


class CollectionProjectRemoval(models.Model):
    """Permanent audit log of collection removals from projects.

    A row is created here *before* the corresponding CollectionProject row is
    deleted, capturing the original ``added_by`` and ``added_at`` values.
    """

    collection = models.ForeignKey(
        Collection,
        on_delete=models.CASCADE,
        related_name='project_removals',
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='collection_removals',
    )
    removed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='project_removals',
    )
    removed_at = models.DateTimeField(auto_now_add=True)
    original_added_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='project_removals_as_adder',
    )
    original_added_at = models.DateTimeField()

    class Meta:
        ordering = ['-removed_at']
        verbose_name = 'Collection–Project removal log'
        verbose_name_plural = 'Collection–Project removal logs'

    def __str__(self) -> str:
        return (
            f"{self.collection} removed from {self.project} "
            f"by {self.removed_by} at {self.removed_at}"
        )
