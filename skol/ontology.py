"""
Ontology Integration for Structured Feature Extraction

This module provides ontology-based context injection for guiding vocabulary
selection in taxonomic feature extraction. It implements the approach described
in docs/slm_optimization_discussion.md Phase 2.

Key components:
- OntologyTerm: Data class for individual ontology terms
- OntologyIndex: Searchable index for a single ontology
- OntologyRegistry: Central registry for multiple ontologies
- OntologyContextBuilder: Builds prompt context from ontology subgraphs
- OntologyGuidedGenerator: Integrates ontology context with constrained generation

Supported ontologies:
- PATO (Phenotype And Trait Ontology) - for quality properties
- FAO (Fungal Anatomy Ontology) - for anatomical structures

Example:
    >>> registry = OntologyRegistry()
    >>> registry.register("pato", "pato.obo", category="base")
    >>> registry.register("fao", "fao.obo", category="base")
    >>> builder = OntologyContextBuilder(registry)
    >>> context = builder.build_context("Pileus convex, 3-5 cm, surface dry")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class OntologyTerm:
    """
    Represents a single term from an ontology.

    Attributes:
        id: Unique identifier in the ontology (e.g., "PATO:0000001")
        name: Human-readable term name
        definition: Optional definition text
        depth: Distance from root in the ontology hierarchy
        ancestors: List of ancestor term names (from immediate parent to root)
        embedding: Vector representation for semantic search
    """
    id: str
    name: str
    definition: Optional[str]
    depth: int
    ancestors: List[str]
    embedding: np.ndarray = field(repr=False)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, OntologyTerm):
            return self.id == other.id
        return False


class OntologyIndex:
    """
    Searchable index for a single ontology.

    Loads an ontology file (OBO format) and creates embeddings for semantic search.
    Terms can be searched by similarity to a query string or filtered by depth.

    Attributes:
        name: Identifier for this ontology index
        terms: List of all indexed terms
        term_embeddings: Matrix of term embeddings for fast similarity search
        category: "base" (always used) or "specialized" (selected dynamically)
    """

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        name: str = "unnamed",
        encoder: Optional[Any] = None,
    ):
        """
        Initialize an ontology index.

        Args:
            ontology_path: Path to .obo file (None for lazy loading)
            name: Identifier for this index
            encoder: SentenceTransformer model (shared across indices)
        """
        self.name = name
        self._encoder = encoder
        self.terms: List[OntologyTerm] = []
        self.term_embeddings: Optional[np.ndarray] = None
        self.category: str = "general"
        self._term_lookup: Dict[str, OntologyTerm] = {}

        if ontology_path:
            self._load_ontology(ontology_path)

    @property
    def encoder(self):
        """Get or lazily load the encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._encoder

    def _load_ontology(self, path: str) -> None:
        """
        Load and index an ontology from an OBO file.

        Args:
            path: Path to the .obo file
        """
        terms_data = []

        # Try pronto first, fall back to simple parser if it fails
        try:
            terms_data = self._load_with_pronto(path)
        except Exception:
            # Fall back to simple OBO parser for problematic files
            terms_data = self._load_with_simple_parser(path)

        self._index_terms(terms_data)

    def _load_with_pronto(self, path: str) -> List[Dict[str, Any]]:
        """Load ontology using pronto library."""
        try:
            from pronto import Ontology
        except ImportError:
            raise ImportError(
                "pronto library required for OBO parsing. "
                "Install with: pip install pronto"
            )

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ont = Ontology(path)

        terms_data = []
        for term in ont.terms():
            if hasattr(term, 'obsolete') and term.obsolete:
                continue

            try:
                ancestors = [a.name for a in term.superclasses(with_self=False)]
            except Exception:
                ancestors = []

            definition = str(term.definition) if term.definition else ""
            terms_data.append({
                "id": term.id,
                "name": term.name,
                "definition": definition if definition else None,
                "depth": len(ancestors),
                "ancestors": ancestors,
                "text": f"{term.name}: {definition}"
            })

        return terms_data

    def _load_with_simple_parser(self, path: str) -> List[Dict[str, Any]]:
        """Simple fallback OBO parser for files pronto can't handle."""
        terms_data = []
        current_term: Optional[Dict[str, Any]] = None

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line == '[Term]':
                    if current_term and current_term.get('id'):
                        terms_data.append(current_term)
                    current_term = {
                        'id': '', 'name': '', 'definition': None,
                        'depth': 0, 'ancestors': [], 'is_obsolete': False
                    }
                elif line.startswith('[') and line.endswith(']'):
                    # Other stanza types
                    if current_term and current_term.get('id'):
                        terms_data.append(current_term)
                    current_term = None
                elif current_term is not None:
                    if line.startswith('id:'):
                        current_term['id'] = line[3:].strip()
                    elif line.startswith('name:'):
                        current_term['name'] = line[5:].strip()
                    elif line.startswith('def:'):
                        # Extract definition from quotes
                        match_start = line.find('"')
                        match_end = line.rfind('"')
                        if match_start != -1 and match_end > match_start:
                            current_term['definition'] = \
                                line[match_start + 1:match_end]
                    elif line.startswith('is_obsolete: true'):
                        current_term['is_obsolete'] = True

            # Don't forget the last term
            if current_term and current_term.get('id'):
                terms_data.append(current_term)

        # Filter obsolete and build text
        result = []
        for term in terms_data:
            if term.get('is_obsolete'):
                continue
            result.append({
                'id': term['id'],
                'name': term['name'],
                'definition': term['definition'],
                'depth': 0,  # Simple parser doesn't compute depth
                'ancestors': [],
                'text': f"{term['name']}: {term['definition'] or ''}"
            })

        return result

    def _index_terms(self, terms_data: List[Dict[str, Any]]) -> None:
        """Index the loaded terms with embeddings."""

        if not terms_data:
            return

        # Batch encode for efficiency
        texts = [t["text"] for t in terms_data]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        for term_data, embedding in zip(terms_data, embeddings):
            term = OntologyTerm(
                id=term_data["id"],
                name=term_data["name"],
                definition=term_data["definition"],
                depth=term_data["depth"],
                ancestors=term_data["ancestors"],
                embedding=embedding
            )
            self.terms.append(term)
            self._term_lookup[term.id] = term
            self._term_lookup[term.name.lower()] = term

        self.term_embeddings = np.stack([t.embedding for t in self.terms])

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[OntologyTerm, float]]:
        """
        Find terms most similar to query.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (term, similarity_score) tuples, sorted by similarity
        """
        if not self.terms or self.term_embeddings is None:
            return []

        query_embedding = self.encoder.encode(query)

        # Compute cosine similarities
        norms = np.linalg.norm(self.term_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        # Avoid division by zero
        valid_mask = (norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(self.terms))
        similarities[valid_mask] = (
            np.dot(self.term_embeddings[valid_mask], query_embedding) /
            (norms[valid_mask] * query_norm)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(self.terms[i], float(similarities[i])) for i in top_indices]

    def get_terms_at_depth(
        self,
        min_depth: int,
        max_depth: int
    ) -> List[OntologyTerm]:
        """
        Get all terms within a depth range.

        Args:
            min_depth: Minimum depth (0 = root)
            max_depth: Maximum depth

        Returns:
            List of terms within the depth range
        """
        return [t for t in self.terms if min_depth <= t.depth <= max_depth]

    def get_term(self, identifier: str) -> Optional[OntologyTerm]:
        """
        Get a specific term by ID or name.

        Args:
            identifier: Term ID (e.g., "PATO:0000001") or name

        Returns:
            OntologyTerm if found, None otherwise
        """
        return self._term_lookup.get(identifier) or \
               self._term_lookup.get(identifier.lower())

    def __len__(self) -> int:
        return len(self.terms)


class OntologyRegistry:
    """
    Central registry for all ontologies.

    Provides a unified interface for managing multiple ontologies,
    designed for easy addition of new ontologies.

    Example:
        >>> registry = OntologyRegistry()
        >>> registry.register("pato", "pato.obo", category="base")
        >>> registry.register("fao", "fao.obo", category="base")
        >>> pato = registry.get("pato")
        >>> results = pato.search("round shape")
    """

    def __init__(self):
        self.ontologies: Dict[str, OntologyIndex] = {}
        self._encoder = None

    @property
    def encoder(self):
        """Shared encoder for all ontologies."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
        return self._encoder

    def register(
        self,
        name: str,
        path: str,
        category: str = "general"
    ) -> OntologyIndex:
        """
        Register a new ontology.

        Args:
            name: Unique identifier (e.g., "pato", "fao", "ascomycete_anatomy")
            path: Path to .obo file
            category: "base" (always used) or "specialized" (selected dynamically)

        Returns:
            The created OntologyIndex
        """
        index = OntologyIndex(
            ontology_path=path,
            name=name,
            encoder=self.encoder
        )
        index.category = category
        self.ontologies[name] = index

        return index

    def register_index(self, name: str, index: OntologyIndex) -> None:
        """
        Register a pre-built index.

        Args:
            name: Unique identifier
            index: Pre-built OntologyIndex
        """
        self.ontologies[name] = index

    def get(self, name: str) -> OntologyIndex:
        """
        Get an ontology by name.

        Args:
            name: Ontology identifier

        Returns:
            OntologyIndex

        Raises:
            KeyError: If ontology not found
        """
        if name not in self.ontologies:
            raise KeyError(f"Ontology '{name}' not registered")
        return self.ontologies[name]

    def get_base_ontologies(self) -> List[OntologyIndex]:
        """Get all base ontologies (always used)."""
        return [o for o in self.ontologies.values() if o.category == "base"]

    def get_specialized_ontologies(self) -> List[OntologyIndex]:
        """Get all specialized ontologies (selected dynamically)."""
        return [o for o in self.ontologies.values() if o.category == "specialized"]

    def list_registered(self) -> List[Dict[str, Any]]:
        """
        List all registered ontologies.

        Returns:
            List of dicts with name, category, and term_count
        """
        return [
            {"name": name, "category": o.category, "term_count": len(o.terms)}
            for name, o in self.ontologies.items()
        ]

    def search_all(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, List[Tuple[OntologyTerm, float]]]:
        """
        Search all registered ontologies.

        Args:
            query: Search query
            top_k: Results per ontology

        Returns:
            Dict mapping ontology name to search results
        """
        results = {}
        for name, index in self.ontologies.items():
            results[name] = index.search(query, top_k=top_k)
        return results

    def __contains__(self, name: str) -> bool:
        return name in self.ontologies


class OntologyContextBuilder:
    """
    Build prompt context from ontology subgraphs.

    Retrieves relevant ontology terms for a description and formats them
    as context for prompt injection in constrained generation.

    Example:
        >>> builder = OntologyContextBuilder(registry)
        >>> context = builder.build_context(
        ...     "Pileus convex, 3-5 cm diameter, surface dry and brown"
        ... )
    """

    def __init__(self, registry: OntologyRegistry):
        """
        Initialize the context builder.

        Args:
            registry: OntologyRegistry with registered ontologies
        """
        self.registry = registry

    def build_context(
        self,
        description: str,
        anatomy_ontology: str = "fao",
        quality_ontology: str = "pato",
        top_k_per_ontology: int = 15,
        max_context_chars: int = 2000
    ) -> str:
        """
        Build linearized ontology context for a description.

        Args:
            description: Taxonomic description text
            anatomy_ontology: Name of anatomy ontology to use
            quality_ontology: Name of quality ontology to use
            top_k_per_ontology: Maximum terms to retrieve per ontology
            max_context_chars: Maximum characters in output

        Returns:
            Formatted string for prompt injection
        """
        lines = []

        # Retrieve from anatomy ontology if available
        if anatomy_ontology in self.registry:
            anatomy_index = self.registry.get(anatomy_ontology)
            anatomy_results = anatomy_index.search(
                description,
                top_k=top_k_per_ontology
            )

            lines.append("ANATOMICAL STRUCTURES (use for top-level feature keys):")
            lines.extend(self._format_hierarchy(anatomy_results))
            lines.append("")

        # Retrieve from quality ontology if available
        if quality_ontology in self.registry:
            quality_index = self.registry.get(quality_ontology)
            quality_results = quality_index.search(
                description,
                top_k=top_k_per_ontology
            )

            lines.append("QUALITY PROPERTIES (use for nested property keys):")
            lines.extend(self._format_hierarchy(quality_results))
            lines.append("")

        # Add hierarchy path examples
        if anatomy_ontology in self.registry or quality_ontology in self.registry:
            lines.append("HIERARCHY EXAMPLES:")
            if anatomy_ontology in self.registry:
                lines.extend(self._format_path_examples(anatomy_results[:5]))
            if quality_ontology in self.registry:
                lines.extend(self._format_path_examples(quality_results[:5]))

        context = "\n".join(lines)

        # Truncate if needed
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[truncated]"

        return context

    def _format_hierarchy(
        self,
        results: List[Tuple[OntologyTerm, float]]
    ) -> List[str]:
        """
        Format terms grouped by depth level.

        Args:
            results: List of (term, score) tuples

        Returns:
            Formatted lines grouped by depth
        """
        by_depth: Dict[int, List[str]] = {}
        for term, score in results:
            depth = min(term.depth, 4)  # Cap display depth
            if depth not in by_depth:
                by_depth[depth] = []
            if term.name not in by_depth[depth]:  # Avoid duplicates
                by_depth[depth].append(term.name)

        lines = []
        for depth in sorted(by_depth.keys()):
            terms = by_depth[depth][:8]  # Limit per level
            lines.append(f"  [L{depth + 1}] {' | '.join(terms)}")

        return lines

    def _format_path_examples(
        self,
        results: List[Tuple[OntologyTerm, float]]
    ) -> List[str]:
        """
        Show full hierarchy paths for context.

        Args:
            results: List of (term, score) tuples

        Returns:
            Formatted path strings
        """
        lines = []
        seen_paths: set = set()

        for term, score in results:
            if term.ancestors:
                # Show path from near-root to term
                path_parts = term.ancestors[-3:] + [term.name]
                path_str = " > ".join(path_parts)

                if path_str not in seen_paths:
                    seen_paths.add(path_str)
                    lines.append(f"  {path_str}")

        return lines[:5]  # Limit examples

    def build_minimal_context(
        self,
        description: str,
        top_k: int = 10
    ) -> str:
        """
        Build minimal context with just term lists (no hierarchy).

        Useful for smaller context windows.

        Args:
            description: Taxonomic description
            top_k: Number of terms per ontology

        Returns:
            Compact context string
        """
        all_terms: List[str] = []

        for index in self.registry.get_base_ontologies():
            results = index.search(description, top_k=top_k)
            terms = [term.name for term, _ in results]
            all_terms.extend(terms)

        # Deduplicate while preserving order
        seen: set = set()
        unique_terms = []
        for term in all_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return "Preferred vocabulary: " + ", ".join(unique_terms[:20])


class OntologyGuidedGenerator:
    """
    Generate structured JSON with ontology-guided vocabulary.

    Combines ontology context injection with constrained JSON generation
    to produce outputs that use standardized terminology.

    Example:
        >>> from skol.constrained_decoder import ConstrainedDecoder, TaxonomySchema
        >>> schema = TaxonomySchema(max_depth=4)
        >>> decoder = ConstrainedDecoder(backend="mock")
        >>> generator = OntologyGuidedGenerator(decoder, registry, schema)
        >>> result = generator.generate("Pileus convex, brown, 5cm diameter")
    """
    # TODO(piggy): More than one level of anatomical terms may be appropriate.
    DEFAULT_PROMPT_TEMPLATE = """Extract structured features from a biological species description.

{ontology_context}

RULES:
1. Use anatomical terms for top-level keys (Level 1)
2. Use quality types for Level 2 keys (shape, color, size, texture, etc.)
3. Use quality subtypes for Level 3 keys (specific aspects of qualities)
4. Level 4+ should contain arrays of observed values
5. Follow the hierarchy patterns shown above
6. Use consistent terminology from the provided vocabularies


DESCRIPTION:
{description}

OUTPUT (valid JSON only):
"""

    def __init__(
        self,
        decoder: Any,
        registry: OntologyRegistry,
        schema: Any,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize the ontology-guided generator.

        Args:
            decoder: ConstrainedDecoder instance (or compatible)
            registry: OntologyRegistry with registered ontologies
            schema: TaxonomySchema for JSON structure
            prompt_template: Custom prompt template (uses default if None)
        """
        self.decoder = decoder
        self.context_builder = OntologyContextBuilder(registry)
        self.schema = schema
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def generate(
        self,
        description: str,
        anatomy_ontology: str = "fao",
        quality_ontology: str = "pato",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate ontology-guided structured output.

        Args:
            description: Taxonomic description text
            anatomy_ontology: Name of anatomy ontology
            quality_ontology: Name of quality ontology
            **kwargs: Additional arguments passed to decoder

        Returns:
            Structured feature dictionary
        """
        # Build ontology context
        ontology_context = self.context_builder.build_context(
            description,
            anatomy_ontology=anatomy_ontology,
            quality_ontology=quality_ontology
        )

        # Construct prompt
        prompt = self.prompt_template.format(
            ontology_context=ontology_context,
            description=description
        )

        # Generate with constrained decoder
        # The decoder handles schema enforcement
        result = self.decoder.extract_features(prompt, **kwargs)

        return result

    def generate_with_context(
        self,
        description: str,
        context_override: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate and return both result and the context used.

        Useful for debugging and analysis.

        Args:
            description: Taxonomic description
            context_override: Use this context instead of building one
            **kwargs: Additional arguments

        Returns:
            Tuple of (result_dict, context_string)
        """
        if context_override:
            context = context_override
        else:
            context = self.context_builder.build_context(description)

        prompt = self.prompt_template.format(
            ontology_context=context,
            description=description
        )

        result = self.decoder.extract_features(prompt, **kwargs)

        return result, context


# Convenience functions for common setups

def create_default_registry(
    pato_path: Optional[str] = None,
    fao_path: Optional[str] = None
) -> OntologyRegistry:
    """
    Create a registry with default PATO and FAO ontologies.

    Args:
        pato_path: Path to pato.obo (or None to skip)
        fao_path: Path to fao.obo (or None to skip)

    Returns:
        Configured OntologyRegistry
    """
    registry = OntologyRegistry()

    if pato_path and Path(pato_path).exists():
        registry.register("pato", pato_path, category="base")

    if fao_path and Path(fao_path).exists():
        registry.register("fao", fao_path, category="base")

    return registry


def load_mock_registry() -> OntologyRegistry:
    """
    Create a mock registry for testing without ontology files.

    Returns:
        OntologyRegistry with mock indices
    """
    registry = OntologyRegistry()

    # Create mock indices without loading files
    mock_pato = OntologyIndex(name="pato")
    mock_pato.category = "base"

    mock_fao = OntologyIndex(name="fao")
    mock_fao.category = "base"

    registry.register_index("pato", mock_pato)
    registry.register_index("fao", mock_fao)

    return registry


# =============================================================================
# Phase 4: Vocabulary Normalization
# =============================================================================


@dataclass
class NormalizationResult:
    """
    Result of normalizing a single term.

    Attributes:
        original: The original term before normalization
        normalized: The normalized term (may be same as original)
        similarity: Cosine similarity to best ontology match
        was_normalized: Whether the term was actually changed
        ontology_source: Which ontology the match came from (if normalized)
        ontology_term_id: The ID of the matched ontology term (if normalized)
    """
    original: str
    normalized: str
    similarity: float
    was_normalized: bool
    ontology_source: Optional[str] = None
    ontology_term_id: Optional[str] = None


@dataclass
class CoverageAnalysis:
    """
    Analysis of vocabulary coverage for a set of terms.

    Attributes:
        total_terms: Total unique terms analyzed
        well_covered: Terms with similarity >= threshold
        partially_covered: Terms with similarity in [0.5, threshold)
        novel: Terms with similarity < 0.5
        coverage_ratio: Fraction of well-covered terms
        term_details: List of (term, best_match, similarity) for each term
    """
    total_terms: int
    well_covered: int
    partially_covered: int
    novel: int
    coverage_ratio: float
    term_details: List[Tuple[str, str, float]]


class ThresholdGatedNormalizer:
    """
    Normalize vocabulary only when confident.

    This implements Phase 4 of the SLM optimization approach. Terms are only
    mapped to ontology vocabulary when the similarity exceeds a threshold.
    Below-threshold terms are preserved as-is to avoid false mappings.

    Key Principle: Never force bad mappings. Terms below similarity threshold
    are preserved as-is.

    Example:
        >>> normalizer = ThresholdGatedNormalizer(registry, threshold=0.85)
        >>> result = normalizer.normalize("convex cap")
        >>> print(f"{result.original} -> {result.normalized} ({result.similarity:.2f})")
        convex cap -> convex (0.92)
    """

    def __init__(
        self,
        registry: OntologyRegistry,
        threshold: float = 0.85,
        min_similarity: float = 0.5
    ):
        """
        Initialize the normalizer.

        Args:
            registry: OntologyRegistry with loaded ontologies
            threshold: Minimum similarity to accept a mapping (default 0.85)
            min_similarity: Minimum similarity to consider partial coverage
        """
        self.registry = registry
        self.threshold = threshold
        self.min_similarity = min_similarity
        self._cache: Dict[str, NormalizationResult] = {}

    @property
    def encoder(self):
        """Get the shared encoder from the registry."""
        return self.registry.encoder

    def normalize(self, term: str) -> NormalizationResult:
        """
        Normalize term if confident match exists.

        Args:
            term: The term to normalize

        Returns:
            NormalizationResult with normalized term and metadata
        """
        # Check cache first
        cache_key = term.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        best_match, best_sim, source, term_id = self._find_best_match(term)

        if best_sim >= self.threshold:
            result = NormalizationResult(
                original=term,
                normalized=best_match,
                similarity=best_sim,
                was_normalized=True,
                ontology_source=source,
                ontology_term_id=term_id
            )
        else:
            # Keep original term
            result = NormalizationResult(
                original=term,
                normalized=term,
                similarity=best_sim,
                was_normalized=False,
                ontology_source=source if best_sim >= self.min_similarity else None,
                ontology_term_id=term_id if best_sim >= self.min_similarity else None
            )

        self._cache[cache_key] = result
        return result

    def _find_best_match(
        self,
        term: str
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Find the best matching ontology term across all registered ontologies.

        Args:
            term: The term to match

        Returns:
            Tuple of (best_match_name, similarity, ontology_name, term_id)
        """
        best_match = term
        best_sim = 0.0
        best_source = None
        best_id = None

        for ont_name, index in self.registry.ontologies.items():
            if not index.terms:
                continue

            results = index.search(term, top_k=1)
            if results:
                ont_term, similarity = results[0]
                if similarity > best_sim:
                    best_sim = similarity
                    best_match = ont_term.name
                    best_source = ont_name
                    best_id = ont_term.id

        return best_match, best_sim, best_source, best_id

    def normalize_batch(
        self,
        terms: List[str]
    ) -> List[NormalizationResult]:
        """
        Normalize multiple terms efficiently.

        Args:
            terms: List of terms to normalize

        Returns:
            List of NormalizationResult objects
        """
        return [self.normalize(term) for term in terms]

    def analyze_coverage(self, terms: List[str]) -> CoverageAnalysis:
        """
        Analyze vocabulary coverage for a set of terms.

        Args:
            terms: List of terms to analyze

        Returns:
            CoverageAnalysis with coverage statistics
        """
        unique_terms = list(set(t.lower().strip() for t in terms if t.strip()))

        well_covered = 0
        partially_covered = 0
        novel = 0
        details = []

        for term in unique_terms:
            result = self.normalize(term)

            if result.similarity >= self.threshold:
                well_covered += 1
            elif result.similarity >= self.min_similarity:
                partially_covered += 1
            else:
                novel += 1

            details.append((
                result.original,
                result.normalized if result.was_normalized else "(no match)",
                result.similarity
            ))

        # Sort by similarity (lowest first to highlight gaps)
        details.sort(key=lambda x: x[2])

        total = len(unique_terms) if unique_terms else 1  # Avoid division by zero
        return CoverageAnalysis(
            total_terms=len(unique_terms),
            well_covered=well_covered,
            partially_covered=partially_covered,
            novel=novel,
            coverage_ratio=well_covered / total,
            term_details=details
        )

    def clear_cache(self) -> None:
        """Clear the normalization cache."""
        self._cache.clear()


class VocabularyAnalyzer:
    """
    Analyze vocabulary in extracted JSON features.

    Extracts all unique terms from a nested JSON structure and analyzes
    their coverage against registered ontologies.

    Example:
        >>> analyzer = VocabularyAnalyzer(registry)
        >>> features = {"pileus": {"shape": ["convex"], "color": ["brown"]}}
        >>> analysis = analyzer.analyze_json(features)
        >>> print(f"Coverage: {analysis.coverage_ratio:.1%}")
    """

    def __init__(
        self,
        registry: OntologyRegistry,
        threshold: float = 0.85
    ):
        """
        Initialize the analyzer.

        Args:
            registry: OntologyRegistry with loaded ontologies
            threshold: Similarity threshold for "well-covered"
        """
        self.registry = registry
        self.normalizer = ThresholdGatedNormalizer(registry, threshold=threshold)

    def extract_terms(self, data: Any, terms: Optional[List[str]] = None) -> List[str]:
        """
        Recursively extract all string terms from a nested structure.

        Filters out:
        - Strings that are entirely punctuation/symbols
        - JSON syntax fragments (brackets, colons, etc.)
        - Very short strings (1-2 chars) that are just punctuation

        Normalizes terms to lowercase for consistent matching.

        Args:
            data: Nested dict/list/str structure
            terms: Accumulator list (used internally)

        Returns:
            List of all string terms found (normalized to lowercase)
        """
        import re

        if terms is None:
            terms = []

        def is_valid_term(s: str) -> bool:
            """Check if a string is a valid vocabulary term (not punctuation/syntax)."""
            if not s or not s.strip():
                return False
            stripped = s.strip()
            # Skip strings that are entirely punctuation, brackets, or whitespace
            # This catches things like "]", "]: [", ",", etc.
            if re.match(r'^[\[\]{}():,;\s\-\.]+$', stripped):
                return False
            # Must contain at least one alphanumeric character
            if not re.search(r'[a-zA-Z0-9]', stripped):
                return False
            return True

        def normalize_term(s: str) -> str:
            """Normalize a term: lowercase and strip whitespace."""
            return s.strip().lower()

        if isinstance(data, str):
            if is_valid_term(data):
                terms.append(normalize_term(data))
        elif isinstance(data, dict):
            for key, value in data.items():
                if is_valid_term(key):
                    terms.append(normalize_term(key))  # Keys are also vocabulary
                self.extract_terms(value, terms)
        elif isinstance(data, list):
            for item in data:
                self.extract_terms(item, terms)

        return terms

    def analyze_json(self, data: Dict[str, Any]) -> CoverageAnalysis:
        """
        Analyze vocabulary coverage in JSON feature data.

        Args:
            data: Nested feature dictionary

        Returns:
            CoverageAnalysis with statistics
        """
        terms = self.extract_terms(data)
        return self.normalizer.analyze_coverage(terms)

    def analyze_batch(
        self,
        data_list: List[Dict[str, Any]]
    ) -> CoverageAnalysis:
        """
        Analyze vocabulary coverage across multiple JSON outputs.

        Args:
            data_list: List of feature dictionaries

        Returns:
            Aggregated CoverageAnalysis
        """
        all_terms = []
        for data in data_list:
            all_terms.extend(self.extract_terms(data))
        return self.normalizer.analyze_coverage(all_terms)

    def get_novel_terms(
        self,
        data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get terms with low ontology coverage.

        Args:
            data: Feature dictionary
            threshold: Override default threshold (default: use normalizer's)

        Returns:
            List of (term, similarity) tuples for poorly-covered terms
        """
        thresh = threshold if threshold is not None else self.normalizer.threshold
        terms = self.extract_terms(data)
        novel = []

        for term in set(terms):
            result = self.normalizer.normalize(term)
            if result.similarity < thresh:
                novel.append((term, result.similarity))

        return sorted(novel, key=lambda x: x[1])


def normalize_json_output(
    data: Dict[str, Any],
    normalizer: ThresholdGatedNormalizer,
    normalize_keys: bool = True,
    normalize_values: bool = True
) -> Tuple[Dict[str, Any], Dict[str, NormalizationResult]]:
    """
    Apply vocabulary normalization to a nested JSON structure.

    This function walks the JSON structure and normalizes terms according
    to the threshold-gated normalizer. Keys and/or values can be normalized.

    Args:
        data: Nested feature dictionary to normalize
        normalizer: ThresholdGatedNormalizer instance
        normalize_keys: Whether to normalize dictionary keys
        normalize_values: Whether to normalize string values

    Returns:
        Tuple of (normalized_data, normalization_log)
        where normalization_log maps original terms to their NormalizationResult
    """
    log: Dict[str, NormalizationResult] = {}

    def normalize_recursive(obj: Any) -> Any:
        if isinstance(obj, str):
            if normalize_values:
                result = normalizer.normalize(obj)
                log[obj] = result
                return result.normalized
            return obj

        elif isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if normalize_keys:
                    key_result = normalizer.normalize(key)
                    log[key] = key_result
                    new_key = key_result.normalized
                else:
                    new_key = key

                new_dict[new_key] = normalize_recursive(value)
            return new_dict

        elif isinstance(obj, list):
            return [normalize_recursive(item) for item in obj]

        else:
            return obj

    normalized = normalize_recursive(data)
    return normalized, log


def print_coverage_report(
    analysis: CoverageAnalysis,
    show_details: bool = True,
    max_details: int = 20
) -> str:
    """
    Format a coverage analysis as a readable report.

    Args:
        analysis: CoverageAnalysis to report
        show_details: Whether to show individual term details
        max_details: Maximum number of terms to show in details

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "VOCABULARY COVERAGE ANALYSIS",
        "=" * 60,
        f"Total unique terms: {analysis.total_terms}",
        f"Well-covered (≥threshold): {analysis.well_covered} "
        f"({analysis.well_covered/analysis.total_terms*100:.1f}%)",
        f"Partially covered: {analysis.partially_covered} "
        f"({analysis.partially_covered/analysis.total_terms*100:.1f}%)",
        f"Novel/unmatched: {analysis.novel} "
        f"({analysis.novel/analysis.total_terms*100:.1f}%)",
        "",
        f"Coverage ratio: {analysis.coverage_ratio:.1%}",
    ]

    if show_details and analysis.term_details:
        lines.extend([
            "",
            "-" * 60,
            "TERM DETAILS (sorted by similarity, lowest first):",
            "-" * 60,
        ])

        for term, match, sim in analysis.term_details[:max_details]:
            status = "✓" if sim >= 0.85 else "~" if sim >= 0.5 else "✗"
            lines.append(f"  {status} {term:30} → {match:20} ({sim:.3f})")

        if len(analysis.term_details) > max_details:
            lines.append(f"  ... and {len(analysis.term_details) - max_details} more")

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# Phase 6: Graceful Degradation for Novel Terms
# =============================================================================

from enum import Enum


class ConfidenceLevel(Enum):
    """
    Confidence level for term mappings.

    Used to distinguish how confident we are in a term's ontology mapping.
    """
    HIGH = "high"          # Similarity >= threshold (safe to normalize)
    MEDIUM = "medium"      # Similarity in [0.5, threshold) (use with caution)
    LOW = "low"            # Similarity in [0.3, 0.5) (probably wrong)
    NOVEL = "novel"        # Similarity < 0.3 (no meaningful match)


@dataclass
class AnnotatedTerm:
    """
    A term with confidence annotation for graceful degradation.

    This preserves both the original term and any mapping information,
    allowing downstream processes to decide how to handle uncertain mappings.

    Attributes:
        original: The original term from the description
        normalized: The mapped ontology term (if confident enough)
        confidence: Confidence level of the mapping
        similarity: Cosine similarity to best ontology match
        ontology_source: Which ontology the match came from
        ontology_term_id: The ID of the matched ontology term
        preserve_original: Whether to keep the original term in output
    """
    original: str
    normalized: str
    confidence: ConfidenceLevel
    similarity: float
    ontology_source: Optional[str] = None
    ontology_term_id: Optional[str] = None
    preserve_original: bool = False

    def to_output(self, include_annotation: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Convert to output format.

        Args:
            include_annotation: If True, return dict with metadata

        Returns:
            String (normalized term) or dict with annotations
        """
        if include_annotation or self.preserve_original:
            result: Dict[str, Any] = {
                "term": self.normalized,
                "confidence": self.confidence.value,
            }
            if self.preserve_original and self.original != self.normalized:
                result["original"] = self.original
            if self.ontology_source:
                result["ontology"] = self.ontology_source
            return result
        return self.normalized


class GracefulDegradationHandler:
    """
    Handle novel terms gracefully without losing information.

    This class implements Phase 6 of the SLM optimization approach.
    It classifies terms by confidence level and preserves original terms
    when ontology mappings are uncertain.

    Key Principles:
    1. Never lose information
    2. Never force bad mappings
    3. Distinguish confidence levels
    4. Preserve original terms with annotations when uncertain

    Example:
        >>> handler = GracefulDegradationHandler(registry)
        >>> annotated = handler.annotate_term("subcanaliculate")
        >>> print(f"Confidence: {annotated.confidence.value}")
        Confidence: novel
        >>> print(f"Preserved: {annotated.preserve_original}")
        Preserved: True
    """

    # Default thresholds for confidence levels
    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.5
    LOW_THRESHOLD = 0.3

    def __init__(
        self,
        registry: OntologyRegistry,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.5,
        low_threshold: float = 0.3,
        preserve_below: Optional[float] = None
    ):
        """
        Initialize the handler.

        Args:
            registry: OntologyRegistry with loaded ontologies
            high_threshold: Threshold for HIGH confidence (default 0.85)
            medium_threshold: Threshold for MEDIUM confidence (default 0.5)
            low_threshold: Threshold for LOW confidence (default 0.3)
            preserve_below: Preserve original term if similarity below this
                           (default: same as medium_threshold)
        """
        self.registry = registry
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.preserve_below = preserve_below if preserve_below is not None else medium_threshold
        self._cache: Dict[str, AnnotatedTerm] = {}

    def _get_confidence_level(self, similarity: float) -> ConfidenceLevel:
        """Determine confidence level from similarity score."""
        if similarity >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif similarity >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif similarity >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NOVEL

    def _find_best_match(
        self,
        term: str
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Find the best matching ontology term."""
        best_match = term
        best_sim = 0.0
        best_source = None
        best_id = None

        for ont_name, index in self.registry.ontologies.items():
            if not index.terms:
                continue

            results = index.search(term, top_k=1)
            if results:
                ont_term, similarity = results[0]
                if similarity > best_sim:
                    best_sim = similarity
                    best_match = ont_term.name
                    best_source = ont_name
                    best_id = ont_term.id

        return best_match, best_sim, best_source, best_id

    def annotate_term(self, term: str) -> AnnotatedTerm:
        """
        Annotate a term with confidence information.

        Args:
            term: The term to annotate

        Returns:
            AnnotatedTerm with confidence and mapping information
        """
        cache_key = term.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        best_match, similarity, source, term_id = self._find_best_match(term)
        confidence = self._get_confidence_level(similarity)

        # Determine whether to preserve original
        preserve = similarity < self.preserve_below

        # For high confidence, use the normalized term
        # For low confidence, keep the original
        if confidence == ConfidenceLevel.HIGH:
            normalized = best_match
        elif confidence == ConfidenceLevel.MEDIUM:
            # Use normalized but flag for review
            normalized = best_match
        else:
            # Keep original for low/novel
            normalized = term

        annotated = AnnotatedTerm(
            original=term,
            normalized=normalized,
            confidence=confidence,
            similarity=similarity,
            ontology_source=source,
            ontology_term_id=term_id,
            preserve_original=preserve
        )

        self._cache[cache_key] = annotated
        return annotated

    def annotate_batch(self, terms: List[str]) -> List[AnnotatedTerm]:
        """Annotate multiple terms."""
        return [self.annotate_term(term) for term in terms]

    def get_novel_terms(self, terms: List[str]) -> List[AnnotatedTerm]:
        """Get terms classified as NOVEL (no good ontology match)."""
        annotated = self.annotate_batch(terms)
        return [a for a in annotated if a.confidence == ConfidenceLevel.NOVEL]

    def get_uncertain_terms(self, terms: List[str]) -> List[AnnotatedTerm]:
        """Get terms below HIGH confidence."""
        annotated = self.annotate_batch(terms)
        return [a for a in annotated if a.confidence != ConfidenceLevel.HIGH]

    def summarize(self, terms: List[str]) -> Dict[str, Any]:
        """
        Summarize the confidence distribution for a set of terms.

        Args:
            terms: List of terms to analyze

        Returns:
            Dict with counts and percentages by confidence level
        """
        annotated = self.annotate_batch(terms)
        total = len(annotated)

        counts = {
            ConfidenceLevel.HIGH: 0,
            ConfidenceLevel.MEDIUM: 0,
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.NOVEL: 0,
        }

        for a in annotated:
            counts[a.confidence] += 1

        return {
            "total": total,
            "high": {"count": counts[ConfidenceLevel.HIGH],
                     "percent": counts[ConfidenceLevel.HIGH] / total * 100 if total else 0},
            "medium": {"count": counts[ConfidenceLevel.MEDIUM],
                       "percent": counts[ConfidenceLevel.MEDIUM] / total * 100 if total else 0},
            "low": {"count": counts[ConfidenceLevel.LOW],
                    "percent": counts[ConfidenceLevel.LOW] / total * 100 if total else 0},
            "novel": {"count": counts[ConfidenceLevel.NOVEL],
                      "percent": counts[ConfidenceLevel.NOVEL] / total * 100 if total else 0},
        }

    def clear_cache(self) -> None:
        """Clear the annotation cache."""
        self._cache.clear()


class RobustOntologyPipeline:
    """
    Full pipeline for vocabulary processing with graceful degradation.

    Combines normalization (Phase 4) with graceful degradation (Phase 6)
    to provide a robust vocabulary handling system.

    Example:
        >>> pipeline = RobustOntologyPipeline(registry)
        >>> result = pipeline.process_json(features, include_annotations=True)
        >>> print(result["_metadata"]["confidence_summary"])
    """

    def __init__(
        self,
        registry: OntologyRegistry,
        normalization_threshold: float = 0.85,
        preserve_threshold: float = 0.5
    ):
        """
        Initialize the pipeline.

        Args:
            registry: OntologyRegistry with loaded ontologies
            normalization_threshold: Threshold for confident normalization
            preserve_threshold: Threshold below which to preserve original
        """
        self.registry = registry
        self.normalizer = ThresholdGatedNormalizer(
            registry,
            threshold=normalization_threshold
        )
        self.degradation_handler = GracefulDegradationHandler(
            registry,
            high_threshold=normalization_threshold,
            preserve_below=preserve_threshold
        )

    def process_term(
        self,
        term: str,
        include_annotation: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Process a single term through the pipeline.

        Args:
            term: Term to process
            include_annotation: If True, return dict with metadata

        Returns:
            Processed term (string or dict)
        """
        annotated = self.degradation_handler.annotate_term(term)
        return annotated.to_output(include_annotation=include_annotation)

    def process_json(
        self,
        data: Dict[str, Any],
        normalize_keys: bool = True,
        normalize_values: bool = True,
        include_annotations: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a nested JSON structure through the pipeline.

        Args:
            data: Nested feature dictionary
            normalize_keys: Whether to process dictionary keys
            normalize_values: Whether to process string values
            include_annotations: If True, include term-level annotations
            include_metadata: If True, add _metadata with summary

        Returns:
            Processed JSON with optional annotations and metadata
        """
        all_terms: List[str] = []

        def process_recursive(obj: Any) -> Any:
            if isinstance(obj, str):
                if normalize_values:
                    all_terms.append(obj)
                    return self.process_term(obj, include_annotations)
                return obj

            elif isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    if normalize_keys:
                        all_terms.append(key)
                        new_key = self.process_term(key, include_annotations)
                        # If key became a dict, extract the term
                        if isinstance(new_key, dict):
                            new_key = new_key.get("term", key)
                    else:
                        new_key = key

                    new_dict[new_key] = process_recursive(value)
                return new_dict

            elif isinstance(obj, list):
                return [process_recursive(item) for item in obj]

            else:
                return obj

        result = process_recursive(data)

        if include_metadata:
            # Add metadata about the processing
            summary = self.degradation_handler.summarize(all_terms)
            novel_terms = self.degradation_handler.get_novel_terms(all_terms)

            result["_metadata"] = {
                "confidence_summary": summary,
                "novel_terms": [
                    {"term": a.original, "similarity": a.similarity}
                    for a in novel_terms
                ],
                "processing": {
                    "normalization_threshold": self.normalizer.threshold,
                    "preserve_threshold": self.degradation_handler.preserve_below,
                }
            }

        return result

    def analyze_and_report(self, data: Dict[str, Any]) -> str:
        """
        Analyze vocabulary and generate a detailed report.

        Args:
            data: Feature dictionary to analyze

        Returns:
            Formatted report string
        """
        # Extract all terms
        analyzer = VocabularyAnalyzer(self.registry)
        terms = analyzer.extract_terms(data)

        # Get confidence summary
        summary = self.degradation_handler.summarize(terms)
        novel = self.degradation_handler.get_novel_terms(terms)
        uncertain = self.degradation_handler.get_uncertain_terms(terms)

        lines = [
            "=" * 60,
            "GRACEFUL DEGRADATION ANALYSIS",
            "=" * 60,
            f"Total terms: {summary['total']}",
            "",
            "CONFIDENCE DISTRIBUTION:",
            f"  HIGH (≥{self.degradation_handler.high_threshold:.0%}):    "
            f"{summary['high']['count']:3d} ({summary['high']['percent']:.1f}%)",
            f"  MEDIUM ({self.degradation_handler.medium_threshold:.0%}-"
            f"{self.degradation_handler.high_threshold:.0%}): "
            f"{summary['medium']['count']:3d} ({summary['medium']['percent']:.1f}%)",
            f"  LOW ({self.degradation_handler.low_threshold:.0%}-"
            f"{self.degradation_handler.medium_threshold:.0%}):    "
            f"{summary['low']['count']:3d} ({summary['low']['percent']:.1f}%)",
            f"  NOVEL (<{self.degradation_handler.low_threshold:.0%}):   "
            f"{summary['novel']['count']:3d} ({summary['novel']['percent']:.1f}%)",
        ]

        if novel:
            lines.extend([
                "",
                "-" * 60,
                "NOVEL TERMS (no good ontology match):",
                "-" * 60,
            ])
            for annotated in novel[:20]:
                lines.append(f"  • {annotated.original} (sim: {annotated.similarity:.3f})")
            if len(novel) > 20:
                lines.append(f"  ... and {len(novel) - 20} more")

        if uncertain and len(uncertain) > len(novel):
            medium_low = [a for a in uncertain if a.confidence != ConfidenceLevel.NOVEL]
            if medium_low:
                lines.extend([
                    "",
                    "-" * 60,
                    "UNCERTAIN MAPPINGS (MEDIUM/LOW confidence):",
                    "-" * 60,
                ])
                for annotated in medium_low[:15]:
                    lines.append(
                        f"  ~ {annotated.original:25} → {annotated.normalized:20} "
                        f"({annotated.confidence.value}, {annotated.similarity:.3f})"
                    )
                if len(medium_low) > 15:
                    lines.append(f"  ... and {len(medium_low) - 15} more")

        lines.extend([
            "",
            "-" * 60,
            "RECOMMENDATIONS:",
            "-" * 60,
        ])

        if summary['novel']['percent'] > 20:
            lines.append("  ⚠ High novel term rate (>20%). Consider:")
            lines.append("    - Adding domain-specific ontologies")
            lines.append("    - Creating custom vocabulary mappings")
        elif summary['novel']['percent'] > 5:
            lines.append("  ~ Moderate novel term rate (5-20%).")
            lines.append("    Novel terms will be preserved with annotations.")
        else:
            lines.append("  ✓ Low novel term rate (<5%). Vocabulary coverage is good.")

        if summary['medium']['percent'] + summary['low']['percent'] > 15:
            lines.append("  ⚠ Many uncertain mappings. Review MEDIUM/LOW terms above.")

        lines.append("=" * 60)
        return "\n".join(lines)


def annotate_json_output(
    data: Dict[str, Any],
    handler: GracefulDegradationHandler,
    annotate_keys: bool = True,
    annotate_values: bool = True,
    include_term_annotations: bool = False
) -> Tuple[Dict[str, Any], Dict[str, AnnotatedTerm]]:
    """
    Apply graceful degradation to a nested JSON structure.

    This is similar to normalize_json_output but uses confidence-based
    handling instead of simple threshold gating.

    Args:
        data: Nested feature dictionary
        handler: GracefulDegradationHandler instance
        annotate_keys: Whether to process dictionary keys
        annotate_values: Whether to process string values
        include_term_annotations: If True, include inline annotations

    Returns:
        Tuple of (processed_data, annotation_log)
    """
    log: Dict[str, AnnotatedTerm] = {}

    def process_recursive(obj: Any) -> Any:
        if isinstance(obj, str):
            if annotate_values:
                annotated = handler.annotate_term(obj)
                log[obj] = annotated
                return annotated.to_output(include_annotation=include_term_annotations)
            return obj

        elif isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if annotate_keys:
                    annotated = handler.annotate_term(key)
                    log[key] = annotated
                    new_key = annotated.to_output(include_annotation=include_term_annotations)
                    # If key became a dict, extract the term
                    if isinstance(new_key, dict):
                        new_key = new_key.get("term", key)
                else:
                    new_key = key

                new_dict[new_key] = process_recursive(value)
            return new_dict

        elif isinstance(obj, list):
            return [process_recursive(item) for item in obj]

        else:
            return obj

    result = process_recursive(data)
    return result, log
