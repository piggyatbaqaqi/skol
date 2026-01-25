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
