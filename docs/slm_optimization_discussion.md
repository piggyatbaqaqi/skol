# Optimizing a Mistral-Based SLM for Automated Synoptic Key Generation

## Version 2 — Expanded with Ontology Integration and Incremental Implementation Plan

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Design Constraints](#design-constraints)
3. [Solution Architecture Overview](#solution-architecture-overview)
4. [Completed Work](#completed-work)
5. [Incremental Implementation Plan](#incremental-implementation-plan)
6. [Technical Reference](#technical-reference)
7. [Appendices](#appendices)

---

## Problem Statement

### Goal

Develop a general framework for building synoptic keys from species descriptions, starting with mycology and expanding to other fields of biology. The system uses a Mistral-based small language model to parse species descriptions into structured JSON.

### Hardware Constraints

- Single RTX 5090 GPU with 24GB VRAM
- 96GB host RAM
- Preference for computations completing within 1-2 days

### Original Problems

1. **Low structural success rate**: Less than 50% of generations produce valid JSON (most return `{}`)
2. **Schema violations**: Leaf nodes are sometimes single values instead of required lists
3. **Vocabulary inconsistency**: Features appear at different hierarchy levels; synonyms are used inconsistently

### Required Output Format

Nested dictionaries with configurable depth (2-4+ levels), with string arrays at leaf nodes:

```json
{
  "feature": {
    "subfeature": {
      "sub-subfeature": ["value1", "value2", "value3"]
    }
  }
}
```

---

## Design Constraints

### Non-Negotiable

- **No manual intervention**: Any step requiring human data-specific decisions is not acceptable
- **Domain-agnostic**: Must generalize across biological domains (mycology → botany → entomology, etc.)
- **Incremental validation**: Each enhancement must be independently assessable

### Practical

- Prefer solutions that can be easily extended (new ontologies, new domains)
- Avoid premature optimization—validate that a problem exists before solving it

---

## Solution Architecture Overview

### Target Pipeline (End State)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT DESCRIPTION                                           │
│         │                                                       │
│         ▼                                                       │
│  2. ONTOLOGY RETRIEVAL (Tiered)                                 │
│     - Base ontologies (PATO) always included                    │
│     - Domain ontologies (FAO) always included                   │
│     - Specialized ontologies selected dynamically               │
│         │                                                       │
│         ▼                                                       │
│  3. CONTEXT CONSTRUCTION                                        │
│     - Linearized ontology subgraphs with level markers          │
│     - Schema constraints                                        │
│     - Few-shot examples (optional)                              │
│         │                                                       │
│         ▼                                                       │
│  4. CONSTRAINED GENERATION                                      │
│     - Schema constraint (nested dict → array structure)         │
│     - Variable depth (configurable 2-6 levels)                  │
│         │                                                       │
│         ▼                                                       │
│  5. POST-PROCESSING                                             │
│     - Vocabulary normalization (threshold-gated)                │
│     - Hierarchy consistency validation                          │
│     - Novel term handling (preserve with annotations)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Incremental Approach

Rather than implementing the full pipeline at once, we build incrementally:

1. Start with constrained decoding (DONE)
2. Add ontology context injection
3. Add vocabulary coverage analysis
4. Add normalization (only if needed based on analysis)
5. Add tiered ontology selection
6. Add graceful degradation for novel terms (only if needed)

---

## Completed Work

### Phase 0: Constrained Decoding with Variable Depth ✓

Implemented schema-constrained generation ensuring:
- 100% valid JSON output
- Correct nested dictionary structure
- String arrays at leaf nodes
- Variable depth (currently 2-4 levels)

**Schema Definition:**

```json
{
  "$defs": {
    "level4": {
      "type": "array",
      "items": {"type": "string"}
    },
    "level3": {
      "oneOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "additionalProperties": {"$ref": "#/$defs/level4"}}
      ]
    },
    "level2": {
      "oneOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "additionalProperties": {"$ref": "#/$defs/level3"}}
      ]
    },
    "level1": {
      "oneOf": [
        {"type": "array", "items": {"type": "string"}},
        {"type": "object", "additionalProperties": {"$ref": "#/$defs/level2"}}
      ]
    }
  },
  "type": "object",
  "additionalProperties": {"$ref": "#/$defs/level1"}
}
```

---

## Incremental Implementation Plan

### Phase 1: Extend Schema Depth

**Goal:** Support deeper hierarchies (up to 6 levels) for complex substructures.

**Rationale:** Some anatomical features have deep hierarchies (e.g., fruiting body → hymenium → basidia → sterigmata → spores → ornamentation).

**Implementation:**

```python
def generate_variable_depth_schema(max_depth: int) -> dict:
    """Generate JSON schema supporting 1 to max_depth levels."""
    
    schema = {
        "$defs": {},
        "type": "object"
    }
    
    # Build from deepest level up
    for level in range(max_depth, 0, -1):
        level_name = f"level{level}"
        
        if level == max_depth:
            # Deepest level is always an array
            schema["$defs"][level_name] = {
                "type": "array",
                "items": {"type": "string"}
            }
        else:
            # Other levels can be array OR object containing next level
            next_level = f"level{level + 1}"
            schema["$defs"][level_name] = {
                "oneOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "object", "additionalProperties": {"$ref": f"#/$defs/{next_level}"}}
                ]
            }
    
    schema["additionalProperties"] = {"$ref": "#/$defs/level1"}
    
    return schema

# Usage
schema_depth_6 = generate_variable_depth_schema(6)
```

**Validation Criteria:**
- Schema validates correctly
- Generation still completes in reasonable time
- Output quality remains stable

**Estimated Effort:** 1-2 hours

---

### Phase 2: Basic Ontology Integration (PATO + FAO)

**Goal:** Inject relevant ontology context into prompts to guide vocabulary selection.

**Approach:** Hierarchical Prompt Injection with Retrieved Subgraphs

**Why This Approach:**
- Directly influences generation (not just post-processing)
- Provides hierarchical context the model can learn from
- Easily extensible to additional ontologies
- No training required

#### 2.1 Ontology Loading Infrastructure

```python
from pronto import Ontology
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class OntologyTerm:
    id: str
    name: str
    definition: Optional[str]
    depth: int
    ancestors: list[str]
    embedding: np.ndarray

class OntologyIndex:
    """Searchable index for a single ontology."""
    
    def __init__(self, ontology_path: str, name: str):
        self.name = name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.terms: list[OntologyTerm] = []
        self.term_embeddings: np.ndarray = None
        
        self._load_ontology(ontology_path)
    
    def _load_ontology(self, path: str):
        ont = Ontology(path)
        
        terms_data = []
        for term in ont.terms():
            ancestors = [a.name for a in term.superclasses(with_self=False)]
            depth = len(ancestors)
            
            text = f"{term.name}: {term.definition or ''}"
            
            terms_data.append({
                "id": term.id,
                "name": term.name,
                "definition": term.definition,
                "depth": depth,
                "ancestors": ancestors,
                "text": text
            })
        
        # Batch encode for efficiency
        texts = [t["text"] for t in terms_data]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        for term_data, embedding in zip(terms_data, embeddings):
            self.terms.append(OntologyTerm(
                id=term_data["id"],
                name=term_data["name"],
                definition=term_data["definition"],
                depth=term_data["depth"],
                ancestors=term_data["ancestors"],
                embedding=embedding
            ))
        
        self.term_embeddings = np.stack([t.embedding for t in self.terms])
    
    def search(self, query: str, top_k: int = 20) -> list[tuple[OntologyTerm, float]]:
        """Find terms most similar to query."""
        query_embedding = self.encoder.encode(query)
        
        similarities = np.dot(self.term_embeddings, query_embedding) / (
            np.linalg.norm(self.term_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.terms[i], similarities[i]) for i in top_indices]
    
    def get_terms_at_depth(self, min_depth: int, max_depth: int) -> list[OntologyTerm]:
        """Get all terms within a depth range."""
        return [t for t in self.terms if min_depth <= t.depth <= max_depth]
```

#### 2.2 Ontology Registry (Extensible Design)

```python
class OntologyRegistry:
    """
    Central registry for all ontologies.
    Designed for easy addition of new ontologies.
    """
    
    def __init__(self):
        self.ontologies: dict[str, OntologyIndex] = {}
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def register(self, name: str, path: str, category: str = "general"):
        """
        Register a new ontology.
        
        Args:
            name: Unique identifier (e.g., "pato", "fao", "ascomycete_anatomy")
            path: Path to .obo file
            category: "base" (always used) or "specialized" (selected dynamically)
        """
        index = OntologyIndex(path, name)
        index.category = category
        self.ontologies[name] = index
        
        print(f"Registered ontology '{name}': {len(index.terms)} terms")
    
    def get(self, name: str) -> OntologyIndex:
        return self.ontologies[name]
    
    def get_base_ontologies(self) -> list[OntologyIndex]:
        return [o for o in self.ontologies.values() if o.category == "base"]
    
    def get_specialized_ontologies(self) -> list[OntologyIndex]:
        return [o for o in self.ontologies.values() if o.category == "specialized"]
    
    def list_registered(self) -> list[dict]:
        return [
            {"name": name, "category": o.category, "term_count": len(o.terms)}
            for name, o in self.ontologies.items()
        ]

# Initial setup
registry = OntologyRegistry()
registry.register("pato", "pato.obo", category="base")
registry.register("fao", "fao.obo", category="base")

# Future additions (examples)
# registry.register("ascomycete", "asco.obo", category="specialized")
# registry.register("plant_ontology", "po.obo", category="specialized")
```

#### 2.3 Subgraph Retrieval and Linearization

```python
class OntologyContextBuilder:
    """Build prompt context from ontology subgraphs."""
    
    def __init__(self, registry: OntologyRegistry):
        self.registry = registry
        self.encoder = registry.encoder
    
    def build_context(self, 
                      description: str,
                      anatomy_ontology: str = "fao",
                      quality_ontology: str = "pato",
                      top_k_per_ontology: int = 15,
                      max_context_chars: int = 2000) -> str:
        """
        Build linearized ontology context for a description.
        
        Returns formatted string for prompt injection.
        """
        
        # Retrieve relevant terms from each ontology
        anatomy_index = self.registry.get(anatomy_ontology)
        quality_index = self.registry.get(quality_ontology)
        
        anatomy_results = anatomy_index.search(description, top_k=top_k_per_ontology)
        quality_results = quality_index.search(description, top_k=top_k_per_ontology)
        
        # Build hierarchical context
        lines = []
        
        # Anatomical hierarchy (for top-level keys)
        lines.append("ANATOMICAL STRUCTURES (use for top-level feature keys):")
        lines.extend(self._format_hierarchy(anatomy_results))
        
        lines.append("")
        
        # Quality hierarchy (for nested property keys)
        lines.append("QUALITY PROPERTIES (use for nested property keys):")
        lines.extend(self._format_hierarchy(quality_results))
        
        lines.append("")
        
        # Add path examples showing hierarchy
        lines.append("HIERARCHY EXAMPLES:")
        lines.extend(self._format_path_examples(anatomy_results[:5]))
        lines.extend(self._format_path_examples(quality_results[:5]))
        
        context = "\n".join(lines)
        
        # Truncate if needed
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[truncated]"
        
        return context
    
    def _format_hierarchy(self, results: list[tuple[OntologyTerm, float]]) -> list[str]:
        """Format terms grouped by depth level."""
        by_depth = {}
        for term, score in results:
            depth = min(term.depth, 4)  # Cap display depth
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(term.name)
        
        lines = []
        for depth in sorted(by_depth.keys()):
            terms = by_depth[depth][:8]  # Limit per level
            lines.append(f"  [L{depth + 1}] {' | '.join(terms)}")
        
        return lines
    
    def _format_path_examples(self, results: list[tuple[OntologyTerm, float]]) -> list[str]:
        """Show full hierarchy paths for context."""
        lines = []
        seen_paths = set()
        
        for term, score in results:
            if term.ancestors:
                # Show path from near-root to term
                path_parts = term.ancestors[-3:] + [term.name]
                path_str = " > ".join(path_parts)
                
                if path_str not in seen_paths:
                    seen_paths.add(path_str)
                    lines.append(f"  {path_str}")
        
        return lines[:5]  # Limit examples
```

#### 2.4 Prompt Template Integration

```python
class OntologyGuidedGenerator:
    """Generate structured JSON with ontology-guided vocabulary."""
    
    def __init__(self, model, registry: OntologyRegistry, schema: dict):
        self.model = model
        self.context_builder = OntologyContextBuilder(registry)
        self.schema = schema
    
    def generate(self, description: str) -> dict:
        """Generate ontology-guided structured output."""
        
        # Build ontology context
        ontology_context = self.context_builder.build_context(description)
        
        # Construct prompt
        prompt = f"""Extract structured features from a biological species description.

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
        
        # Generate with schema constraint
        # (Implementation depends on your constrained decoding setup)
        result = self.model.generate(prompt, schema=self.schema)
        
        return result
```

**Validation Criteria:**
- Ontology loading completes without errors
- Retrieval returns sensible results for test descriptions
- Generated vocabulary shows increased consistency with ontology terms
- No significant increase in generation time

**Estimated Effort:** 1-2 days

---

### Phase 3: Vocabulary Coverage Analysis Tool

**Goal:** Before implementing complex normalization, assess how well ontologies cover the actual vocabulary being generated.

**Rationale:** Don't build solutions for problems that may not be widespread.

#### 3.1 Coverage Analyzer

```python
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import json

@dataclass
class TermAnalysis:
    term: str
    frequency: int
    best_ontology_match: Optional[str]
    best_similarity: float
    match_quality: str  # "exact", "high", "medium", "low", "none"
    suggested_mapping: Optional[str]

class VocabularyCoverageAnalyzer:
    """
    Analyze how well ontologies cover extracted vocabulary.
    
    Use this BEFORE implementing complex normalization to understand
    the scope of the vocabulary gap problem.
    """
    
    def __init__(self, registry: OntologyRegistry):
        self.registry = registry
        self.encoder = registry.encoder
        
        # Thresholds for match quality
        self.thresholds = {
            "exact": 0.98,
            "high": 0.85,
            "medium": 0.65,
            "low": 0.45
        }
    
    def analyze_extractions(self, 
                            extractions: list[dict],
                            output_path: Optional[str] = None) -> dict:
        """
        Analyze vocabulary coverage across multiple extractions.
        
        Args:
            extractions: List of extracted JSON structures
            output_path: Optional path to save detailed CSV report
        
        Returns:
            Summary statistics and detailed term analysis
        """
        
        # Collect all terms with frequencies
        term_frequencies = defaultdict(int)
        term_contexts = defaultdict(list)  # term -> list of (path, source_description)
        
        for i, extraction in enumerate(extractions):
            self._collect_terms(extraction, term_frequencies, term_contexts, 
                               path=[], source_id=i)
        
        # Analyze each unique term
        analyses = []
        for term, freq in term_frequencies.items():
            analysis = self._analyze_term(term, freq)
            analyses.append(analysis)
        
        # Compute statistics
        stats = self._compute_statistics(analyses)
        
        # Generate report
        report = {
            "summary": stats,
            "terms_by_quality": self._group_by_quality(analyses),
            "recommendations": self._generate_recommendations(stats),
            "detailed_analyses": analyses
        }
        
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _collect_terms(self, obj, frequencies, contexts, path, source_id):
        """Recursively collect terms from JSON structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                frequencies[key.lower()] += 1
                contexts[key.lower()].append({
                    "path": ".".join(path),
                    "source_id": source_id,
                    "position": "key"
                })
                self._collect_terms(value, frequencies, contexts, 
                                   path + [key], source_id)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    frequencies[item.lower()] += 1
                    contexts[item.lower()].append({
                        "path": ".".join(path),
                        "source_id": source_id,
                        "position": "value"
                    })
    
    def _analyze_term(self, term: str, frequency: int) -> TermAnalysis:
        """Analyze a single term against all ontologies."""
        
        best_match = None
        best_similarity = 0.0
        best_ontology = None
        
        term_embedding = self.encoder.encode(term)
        
        for ont_name, ont_index in self.registry.ontologies.items():
            # Check for exact match first
            exact_matches = [t for t in ont_index.terms 
                           if t.name.lower() == term.lower()]
            if exact_matches:
                return TermAnalysis(
                    term=term,
                    frequency=frequency,
                    best_ontology_match=ont_name,
                    best_similarity=1.0,
                    match_quality="exact",
                    suggested_mapping=exact_matches[0].name
                )
            
            # Find nearest neighbor
            similarities = np.dot(ont_index.term_embeddings, term_embedding) / (
                np.linalg.norm(ont_index.term_embeddings, axis=1) * 
                np.linalg.norm(term_embedding)
            )
            
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_match = ont_index.terms[max_idx].name
                best_ontology = ont_name
        
        # Determine match quality
        if best_similarity >= self.thresholds["high"]:
            quality = "high"
        elif best_similarity >= self.thresholds["medium"]:
            quality = "medium"
        elif best_similarity >= self.thresholds["low"]:
            quality = "low"
        else:
            quality = "none"
        
        return TermAnalysis(
            term=term,
            frequency=frequency,
            best_ontology_match=best_ontology,
            best_similarity=best_similarity,
            match_quality=quality,
            suggested_mapping=best_match if quality in ("high", "exact") else None
        )
    
    def _compute_statistics(self, analyses: list[TermAnalysis]) -> dict:
        """Compute coverage statistics."""
        total_terms = len(analyses)
        total_occurrences = sum(a.frequency for a in analyses)
        
        by_quality = defaultdict(lambda: {"terms": 0, "occurrences": 0})
        for a in analyses:
            by_quality[a.match_quality]["terms"] += 1
            by_quality[a.match_quality]["occurrences"] += a.frequency
        
        return {
            "total_unique_terms": total_terms,
            "total_occurrences": total_occurrences,
            "coverage_by_quality": {
                quality: {
                    "terms": data["terms"],
                    "term_percentage": data["terms"] / total_terms * 100,
                    "occurrences": data["occurrences"],
                    "occurrence_percentage": data["occurrences"] / total_occurrences * 100
                }
                for quality, data in by_quality.items()
            },
            "well_covered": {
                "terms": by_quality["exact"]["terms"] + by_quality["high"]["terms"],
                "percentage": (by_quality["exact"]["terms"] + by_quality["high"]["terms"]) / total_terms * 100
            },
            "needs_attention": {
                "terms": by_quality["low"]["terms"] + by_quality["none"]["terms"],
                "percentage": (by_quality["low"]["terms"] + by_quality["none"]["terms"]) / total_terms * 100
            }
        }
    
    def _group_by_quality(self, analyses: list[TermAnalysis]) -> dict:
        """Group terms by match quality for review."""
        groups = defaultdict(list)
        
        for a in analyses:
            groups[a.match_quality].append({
                "term": a.term,
                "frequency": a.frequency,
                "similarity": round(a.best_similarity, 3),
                "suggested": a.suggested_mapping
            })
        
        # Sort each group by frequency (most common first)
        for quality in groups:
            groups[quality].sort(key=lambda x: x["frequency"], reverse=True)
        
        return dict(groups)
    
    def _generate_recommendations(self, stats: dict) -> list[str]:
        """Generate actionable recommendations based on coverage."""
        recommendations = []
        
        well_covered_pct = stats["well_covered"]["percentage"]
        needs_attention_pct = stats["needs_attention"]["percentage"]
        
        if well_covered_pct >= 90:
            recommendations.append(
                "EXCELLENT COVERAGE: >90% of terms well-covered by ontologies. "
                "Simple post-hoc normalization should suffice."
            )
        elif well_covered_pct >= 75:
            recommendations.append(
                "GOOD COVERAGE: 75-90% of terms covered. Consider implementing "
                "corpus vocabulary augmentation for the remaining terms."
            )
        elif well_covered_pct >= 50:
            recommendations.append(
                "MODERATE COVERAGE: 50-75% of terms covered. Recommend implementing "
                "hybrid vocabulary (ontology + corpus-derived) and graceful degradation."
            )
        else:
            recommendations.append(
                "LOW COVERAGE: <50% of terms covered. May need domain-specific "
                "ontologies or significant corpus vocabulary augmentation."
            )
        
        if needs_attention_pct > 20:
            recommendations.append(
                f"WARNING: {needs_attention_pct:.1f}% of terms have poor ontology matches. "
                "Review the 'low' and 'none' quality groups to identify patterns."
            )
        
        return recommendations
    
    def _save_report(self, report: dict, path: str):
        """Save detailed report to files."""
        import csv
        
        # Save JSON summary
        json_path = path.replace(".csv", "_summary.json")
        with open(json_path, "w") as f:
            json.dump({
                "summary": report["summary"],
                "recommendations": report["recommendations"],
                "terms_by_quality": report["terms_by_quality"]
            }, f, indent=2)
        
        # Save detailed CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "term", "frequency", "match_quality", 
                "similarity", "best_ontology", "suggested_mapping"
            ])
            
            for analysis in sorted(report["detailed_analyses"], 
                                  key=lambda x: x.frequency, reverse=True):
                writer.writerow([
                    analysis.term,
                    analysis.frequency,
                    analysis.match_quality,
                    round(analysis.best_similarity, 3),
                    analysis.best_ontology_match,
                    analysis.suggested_mapping
                ])
        
        print(f"Saved summary to {json_path}")
        print(f"Saved detailed report to {path}")
```

#### 3.2 Quick Assessment Script

```python
def quick_coverage_assessment(extractions: list[dict], 
                               registry: OntologyRegistry) -> None:
    """
    Run quick coverage assessment and print summary.
    
    Usage:
        extractions = [json.load(open(f)) for f in extraction_files]
        quick_coverage_assessment(extractions, registry)
    """
    
    analyzer = VocabularyCoverageAnalyzer(registry)
    report = analyzer.analyze_extractions(extractions, 
                                          output_path="coverage_report.csv")
    
    print("\n" + "="*60)
    print("VOCABULARY COVERAGE ASSESSMENT")
    print("="*60)
    
    stats = report["summary"]
    print(f"\nTotal unique terms: {stats['total_unique_terms']}")
    print(f"Total occurrences: {stats['total_occurrences']}")
    
    print("\nCoverage by quality:")
    for quality, data in stats["coverage_by_quality"].items():
        print(f"  {quality.upper():8s}: {data['terms']:4d} terms "
              f"({data['term_percentage']:5.1f}%), "
              f"{data['occurrences']:5d} occurrences "
              f"({data['occurrence_percentage']:5.1f}%)")
    
    print(f"\nWell-covered (exact + high): {stats['well_covered']['percentage']:.1f}%")
    print(f"Needs attention (low + none): {stats['needs_attention']['percentage']:.1f}%")
    
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    print("\nTop uncovered terms (review these):")
    uncovered = report["terms_by_quality"].get("none", [])[:10]
    for t in uncovered:
        print(f"  - '{t['term']}' (freq: {t['frequency']}, "
              f"nearest: '{t['suggested']}' @ {t['similarity']})")
    
    print("\nSee coverage_report.csv for full details.")
```

**Validation Criteria:**
- Tool runs successfully on existing extractions
- Report clearly identifies coverage gaps
- Recommendations align with observed vocabulary quality

**Estimated Effort:** 0.5-1 day

---

### Phase 4: Vocabulary Normalization (Conditional)

**Goal:** Implement threshold-gated vocabulary normalization.

**Prerequisite:** Phase 3 analysis shows normalization is needed (well-covered < 90%).

**Implementation:** Only proceed if Phase 3 indicates significant coverage gaps. See Technical Reference section for full implementation.

**Key Principle:** Never force bad mappings. Terms below similarity threshold are preserved as-is.

```python
class ThresholdGatedNormalizer:
    """Normalize vocabulary only when confident."""
    
    def __init__(self, registry: OntologyRegistry, threshold: float = 0.85):
        self.registry = registry
        self.threshold = threshold
        self.encoder = registry.encoder
    
    def normalize(self, term: str) -> tuple[str, float, bool]:
        """
        Normalize term if confident match exists.
        
        Returns:
            (normalized_term, similarity, was_normalized)
        """
        best_match, best_sim = self._find_best_match(term)
        
        if best_sim >= self.threshold:
            return (best_match, best_sim, True)
        else:
            return (term, best_sim, False)  # Keep original
    
    def _find_best_match(self, term: str) -> tuple[str, float]:
        # Implementation similar to analyzer
        ...
```

**Estimated Effort:** 0.5-1 day (if needed)

---

### Phase 5: Tiered Ontology Selection

**Goal:** Dynamically select specialized ontologies based on description content.

**Prerequisite:** Base ontology integration (Phase 2) is working well.

**Implementation:**

```python
class TieredOntologyRetriever:
    """
    Tier 1: Base ontologies (always included)
    Tier 2: Specialized ontologies (selected per description)
    """
    
    def __init__(self, registry: OntologyRegistry):
        self.registry = registry
        self.encoder = registry.encoder
        self.specialized_profiles = {}
        
        self._build_specialized_profiles()
    
    def _build_specialized_profiles(self):
        """Build selection profiles for specialized ontologies."""
        for name, index in self.registry.ontologies.items():
            if index.category == "specialized":
                # Build fingerprint from key terms
                key_terms = [t.name for t in index.terms if t.depth <= 2][:50]
                fingerprint_text = " ".join(key_terms)
                
                self.specialized_profiles[name] = {
                    "fingerprint": self.encoder.encode(fingerprint_text),
                    "triggers": getattr(index, "trigger_terms", set())
                }
    
    def select_ontologies(self, 
                          description: str,
                          max_specialized: int = 2) -> list[str]:
        """Select relevant ontologies for a description."""
        
        # Always include base ontologies
        selected = [name for name, idx in self.registry.ontologies.items() 
                   if idx.category == "base"]
        
        # Score specialized ontologies
        if self.specialized_profiles:
            desc_embedding = self.encoder.encode(description)
            desc_lower = description.lower()
            
            scores = {}
            for name, profile in self.specialized_profiles.items():
                # Trigger-based score
                trigger_score = sum(1 for t in profile.get("triggers", []) 
                                   if t in desc_lower) * 2.0
                
                # Embedding similarity
                sim = np.dot(desc_embedding, profile["fingerprint"]) / (
                    np.linalg.norm(desc_embedding) * 
                    np.linalg.norm(profile["fingerprint"])
                )
                
                scores[name] = trigger_score + sim
            
            # Add top-scoring specialized ontologies
            sorted_specialized = sorted(scores.items(), 
                                        key=lambda x: x[1], reverse=True)
            for name, score in sorted_specialized[:max_specialized]:
                if score > 0.3:  # Minimum relevance threshold
                    selected.append(name)
        
        return selected
```

**Validation Criteria:**
- Correct ontology selection for known test cases (asco descriptions → asco ontology)
- No regression in base ontology performance
- Selection adds value (improved vocabulary consistency for specialized descriptions)

**Estimated Effort:** 1 day

---

### Phase 6: Graceful Degradation for Novel Terms (Conditional)

**Goal:** Handle terms not well-represented in ontologies without losing information.

**Prerequisite:** Phase 3 analysis shows significant vocabulary gaps that can't be addressed by adding ontologies.

**Key Principles:**
1. Never lose information
2. Never force bad mappings
3. Distinguish confidence levels
4. Preserve original terms with annotations when uncertain

**Implementation:** See Technical Reference for full `RobustOntologyPipeline` implementation.

**Estimated Effort:** 1-2 days (if needed)

---

## Implementation Timeline Summary

| Phase | Description | Prerequisites | Effort | Priority |
|-------|-------------|---------------|--------|----------|
| 1 | Extend schema depth (6 levels) | Phase 0 complete | 1-2 hours | High |
| 2 | Basic ontology integration (PATO + FAO) | Phase 1 | 1-2 days | High |
| 3 | Vocabulary coverage analysis tool | Phase 2 | 0.5-1 day | High |
| 4 | Vocabulary normalization | Phase 3 shows need | 0.5-1 day | Conditional |
| 5 | Tiered ontology selection | Phase 2 working well | 1 day | Medium |
| 6 | Graceful degradation | Phase 3 shows need | 1-2 days | Conditional |

**Minimum viable pipeline:** Phases 1-3 (~3-4 days)
**Full pipeline:** All phases (~5-8 days, depending on conditional phases)

---

## Technical Reference

### Obtaining Ontology Files

```bash
# PATO (Phenotype and Trait Ontology)
wget http://purl.obolibrary.org/obo/pato.obo

# FAO (Fungal Anatomy Ontology) - check OBO Foundry for current URL
# Alternative: use closest equivalent from OBO Foundry
wget http://purl.obolibrary.org/obo/fao.obo

# Additional ontologies (examples)
# Plant Ontology
wget http://purl.obolibrary.org/obo/po.obo

# Uberon (cross-species anatomy)
wget http://purl.obolibrary.org/obo/uberon.obo
```

### Memory Estimates

| Component | Size |
|-----------|------|
| PATO (~15k terms) embeddings | ~23 MB |
| FAO (~2k terms) embeddings | ~3 MB |
| SentenceTransformer model | ~90 MB |
| Total base memory | ~120 MB |

### Dependencies

```
pronto>=2.5.0          # Ontology parsing
sentence-transformers  # Text embeddings
numpy                  # Numerical operations
scikit-learn          # Clustering (if needed)
outlines              # Constrained decoding (if using)
```

---

## Appendices

### A. Failure Modes for Novel Vocabulary

| Failure Mode | Example | Mitigation |
|--------------|---------|------------|
| **False Synonymy** | "fibrillose" → "fibrous" (wrong: distinct terms) | Threshold-gated mapping |
| **Hierarchical Misplacement** | "subdecurrent" placed at wrong level | Depth inference from neighbors |
| **Term Loss** | "chrysocystidia" disappears | Preserve with annotations |

### B. Adding a New Ontology Checklist

1. Obtain .obo file from OBO Foundry or domain source
2. Register with `registry.register(name, path, category)`
3. If specialized, define trigger terms for selection
4. Run coverage analysis to validate improvement
5. Update prompt templates if needed

### C. Example Prompt with Ontology Context

```
Extract structured features from a biological species description.

ANATOMICAL STRUCTURES (use for top-level feature keys):
  [L1] fruiting body | basidioma
  [L2] pileus | stipe | hymenium
  [L3] lamellae | context | surface

QUALITY PROPERTIES (use for nested property keys):
  [L1] physical quality
  [L2] morphology | color | size
  [L3] shape | hue | length

HIERARCHY EXAMPLES:
  morphology > shape > curvature > convex
  morphology > shape > curvature > flat
  color > hue > brown
  physical quality > texture > surface texture > dry

RULES:
1. Use anatomical terms for top-level keys (Level 1)
2. Use quality types for Level 2 keys
3. Use quality subtypes for Level 3 keys
4. Leaf nodes contain arrays of observed values
5. Follow hierarchy patterns shown above

DESCRIPTION:
Pileus convex becoming flat, 3-5 cm broad, surface dry, brown to tan.
Lamellae adnate, crowded, white becoming cream.

OUTPUT (valid JSON only):
```

### D. Decision Tree: When to Implement Optional Phases

```
                    Run Phase 3 Coverage Analysis
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Well-covered ≥ 90%?  │
                   └──────────────────────┘
                      │              │
                     YES             NO
                      │              │
                      ▼              ▼
              Skip Phase 4    Implement Phase 4
              Skip Phase 6    (Normalization)
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Gaps from missing    │
                         │ ontologies or truly  │
                         │ novel terms?         │
                         └──────────────────────┘
                            │              │
                      MISSING ONT     NOVEL TERMS
                            │              │
                            ▼              ▼
                    Add ontologies   Implement Phase 6
                    to registry      (Graceful Degradation)
```
