# Optimizing a Mistral-Based SLM for Automated Synoptic Key Generation

## Problem Statement

The goal is to develop a general framework for building synoptic keys from species descriptions, starting with mycology and expanding to other fields of biology. The system uses a Mistral-based small language model to parse species descriptions into structured JSON.

### Hardware Constraints
- Single RTX 5090 GPU with 24GB VRAM
- 96GB host RAM
- Preference for computations completing within 1-2 days

### Design Constraints
- **No manual intervention**: Any step requiring human data-specific decisions is not acceptable
- **Domain-agnostic**: Must generalize across biological domains (mycology → botany → entomology, etc.)

### Current Problems

1. **Low structural success rate**: Less than 50% of generations produce valid JSON (most return `{}`)
2. **Schema violations**: Leaf nodes are sometimes single values instead of required lists
3. **Vocabulary inconsistency**: Features appear at different hierarchy levels; synonyms are used inconsistently

### Required Output Format
Nested dictionaries with configurable depth (2-4 levels), with string arrays at leaf nodes:

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

## Analysis of Initially Proposed Approaches

### 1. Fine-tuning on Valid Examples

**Assessment: Partially addresses Problem 1, unlikely to help with 2 & 3**

Concerns:
- Discards ~50% of data
- May select for *easier* descriptions rather than teaching the model to handle *harder* ones
- Doesn't address schema enforcement directly
- Vocabulary inconsistency persists in "good" examples

**If pursued:** Use LoRA/QLoRA rather than full fine-tuning. Mistral 7B with 4-bit quantization + LoRA adapters fits comfortably on 24GB. Training time: hours, not days.

### 2. Reinforcement Learning

**Assessment: High effort, uncertain payoff**

Concerns:
- Reward hacking risk—model may find degenerate solutions
- Length bias requires careful reward normalization
- PPO training loops are computationally expensive and difficult to stabilize
- Would likely take multiple days with significant iteration

**Alternative:** If reward-based optimization is desired, consider DPO (Direct Preference Optimization)—more stable, no separate reward model required.

### 3. BERT Clustering for Vocabulary Normalization

**Assessment: Tractable computation, but post-hoc**

The computational concern is overstated:
```python
# Rough compute estimate
# 10,000 terms × 768 dimensions = ~30MB
# K-means with k=500 clusters: seconds to minutes
# Hierarchical clustering: minutes
# HDBSCAN: minutes
```

The real limitation is that this is post-processing—it doesn't teach the model to generate consistent vocabulary, it cleans up afterward. However, it can be part of a viable automated pipeline.

---

## Recommended Approaches

### Constrained Decoding (Solves Problems 1 & 2)

Tools like **Outlines**, **guidance**, or **llama.cpp grammars** force the model to generate valid JSON conforming to a specific schema. This solves structural problems by construction with zero training cost.

#### Variable-Depth Schema Definition

JSON Schema supports the required "2-4 levels with arrays at leaves" pattern:

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

At each level, the model can choose to terminate with an array or continue nesting. This explicit unrolling is more reliable across tools than true recursion.

#### Basic Outlines Usage

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.json(model, schema)
result = generator(prompt)  # Guaranteed valid JSON matching schema
```

---

## Fully Automated Vocabulary Normalization (Solves Problem 3)

### Approach 1: Frequency-Based Canonical Selection

After clustering, the most frequent term in each cluster becomes canonical. No human decisions required.

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

def build_canonical_mapping(terms_with_counts: dict[str, int], 
                            similarity_threshold: float = 0.85) -> dict[str, str]:
    """
    terms_with_counts: {"cap shape": 500, "pileus shape": 50, "cap form": 30, ...}
    Returns: {"pileus shape": "cap shape", "cap form": "cap shape", ...}
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    terms = list(terms_with_counts.keys())
    embeddings = model.encode(terms)
    
    # Cluster based on semantic similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    
    # For each cluster, pick the most frequent term as canonical
    canonical_map = {}
    for cluster_id in set(labels):
        cluster_terms = [t for t, l in zip(terms, labels) if l == cluster_id]
        canonical = max(cluster_terms, key=lambda t: terms_with_counts[t])
        for term in cluster_terms:
            canonical_map[term] = canonical
    
    return canonical_map
```

**Assumption:** The "correct" term is used most often—reasonable for scientific descriptions.

### Approach 2: External Ontology Alignment

Biology has rich, curated ontologies that provide grounded vocabulary:

| Domain | Ontologies |
|--------|-----------|
| General traits | PATO (Phenotype and Trait Ontology) |
| Fungi | MycoBank, Index Fungorum, FungiDB |
| Plants | Plant Ontology (PO), Plant Trait Ontology |
| Animals | Uberon (anatomy), VT (vertebrate traits) |
| Cross-cutting | Gene Ontology, NCBI Taxonomy |

Map extracted terms to nearest neighbors in the ontology:

```python
from sklearn.metrics.pairwise import cosine_similarity

class OntologyNormalizer:
    def __init__(self, ontology_terms: list[str]):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ontology_terms = ontology_terms
        self.ontology_embeddings = self.model.encode(ontology_terms)
        
    def normalize(self, extracted_term: str, threshold: float = 0.7) -> str:
        """Map extracted term to nearest ontology term, or keep original if no good match."""
        embedding = self.model.encode([extracted_term])[0]
        similarities = cosine_similarity([embedding], self.ontology_embeddings)[0]
        best_idx = similarities.argmax()
        
        if similarities[best_idx] >= threshold:
            return self.ontology_terms[best_idx]
        return extracted_term  # No good match, keep original
```

**Benefits:**
- Vocabulary grounded in established scientific terminology
- Works across domains (just swap the ontology)
- Hierarchical structure of ontologies can inform nesting decisions
- Zero manual work per domain
- PATO is particularly useful as cross-domain foundation (describes color, shape, texture, size)

### Approach 3: Self-Consistency Voting

Generate multiple outputs per description, then vote on vocabulary:

```python
from collections import defaultdict, Counter

def extract_with_voting(description: str, generator, n_samples: int = 7):
    outputs = [generator(description) for _ in range(n_samples)]
    valid_jsons = [json.loads(o) for o in outputs if is_valid(o)]
    
    # Collect all keys at each path
    key_votes = defaultdict(Counter)  # path -> Counter of keys
    for j in valid_jsons:
        for path, key in extract_all_keys_with_paths(j):
            key_votes[path][key] += 1
    
    # Build canonical structure using majority keys
    # ... (alignment logic using embeddings for fuzzy matching)
```

Variance in generation becomes signal for identifying stable vocabulary.

### Approach 4: Hierarchical Consistency Enforcement

Enforce that terms used at level N in one description must be at level N (or unused) in all descriptions:

```python
class HierarchyEnforcer:
    def __init__(self):
        self.term_levels = {}  # term -> set of levels it's appeared at
        
    def update(self, json_output: dict, path_depth: int = 0):
        for key, value in json_output.items():
            canonical_key = self.normalize(key)  # your normalization function
            
            if canonical_key in self.term_levels:
                # Term seen before—should be at same level
                expected_levels = self.term_levels[canonical_key]
                if path_depth not in expected_levels:
                    # Inconsistency detected—could reject, or update model
                    pass
            else:
                self.term_levels[canonical_key] = {path_depth}
            
            if isinstance(value, dict):
                self.update(value, path_depth + 1)
    
    def get_terms_for_level(self, level: int) -> set[str]:
        return {t for t, levels in self.term_levels.items() if level in levels}
```

Build level-specific vocabularies over time, then constrain generation to level-appropriate terms.

---

## Proposed Fully-Automated Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     BOOTSTRAP PHASE (once per domain)           │
├─────────────────────────────────────────────────────────────────┤
│  1. Run constrained generation on N descriptions                │
│  2. Collect all (path, key) pairs from valid outputs            │
│  3. Cluster keys by embedding similarity                        │
│  4. Select canonical term per cluster (frequency-based)         │
│  5. Optionally: align to external ontology (PATO, domain-specific)│
│  6. Build level-specific vocabulary constraints                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Constrained generation (schema + vocabulary constraints)    │
│  2. Post-hoc normalization (map any novel terms to nearest      │
│     canonical term)                                             │
│  3. Validate hierarchy consistency                              │
│  4. Periodically update canonical vocabulary as corpus grows    │
└─────────────────────────────────────────────────────────────────┘
```

### Adapting to New Domains

When moving to a new domain (botany, entomology, etc.):

1. Point the system at a new corpus
2. Optionally provide a domain ontology (PATO works as a fallback)
3. Let it bootstrap vocabulary automatically

No manual intervention required.

---

## Implementation Phases

### Phase 1: Immediate Wins (Hours)
1. Implement constrained decoding with Outlines or llama.cpp grammars
2. Use the variable-depth schema definition above
3. Improve prompts with explicit schema definition and 2-3 few-shot examples

**Expected outcome:** 100% valid JSON, correct structure

### Phase 2: Vocabulary Normalization (1-2 Days)
1. Build canonical vocabulary using BERT embeddings + clustering
2. Extract all terms from valid outputs
3. Embed with `sentence-transformers`
4. Cluster with agglomerative clustering or HDBSCAN
5. Auto-select canonical terms by frequency
6. Optionally align to PATO or domain-specific ontology

**Expected outcome:** Consistent vocabulary across outputs

### Phase 3: Optional Fine-tuning (1-2 Days)
If results still need improvement:
1. QLoRA fine-tune on cleaned, vocabulary-normalized examples
2. Use canonical vocabulary in all training examples
3. ~1000-5000 examples should be sufficient
4. Training time: 2-8 hours on RTX 5090

**Expected outcome:** Model naturally produces consistent vocabulary

---

## Key Libraries and Resources

### Constrained Decoding
- [Outlines](https://github.com/outlines-dev/outlines) - Python library for structured generation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GBNF grammars for constrained generation
- [guidance](https://github.com/guidance-ai/guidance) - Microsoft's structured generation library

### Embeddings and Clustering
- [sentence-transformers](https://www.sbert.net/) - Fast, high-quality text embeddings
- `all-MiniLM-L6-v2` - Good balance of speed and quality
- scikit-learn - AgglomerativeClustering, HDBSCAN

### Biological Ontologies
- [PATO](https://github.com/pato-ontology/pato) - Phenotype and Trait Ontology
- [OBO Foundry](http://www.obofoundry.org/) - Collection of biological ontologies
- [BioPortal](https://bioportal.bioontology.org/) - Ontology search and access

### Fine-tuning
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning (LoRA, QLoRA)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit quantization
