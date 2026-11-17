#!/usr/bin/env python3
"""
Build Vocabulary Tree for UI Menus

This script reads JSON representations from CouchDB and organizes all vocabulary
into an efficient tree data structure for retrieving successive vocabularies
at each level. The tree is saved to Redis with a versioned key.

The tree structure:
- Level 1: All tokens appearing as top-level keys in JSON representations
- Level 2: Under each level1 token, all tokens appearing as level2 keys
- Continue recursively to the leaves (array values)

Usage:
    python bin/build_vocab_tree.py [options]

Examples:
    # Use defaults (skol_taxa_full_dev database)
    python bin/build_vocab_tree.py

    # Specify database
    python bin/build_vocab_tree.py --db skol_taxa_full_dev

    # Custom Redis connection
    python bin/build_vocab_tree.py --redis-host localhost --redis-port 6379

    # Dry run (don't save to Redis)
    python bin/build_vocab_tree.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import couchdb
import redis

# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import (  # noqa: E402
    common_parser, create_redis_client, get_env_config,
)


# ============================================================================
# Locking Constants (shared with Django views)
# ============================================================================

LOCK_KEY = 'skol:build:vocab_tree:lock'
LOCK_TTL = 360  # 6 minutes


_LATIN_PLURALS: Dict[str, str] = {
    'conidia': 'conidium',
    'asci': 'ascus',
    'hyphae': 'hypha',
    'basidia': 'basidium',
    'perithecia': 'perithecium',
    'apothecia': 'apothecium',
    'cystidia': 'cystidium',
    'sporangia': 'sporangium',
}
_PLURAL_SPARED_ENDINGS = ('ss', 'us', 'is', 'ies', 'ous')
_PLURAL_EXCEPTIONS = frozenset({'lens'})
_EDGE_QUOTES = '`"\''
_PRESENT = frozenset({'yes', 'true', 'present'})
_ABSENT = frozenset({'no', 'false', 'absent'})


def _unwrap(text: str) -> str:
    """Strip wrappers until nothing changes: whitespace, backticks and
    quotes, ``>...<``, a leading ``./``, ``:``, ``?`` or ``,``, and a
    trailing ``:`` or ``,``.  A trailing ``-`` or ``+`` is kept."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip().strip(_EDGE_QUOTES)
        if text.startswith('>'):
            text = text[1:]
        if text.endswith('<'):
            text = text[:-1]
        if text.startswith('./'):
            text = text[2:]
        text = text.lstrip(':?,').rstrip(':,')
    return text


def _singular(word: str) -> str:
    """Fold one word's plural to its singular."""
    if word in _LATIN_PLURALS:
        return _LATIN_PLURALS[word]
    if (len(word) > 3 and word.endswith('s')
            and word not in _PLURAL_EXCEPTIONS
            and not word.endswith(_PLURAL_SPARED_ENDINGS)):
        return word[:-1]
    return word


def normalize_key(term: Any) -> str:
    """Grouping key for a JSON key; never shown to users.

    Spellings with the same key are one menu entry, shown under the
    group's most frequent ``display_form`` (see VocabularyTree).  Folds:
    case and surrounding whitespace; wrappers (backticks, quotes,
    ``>...<``, a leading ``./``, ``:``, ``?`` or ``,``); ``_``, ``-`` and
    whitespace as equivalent separators; simple English plurals per word
    (sparing words ending in ss/us/is/ies/ous, words of three letters or
    fewer, and ``lens``); and Latin pairs (conidium/conidia, ascus/asci,
    hypha/hyphae, basidium/basidia, perithecium/perithecia,
    apothecium/apothecia, cystidium/cystidia, sporangium/sporangia).
    Keeps a trailing ``-`` or ``+``: IKI- and KOH+ are reaction results.
    Idempotent.
    """
    shown = display_form(term)
    # A hyphen followed by more of the word is a separator; one ending a
    # word (IKI-) is a reaction result and stays.
    spaced = re.sub(r'-(?=[^\s-])', ' ', shown)
    return ' '.join(_singular(word) for word in spaced.split())


def display_form(term: Any) -> str:
    """How one spelling of a key is shown: lowercase, trimmed, wrappers
    removed, ``_`` shown as a space; hyphens and plurals kept (choosing
    among spellings is the majority vote's job)."""
    text = _unwrap(str(term).lower())
    return ' '.join(text.replace('_', ' ').split())


def normalize_value(term: Any) -> str:
    """Menu form of a JSON leaf value.

    Lowercase; surrounding whitespace, backticks and quotes removed;
    inner punctuation kept (``thin-walled``, ``6-8 um``); yes/true/present
    become ``present`` and no/false/absent become ``absent``.  No plural
    folding.  Idempotent.
    """
    text = str(term).lower()
    previous = None
    while text != previous:
        previous = text
        text = text.strip().strip(_EDGE_QUOTES)
    if text in _PRESENT:
        return 'present'
    if text in _ABSENT:
        return 'absent'
    return text


COUNTS_SUFFIX = ":counts"


def counts_key(menus_key: str) -> str:
    """The Redis key of the path -> count map stored beside the tree at
    ``menus_key``.  Stub: Proposal 1 of docs/plans/vocab-tree-narrowing.md."""
    raise NotImplementedError


class VocabularyTree:
    """
    A tree data structure for organizing JSON vocabulary by levels.

    The tree supports efficient retrieval of vocabulary at any path:
    - get_children([]) returns all level-1 terms
    - get_children(["pileus"]) returns all level-2 terms under "pileus"
    - get_children(["pileus", "shape"]) returns all level-3 terms under pileus.shape

    Keys are stored under their grouping key (``normalize_key``) and
    leaf values under ``normalize_value``.  Everything the tree returns
    or serializes shows each key group under its most frequent
    ``display_form``, counted across the whole tree (ties go to the
    alphabetically first spelling), and every lookup accepts any
    spelling.

    Attributes:
        tree: Nested dict keyed by grouping key / normalized value
        stats: Statistics about the tree (depth, node counts, etc.)
        display_counts: Per key group, how often each display form occurred
        top_level_doc_counts: Per top-level key group, documents it is in
    """

    def __init__(self) -> None:
        """Initialize an empty vocabulary tree."""
        self.tree: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {
            "total_nodes": 0,
            "max_depth": 0,
            "level_counts": defaultdict(int),
            "leaf_count": 0,
        }
        self.display_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        # How many input documents each top-level key group appeared in.
        # Used by prune_top_level_singletons() to drop menu items that show
        # up in exactly one JSON description.
        self.top_level_doc_counts: Dict[str, int] = defaultdict(int)

    def _is_valid_term(self, term: str) -> bool:
        """
        Check if a term is valid vocabulary (not garbage/syntax).

        Filters out:
        - Punctuation-only strings
        - JSON syntax fragments
        - Very long strings (likely raw JSON or code)
        - Strings with too many special characters

        Args:
            term: The term to validate

        Returns:
            True if term is valid vocabulary
        """
        if not term or not term.strip():
            return False

        stripped = term.strip()

        # Skip very long strings (likely malformed JSON)
        if len(stripped) > 100:
            return False

        # Skip strings that look like raw JSON
        if stripped.startswith('{') or stripped.startswith('['):
            return False
        if '":' in stripped or '": ' in stripped:
            return False

        # Skip strings that are mostly punctuation/syntax
        if re.match(r'^[\[\]{}():,;\s\-\.\"\'`><=]+$', stripped):
            return False

        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', stripped):
            return False

        # Skip if more than 30% of chars are brackets/quotes
        syntax_chars = sum(1 for c in stripped if c in '[]{}()":,;')
        if len(stripped) > 0 and syntax_chars / len(stripped) > 0.3:
            return False

        return True

    def _key_group(self, key: Any) -> Optional[str]:
        """The grouping key for a JSON key, or None if it is not vocabulary."""
        shown = display_form(key)
        if not self._is_valid_term(shown):
            return None
        group = normalize_key(shown)
        return group or None

    def _display(self, group: str) -> str:
        """The spelling a group is shown under."""
        counts = self.display_counts.get(group)
        if not counts:
            return group
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def _node(self, path: List[str]) -> Dict[str, Any]:
        """The internal node at a path of internal keys, created on demand."""
        current = self.tree
        for p in path:
            current = current.setdefault(p, {})
        return current

    def _add_node(self, path: List[str], name: str, depth: int,
                  leaf: bool) -> None:
        """Add ``name`` under ``path`` if new, keeping stats."""
        current = self._node(path)
        if name not in current:
            current[name] = {}
            self.stats["total_nodes"] += 1
            self.stats["level_counts"][depth + 1] += 1
            if leaf:
                self.stats["leaf_count"] += 1

    def add_json(self, data: Any, path: Optional[List[str]] = None, depth: int = 0) -> None:
        """
        Add a JSON structure to the vocabulary tree.

        Recursively traverses the JSON and adds all keys and values to the tree
        at their appropriate positions.

        Args:
            data: JSON data (dict, list, or primitive)
            path: Current path in the tree, as internal keys (for recursion)
            depth: Current depth (for statistics)
        """
        if path is None:
            path = []

        if isinstance(data, dict):
            for key, value in data.items():
                group = self._key_group(key)
                if group is None:
                    continue
                self.display_counts[group][display_form(key)] += 1
                self._add_node(path, group, depth, leaf=False)
                self.add_json(value, path + [group], depth + 1)

        elif isinstance(data, list):
            for item in data:
                self.add_json(item, path, depth)

        elif isinstance(data, (str, bool)):
            value = normalize_value(data)
            if not self._is_valid_term(value):
                return
            self._add_node(path, value, depth, leaf=True)
            if depth + 1 > self.stats["max_depth"]:
                self.stats["max_depth"] = depth + 1

    def add_document_json(self, data: Any) -> None:
        """Add one document's JSON to the tree.

        Identical to ``add_json`` for tree-building purposes, but also
        increments ``top_level_doc_counts`` once for each valid top-level
        key group in the document so that ``prune_top_level_singletons``
        can later drop menu entries that appear in only one JSON
        description.

        For a top-level ``list``, each item that is itself a dict
        contributes its keys to the per-document counter — matching how
        ``add_json`` recurses into a top-level list.
        """
        groups = {self._key_group(k) for k in self._iter_top_level_keys(data)}
        for group in groups:
            if group is not None:
                self.top_level_doc_counts[group] += 1
        self.add_json(data)

    @staticmethod
    def _iter_top_level_keys(data: Any) -> Any:
        """Yield each top-level key contributed by one document."""
        if isinstance(data, dict):
            yield from data.keys()
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield from item.keys()

    def prune_top_level_singletons(self) -> List[str]:
        """Drop top-level keys that occurred in only one input document.

        Returns the sorted display names of the pruned keys so the caller
        can log what was removed.  Stats (total_nodes, max_depth,
        level_counts, leaf_count) are recomputed from the remaining tree.
        """
        pruned_groups = [
            group for group, count in self.top_level_doc_counts.items()
            if count <= 1 and group in self.tree
        ]
        pruned = sorted(self._display(g) for g in pruned_groups)
        for group in pruned_groups:
            del self.tree[group]
            # Drop the counter entry too so a second call is a no-op.
            self.top_level_doc_counts.pop(group, None)
        if pruned_groups:
            self._recompute_stats()
        return pruned

    def _recompute_stats(self) -> None:
        """Walk the tree from scratch to recompute the stats dictionary."""
        self.stats = {
            "total_nodes": 0,
            "max_depth": 0,
            "level_counts": defaultdict(int),
            "leaf_count": 0,
        }

        def _walk(subtree: Dict[str, Any], depth: int) -> None:
            for _key, children in subtree.items():
                self.stats["total_nodes"] += 1
                self.stats["level_counts"][depth] += 1
                if depth > self.stats["max_depth"]:
                    self.stats["max_depth"] = depth
                if isinstance(children, dict) and children:
                    _walk(children, depth + 1)
                else:
                    self.stats["leaf_count"] += 1

        _walk(self.tree, 1)

    def _find(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """The internal node at a path of any spellings, or None."""
        current = self.tree
        for p in path:
            for candidate in (normalize_key(p), normalize_value(p)):
                if candidate in current:
                    current = current[candidate]
                    break
            else:
                return None
        return current

    def _shown(self, subtree: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of an internal subtree keyed by display spellings."""
        shown: Dict[str, Any] = {}
        for name, children in subtree.items():
            merged = shown.setdefault(self._display(name), {})
            merged.update(self._shown(children))
        return shown

    def get_children(self, path: List[str]) -> List[str]:
        """
        Get all children (vocabulary) at the given path.

        Args:
            path: List of keys representing the path in the tree, in any
                  spelling.  Empty list returns top-level vocabulary.

        Returns:
            Sorted list of child display spellings at the given path
        """
        current = self._find(path)
        if current is None:
            return []
        return sorted({self._display(name) for name in current})

    def get_subtree(self, path: List[str]) -> Dict[str, Any]:
        """
        Get the subtree rooted at the given path.

        Args:
            path: List of keys representing the path, in any spelling

        Returns:
            Subtree dict keyed by display spellings, or empty dict if the
            path is not found
        """
        current = self._find(path)
        if current is None:
            return {}
        return self._shown(current)

    def get_support(self, path: List[str]) -> int:
        """How many documents contain ``path`` (any spelling); 0 if none.
        Stub: Proposal 1."""
        raise NotImplementedError

    def prune_below_support(self, min_support: int) -> int:
        """Drop every node, at every level, whose support is below
        ``min_support``, with its subtree.  Returns the number of nodes
        removed.  Stub: Proposal 1."""
        raise NotImplementedError

    def count_map(self) -> Dict[str, int]:
        """The path -> support map: each field is the JSON-encoded list of
        display spellings on the path.  Stub: Proposal 1."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tree to a dictionary for serialization.

        Returns:
            Dictionary with 'tree' (keyed by display spellings) and
            'stats' keys
        """
        return {
            "tree": self._shown(self.tree),
            "stats": {
                "total_nodes": self.stats["total_nodes"],
                "max_depth": self.stats["max_depth"],
                "level_counts": dict(self.stats["level_counts"]),
                "leaf_count": self.stats["leaf_count"],
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyTree":
        """
        Create a VocabularyTree from a dictionary.

        Nodes with children are regrouped as keys; childless nodes keep
        their spelling as values.

        Args:
            data: Dictionary with 'tree' and 'stats' keys

        Returns:
            VocabularyTree instance
        """
        tree = cls()

        def _load(saved: Dict[str, Any]) -> Dict[str, Any]:
            node: Dict[str, Any] = {}
            for name, children in saved.items():
                if isinstance(children, dict) and children:
                    group = normalize_key(name)
                    tree.display_counts[group][display_form(name)] += 1
                    node.setdefault(group, {}).update(_load(children))
                else:
                    node.setdefault(normalize_value(name), {})
            return node

        tree.tree = _load(data.get("tree", {}))
        stats = data.get("stats", {})
        tree.stats["total_nodes"] = stats.get("total_nodes", 0)
        tree.stats["max_depth"] = stats.get("max_depth", 0)
        tree.stats["level_counts"] = defaultdict(int, stats.get("level_counts", {}))
        tree.stats["leaf_count"] = stats.get("leaf_count", 0)
        return tree


def build_vocabulary_tree(
    couchdb_url: str,
    db_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    limit: Optional[int] = None,
    verbosity: int = 1
) -> VocabularyTree:
    """
    Build a vocabulary tree from JSON representations in CouchDB.

    Args:
        couchdb_url: CouchDB server URL
        db_name: Database name containing json_annotated documents
        username: CouchDB username
        password: CouchDB password
        limit: Maximum number of documents to process (None for all)
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns:
        VocabularyTree containing merged vocabulary from all documents
    """
    # Connect to CouchDB
    if verbosity >= 1:
        print(f"Connecting to CouchDB at {couchdb_url}...")

    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if db_name not in server:
        raise ValueError(f"Database '{db_name}' not found")

    db = server[db_name]

    # Build the tree
    tree = VocabularyTree()
    processed = 0
    skipped = 0

    if verbosity >= 1:
        print(f"Processing documents from {db_name}...")

    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue

        if limit and processed >= limit:
            break

        try:
            doc = db[doc_id]
            json_annotated = doc.get('json_annotated')

            if not json_annotated or not isinstance(json_annotated, dict):
                skipped += 1
                continue

            # Add this document's JSON to the tree
            tree.add_document_json(json_annotated)
            processed += 1

            if verbosity >= 2 and processed % 100 == 0:
                print(f"  Processed {processed} documents...")
            elif verbosity >= 1 and processed % 500 == 0:
                print(f"  Processed {processed} documents...")

        except Exception as e:
            if verbosity >= 2:
                print(f"  Warning: Error processing {doc_id}: {e}")
            skipped += 1

    # Drop top-level menu items that occurred in only a single JSON
    # description.  These are typically OCR noise or one-off labels and
    # clutter the cascading dropdowns without aiding navigation.
    pruned = tree.prune_top_level_singletons()
    if verbosity >= 1 and pruned:
        print(
            f"\nPruned {len(pruned)} top-level singleton "
            f"term(s) from the menu tree."
        )
        if verbosity >= 2:
            for term in pruned:
                print(f"  - {term}")

    if verbosity >= 1:
        print(f"\nProcessed {processed} documents, skipped {skipped}")
        print("Tree statistics:")
        print(f"  Total nodes: {tree.stats['total_nodes']}")
        print(f"  Max depth: {tree.stats['max_depth']}")
        print(f"  Leaf nodes: {tree.stats['leaf_count']}")
        for level, count in sorted(tree.stats['level_counts'].items()):
            print(f"  Level {level}: {count} terms")

    return tree


def save_to_redis(
    tree: VocabularyTree,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    version: Optional[str] = None,
    key_override: Optional[str] = None,
    ttl: Optional[int] = None,
    verbosity: int = 1
) -> str:
    """
    Save the vocabulary tree to Redis.

    Args:
        tree: VocabularyTree to save
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database number
        redis_password: Redis password (optional)
        version: Version string for the key (default: current timestamp)
        key_override: Full Redis key to use directly (overrides
                      version-based key).  Set from the experiment
                      record's redis_keys.menus when --experiment is
                      used.  No other key is written.
        ttl: Time-to-live in seconds (optional, None means no expiration)
        verbosity: Verbosity level

    Returns:
        The Redis key where the tree was saved
    """
    # Determine key: explicit override takes precedence over version-based key
    if key_override:
        key = key_override
        version = key_override  # for metadata
    else:
        # Generate version string
        if version is None:
            version = datetime.now().strftime("%Y_%m_%d_%H_%M")
        key = f"skol:ui:menus_{version}"

    # Connect to Redis (respects REDIS_TLS settings from env_config)
    if verbosity >= 1:
        print(f"\nConnecting to Redis at {redis_host}:{redis_port}...")

    r = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True
    )

    # Test connection
    try:
        r.ping()
    except redis.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Redis: {e}")

    # Serialize and save
    data = tree.to_dict()
    data["version"] = version
    data["created_at"] = datetime.now().isoformat()

    json_str = json.dumps(data, ensure_ascii=False)

    if verbosity >= 1:
        print(f"Saving tree to Redis key: {key}")
        print(f"  Data size: {len(json_str):,} bytes")
        if ttl:
            print(f"  TTL: {ttl} seconds ({ttl // 3600}h {(ttl % 3600) // 60}m)")

    # Save with optional TTL
    if ttl:
        r.set(key, json_str, ex=ttl)
    else:
        r.set(key, json_str)

    # Verify the data was actually saved
    if not r.exists(key):
        raise RuntimeError(f"Failed to save vocabulary tree to Redis key: {key}")

    # No shared "latest" pointer: the experiment record's
    # redis_keys.menus names the tree key (see
    # docs/plans/experiment-record-canonical.md).
    if ttl and verbosity >= 1:
        print(f"  WARNING: TTL is set. The tree will expire in {ttl}s.")
        print("           Run without --ttl for a persistent vocabulary tree.")

    return key


def load_from_redis(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    version: Optional[str] = None
) -> Tuple[VocabularyTree, Dict[str, Any]]:
    """
    Load a vocabulary tree from Redis.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database number
        redis_password: Redis password
        version: Version to load, ``skol:ui:menus_<version>``.  Required:
                 there is no shared "latest" pointer.

    Returns:
        Tuple of (VocabularyTree, metadata dict)
    """
    if not version:
        raise ValueError(
            "load_from_redis needs a version; there is no "
            "skol:ui:menus_latest pointer")
    r = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True
    )

    key = f"skol:ui:menus_{version}"

    # Load data
    json_str = r.get(key)
    if not json_str:
        raise ValueError(f"Key '{key}' not found in Redis")

    data = json.loads(json_str)
    tree = VocabularyTree.from_dict(data)

    metadata = {
        "key": key,
        "version": data.get("version"),
        "created_at": data.get("created_at"),
    }

    return tree, metadata


def acquire_lock(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    verbosity: int = 1
) -> Optional[redis.Redis]:
    """
    Acquire the build lock to prevent concurrent builds.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        verbosity: Verbosity level

    Returns:
        Redis client if lock acquired

    Raises:
        SystemExit: If lock cannot be acquired (another build in progress)
    """
    redis_client = create_redis_client(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )

    # Try to acquire lock (SETNX = SET if Not eXists)
    lock_acquired = redis_client.set(LOCK_KEY, 'building', nx=True, ex=LOCK_TTL)

    if not lock_acquired:
        if verbosity >= 1:
            print(f"✓ Another vocab tree build is already in progress (lock: {LOCK_KEY})")
            print("  Exiting gracefully. Try again later or check the other process.")
        sys.exit(0)  # Exit with success - not an error, just already running

    if verbosity >= 2:
        print(f"✓ Acquired build lock: {LOCK_KEY} (TTL: {LOCK_TTL}s)")

    client: redis.Redis = redis_client
    return client


def release_lock(redis_client: redis.Redis, verbosity: int = 1) -> None:
    """
    Release the build lock.

    Args:
        redis_client: Redis client
        verbosity: Verbosity level
    """
    try:
        redis_client.delete(LOCK_KEY)
        if verbosity >= 2:
            print(f"✓ Released build lock: {LOCK_KEY}")
    except Exception as e:
        # Don't fail if we can't release - TTL will handle it
        if verbosity >= 1:
            print(f"⚠ Could not release lock (will auto-expire): {e}")


def print_tree_sample(tree: VocabularyTree, max_items: int = 10) -> None:
    """
    Print a sample of the tree structure for verification.

    Args:
        tree: VocabularyTree to display
        max_items: Maximum items to show at each level
    """
    print("\n" + "=" * 70)
    print("VOCABULARY TREE SAMPLE")
    print("=" * 70)

    # Get top-level terms
    level1_terms = tree.get_children([])[:max_items]
    print(f"\nLevel 1 terms ({len(tree.get_children([]))} total, showing {len(level1_terms)}):")

    for term in level1_terms:
        level2_terms = tree.get_children([term])
        print(f"\n  {term} ({len(level2_terms)} children)")

        for child in level2_terms[:5]:
            level3_terms = tree.get_children([term, child])
            if level3_terms:
                sample = level3_terms[:3]
                more = f"... +{len(level3_terms) - 3} more" if len(level3_terms) > 3 else ""
                print(f"    {child}: [{', '.join(sample)}{more}]")
            else:
                print(f"    {child}: (leaf)")

        if len(level2_terms) > 5:
            print(f"    ... +{len(level2_terms) - 5} more children")

    if len(tree.get_children([])) > max_items:
        print(f"\n  ... +{len(tree.get_children([])) - max_items} more level-1 terms")

    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser: script flags plus the shared ``common_parser()``
    (``--experiment``, ``--dry-run``, ``--limit``, connection flags), so
    unknown flags are rejected."""
    parser = argparse.ArgumentParser(
        description="Build vocabulary tree from JSON representations "
                    "and save to Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parser()],
        epilog="""
Examples:
  %(prog)s                          # Use defaults
  %(prog)s --experiment production_v4_1   # Experiment's structured DB
  %(prog)s --db skol_taxa_full_dev  # Specify database
  %(prog)s --dry-run                # Build tree but don't save
  %(prog)s --show-sample            # Display tree sample after building
        """
    )

    # --couchdb-url/-password, --redis-host/-port/-password, --limit,
    # --dry-run, --verbosity and --experiment come from common_parser().
    parser.add_argument(
        '--db', '--database',
        default=None,
        help='CouchDB database to read json_annotated from (default: the '
             "experiment's structured treatments database, else "
             'skol_taxa_full_dev)'
    )
    parser.add_argument(
        '--couchdb-user',
        default=None,
        help='CouchDB username (default: from env_config or admin)'
    )
    parser.add_argument(
        '--redis-db',
        type=int,
        default=int(os.environ.get('REDIS_DB', '0')),
        help='Redis database number (default: $REDIS_DB or 0)'
    )
    parser.add_argument(
        '--version',
        default=None,
        help='Version string for Redis key '
             '(default: current timestamp YYYY_MM_DD_HH_MM)'
    )
    parser.add_argument(
        '--ttl',
        type=int,
        default=None,
        help='Time-to-live in seconds for Redis keys (default: no '
             'expiration). Examples: 3600 (1 hour), 86400 (1 day), '
             '604800 (1 week)'
    )
    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Display a sample of the tree structure'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Also save tree to a JSON file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (can be repeated)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output (same as --verbosity 0)'
    )
    return parser


def resolve_source_db(cli_db: Optional[str], config: Dict[str, Any]) -> str:
    """The database to read ``json_annotated`` from: ``--db``, else the
    experiment's structured treatments database, else
    ``skol_taxa_full_dev``.  Never the prose database, which has no
    ``json_annotated``."""
    return (cli_db
            or config.get('treatments_structured_db_name')
            or 'skol_taxa_full_dev')


def main():
    """Main entry point for the vocabulary tree builder."""
    args = build_parser().parse_args()

    # Load experiment/environment config; CLI flags win over it.
    config = get_env_config(cli_args=args)

    # Resolve connection args: CLI arg > env_config > hardcoded default.
    db = resolve_source_db(args.db, config)
    couchdb_url = config.get('couchdb_url') or 'http://localhost:5984'
    couchdb_user = (args.couchdb_user or config.get('couchdb_username')
                    or 'admin')
    couchdb_password = config.get('couchdb_password') or ''
    redis_host = config.get('redis_host') or 'localhost'
    redis_port = int(config.get('redis_port') or 6379)
    redis_password = config.get('redis_password')

    # When --experiment provides a menus key and --version was not explicitly
    # set, use the experiment key directly instead of constructing one from
    # --version (which would ignore the experiment's named key).
    menus_key_override: Optional[str] = None
    if config.get('menus_key') and args.version is None:
        menus_key_override = config['menus_key']

    # Set verbosity: --verbosity takes precedence, then -q, then -v
    if args.verbosity is not None:
        verbosity = args.verbosity
    elif args.quiet:
        verbosity = 0
    else:
        verbosity = 1 + args.verbose  # Base level 1, plus any -v flags

    # Acquire build lock (exits gracefully if another build is running)
    # Skip lock for dry-run since we're not actually saving anything
    lock_client = None
    if not args.dry_run:
        lock_client = acquire_lock(
            redis_host=redis_host,
            redis_port=redis_port,
            verbosity=verbosity
        )

    try:
        # Build the tree
        if verbosity >= 1:
            print("=" * 70)
            print("Building Vocabulary Tree")
            print("=" * 70)

        tree = build_vocabulary_tree(
            couchdb_url=couchdb_url,
            db_name=db,
            username=couchdb_user,
            password=couchdb_password,
            limit=args.limit,
            verbosity=verbosity
        )

        # Show sample if requested
        if args.show_sample:
            print_tree_sample(tree)

        # Save to JSON file if requested
        if args.output_json:
            if verbosity >= 1:
                print(f"\nSaving tree to {args.output_json}...")

            data = tree.to_dict()
            data["version"] = args.version or datetime.now().strftime("%Y_%m_%d_%H_%M")
            data["created_at"] = datetime.now().isoformat()
            data["source_db"] = db

            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if verbosity >= 1:
                print(f"  Saved to {args.output_json}")

        # Save to Redis unless dry run
        if not args.dry_run:
            redis_key = save_to_redis(
                tree=tree,
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=args.redis_db,
                redis_password=redis_password,
                version=args.version,
                key_override=menus_key_override,
                ttl=args.ttl,
                verbosity=verbosity
            )

            if verbosity >= 1:
                print(f"\nVocabulary tree saved to Redis: {redis_key}")
        else:
            if verbosity >= 1:
                print("\n[DRY RUN] Tree built but not saved to Redis")

        if verbosity >= 1:
            print("\nDone!")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # Always release the lock
        if lock_client:
            release_lock(lock_client, verbosity)


if __name__ == "__main__":
    sys.exit(main())
