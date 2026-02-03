"""Taxa JSON Classifier using TF-IDF on flattened JSON features.

This module builds a Decision Tree classifier for taxa identification
based on structured JSON annotations, using TF-IDF on flattened JSON tokens.

The json_annotated field contains structured taxonomic information that
gets flattened into tokens like:
    "taxon_name_genus=Aspergillus"
    "morphology_spore_shape=globose"
    "habitat_substrate_0=wood"

These tokens are then processed with TF-IDF to create feature vectors.

Example usage:

    from taxa_classifier import TaxaJsonClassifier

    classifier = TaxaJsonClassifier(
        couchdb_url='http://localhost:5984',
        database='skol_taxa_full_dev',
        username='admin',
        password='password'
    )

    # Train on all taxa
    classifier.fit(limit=1000)

    # Export decision tree
    tree_json = classifier.tree_to_json()
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


def flatten_json(y: Any) -> Dict[str, Any]:
    """Flatten a nested JSON structure into a flat dictionary.

    Nested keys are joined with underscores, list indices become numeric keys.

    Args:
        y: JSON-compatible Python object (dict, list, or primitive)

    Returns:
        Flat dictionary with compound keys

    Example:
        >>> flatten_json({'a': {'b': 1}, 'c': [2, 3]})
        {'a_b': 1, 'c_0': 2, 'c_1': 3}
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


def flatten_to_tokens(json_data: Any) -> str:
    """Flatten JSON and convert to space-separated tokens for TF-IDF.

    Each key-value pair becomes a token like "key=value".
    Spaces in keys and values are replaced with underscores to prevent
    token splitting by TF-IDF.

    Args:
        json_data: JSON-compatible Python object

    Returns:
        Space-separated string of tokens

    Example:
        >>> flatten_to_tokens({'genus': 'Aspergillus', 'color': 'light brown'})
        'genus=Aspergillus color=light_brown'
    """
    flat = flatten_json(json_data)
    tokens = []
    for k, v in flat.items():
        if v is not None:
            # Replace spaces with underscores to keep tokens intact
            key = str(k).replace(' ', '_')
            val = str(v).replace(' ', '_')
            tokens.append(f"{key}={val}")
    return ' '.join(tokens)


class TaxaJsonClassifier:
    """Classifier for taxa identification using TF-IDF on flattened JSON.

    This classifier takes structured JSON annotations, flattens them to
    key=value tokens, applies TF-IDF encoding, and uses a Decision Tree
    to classify taxa.

    Attributes:
        vectorizer: TfidfVectorizer for converting tokens to feature vectors
        classifier: Decision Tree classifier
        taxa_mapping: Dict mapping internal indices to taxon IDs
        reverse_mapping: Dict mapping taxon IDs to internal indices
    """

    def __init__(
        self,
        couchdb_url: Optional[str] = None,
        database: str = 'skol_taxa_full_dev',
        username: Optional[str] = None,
        password: Optional[str] = None,
        # TF-IDF parameters
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 1),
        # Decision Tree parameters
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        random_state: int = 42,
        verbosity: int = 1
    ):
        """Initialize the TaxaJsonClassifier.

        Args:
            couchdb_url: CouchDB server URL (default: from env_config)
            database: CouchDB database name (default: skol_taxa_full_dev)
            username: CouchDB username (default: from env_config)
            password: CouchDB password (default: from env_config)
            max_features: Maximum vocabulary size for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            ngram_range: N-gram range for TF-IDF
            max_depth: Maximum depth of decision tree (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            criterion: Split criterion ('gini' or 'entropy')
            random_state: Random state for reproducibility
            verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        # Load config from environment if not provided
        from env_config import get_env_config
        config = get_env_config()

        self.couchdb_url = couchdb_url or config.get('couchdb_url')
        self.database = database
        self.username = username or config.get('couchdb_username')
        self.password = password or config.get('couchdb_password')
        self.verbosity = verbosity

        # TF-IDF parameters
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

        # Decision Tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

        # Initialize models (will be created during fit)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[DecisionTreeClassifier] = None

        # Label mappings
        self.taxa_mapping: Dict[int, str] = {}
        self.reverse_mapping: Dict[str, int] = {}

        # Training statistics
        self._train_stats: Optional[Dict[str, Any]] = None

    def _connect_db(self):
        """Connect to CouchDB and return database handle."""
        import couchdb

        server = couchdb.Server(self.couchdb_url)
        if self.username and self.password:
            server.resource.credentials = (self.username, self.password)

        if self.database not in server:
            raise ValueError(f"Database '{self.database}' not found")

        return server[self.database]

    def fetch_taxa(
        self,
        taxa_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch taxa documents from CouchDB.

        Args:
            taxa_ids: List of specific taxon IDs to fetch. If None, fetches all.
            limit: Maximum number of documents to fetch

        Returns:
            List of taxa documents with _id and json_annotated fields
        """
        db = self._connect_db()
        documents = []

        if taxa_ids:
            if self.verbosity >= 1:
                print(f"Fetching {len(taxa_ids)} taxa documents...")

            for i, doc_id in enumerate(taxa_ids):
                try:
                    doc = db[doc_id]
                    json_annotated = doc.get('json_annotated')
                    if json_annotated:
                        documents.append({
                            '_id': doc['_id'],
                            'json_annotated': json_annotated
                        })
                    if self.verbosity >= 2 and (i + 1) % 100 == 0:
                        print(f"  Fetched {i + 1}/{len(taxa_ids)} documents")
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  Warning: Could not fetch {doc_id}: {e}")
        else:
            if self.verbosity >= 1:
                print(f"Fetching taxa documents from {self.database}...")

            count = 0
            for doc_id in db:
                if doc_id.startswith('_'):
                    continue

                try:
                    doc = db[doc_id]
                    json_annotated = doc.get('json_annotated')
                    if json_annotated:
                        documents.append({
                            '_id': doc['_id'],
                            'json_annotated': json_annotated
                        })
                        count += 1

                        if self.verbosity >= 2 and count % 100 == 0:
                            print(f"  Fetched {count} documents...")

                        if limit and count >= limit:
                            break
                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"  Warning: Could not fetch {doc_id}: {e}")

        if self.verbosity >= 1:
            print(f"Fetched {len(documents)} documents with json_annotated")

        return documents

    def _prepare_data(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Prepare token strings and labels from documents.

        Args:
            documents: List of taxa documents

        Returns:
            Tuple of (token_strings, taxa_ids)
        """
        token_strings = []
        taxa_ids = []

        for doc in documents:
            json_annotated = doc.get('json_annotated')
            if not json_annotated:
                continue

            # Parse JSON if it's a string
            if isinstance(json_annotated, str):
                try:
                    json_annotated = json.loads(json_annotated)
                except json.JSONDecodeError:
                    if self.verbosity >= 2:
                        print(f"  Warning: Invalid JSON for {doc['_id']}")
                    continue

            # Flatten to tokens
            tokens = flatten_to_tokens(json_annotated)

            if tokens:
                token_strings.append(tokens)
                taxa_ids.append(doc['_id'])

        return token_strings, taxa_ids

    def fit(
        self,
        taxa_ids: Optional[List[str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        test_size: float = 0.2,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the classifier on taxa JSON annotations.

        Args:
            taxa_ids: List of specific taxon IDs to train on
            documents: Pre-fetched documents (alternative to taxa_ids)
            test_size: Fraction of data for testing (0.0-1.0)
            limit: Maximum number of documents

        Returns:
            Dictionary with training statistics
        """
        if documents is None:
            documents = self.fetch_taxa(taxa_ids, limit=limit)

        if len(documents) < 2:
            raise ValueError(f"Need at least 2 documents, got {len(documents)}")

        if self.verbosity >= 1:
            print("Preparing training data...")

        token_strings, ids = self._prepare_data(documents)

        if len(token_strings) < 2:
            raise ValueError(f"Need at least 2 valid records, got {len(token_strings)}")

        # Create label mappings
        unique_ids = list(set(ids))
        self.taxa_mapping = {i: tid for i, tid in enumerate(unique_ids)}
        self.reverse_mapping = {tid: i for i, tid in enumerate(unique_ids)}

        y = np.array([self.reverse_mapping[tid] for tid in ids])

        if self.verbosity >= 1:
            print(f"Training on {len(token_strings)} samples, "
                  f"{len(unique_ids)} unique taxa")

        # Create and fit TF-IDF vectorizer
        if self.verbosity >= 1:
            print(f"Building TF-IDF vocabulary (max_features={self.max_features})...")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            token_pattern=r'[^\s]+'  # Tokens are separated by spaces
        )

        X = self.vectorizer.fit_transform(token_strings)

        if self.verbosity >= 1:
            print(f"TF-IDF matrix shape: {X.shape}")

        # Split data
        if test_size > 0 and len(unique_ids) > 1:
            stratify = y if len(unique_ids) <= len(y) * test_size else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=self.random_state, stratify=stratify
            )
        else:
            X_train, X_test, y_train, y_test = X, None, y, None

        # Train Decision Tree
        if self.verbosity >= 1:
            print(f"Training Decision Tree (max_depth={self.max_depth})...")

        self.classifier = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state
        )

        self.classifier.fit(X_train, y_train)

        # Calculate statistics
        stats = {
            'n_samples': len(token_strings),
            'n_classes': len(unique_ids),
            'n_features': X.shape[1],
            'tree_depth': self.classifier.get_depth(),
            'tree_n_leaves': self.classifier.get_n_leaves()
        }

        train_pred = self.classifier.predict(X_train)
        stats['train_accuracy'] = accuracy_score(y_train, train_pred)

        if X_test is not None:
            test_pred = self.classifier.predict(X_test)
            stats['test_accuracy'] = accuracy_score(y_test, test_pred)
            stats['test_size'] = len(y_test)

            if self.verbosity >= 1:
                print(f"\nTraining Results:")
                print(f"  Train accuracy: {stats['train_accuracy']:.4f}")
                print(f"  Test accuracy:  {stats['test_accuracy']:.4f}")
                print(f"  Tree depth:     {stats['tree_depth']}")
                print(f"  Tree leaves:    {stats['tree_n_leaves']}")

            if self.verbosity >= 2:
                y_test_labels = [self.taxa_mapping[i] for i in y_test]
                pred_labels = [self.taxa_mapping[i] for i in test_pred]
                report = classification_report(
                    y_test_labels, pred_labels, zero_division=0
                )
                print(f"\nClassification Report:\n{report}")
                stats['classification_report'] = report
        else:
            if self.verbosity >= 1:
                print(f"\nTraining Results:")
                print(f"  Train accuracy: {stats['train_accuracy']:.4f}")
                print(f"  Tree depth:     {stats['tree_depth']}")
                print(f"  Tree leaves:    {stats['tree_n_leaves']}")

        self._train_stats = stats
        return stats

    def predict(
        self,
        json_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Predict taxa IDs for given JSON data.

        Args:
            json_data: Single JSON dict or list of JSON dicts

        Returns:
            List of prediction results with predicted_id and confidence
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(json_data, dict):
            json_data = [json_data]

        # Convert to token strings
        token_strings = [flatten_to_tokens(data) for data in json_data]

        X = self.vectorizer.transform(token_strings)

        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)

        results = []
        for i, pred_idx in enumerate(predictions):
            probs = probabilities[i]
            top_indices = np.argsort(probs)[::-1]

            result = {
                'predicted_id': self.taxa_mapping[pred_idx],
                'confidence': float(probs[pred_idx]),
                'top_predictions': [
                    {
                        'taxa_id': self.taxa_mapping[idx],
                        'probability': float(probs[idx])
                    }
                    for idx in top_indices[:5]
                    if probs[idx] > 0.01
                ]
            }
            results.append(result)

        return results

    def get_feature_importances(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get the most important features for classification.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        importances = self.classifier.feature_importances_
        feature_names = self.vectorizer.get_feature_names_out()

        top_indices = np.argsort(importances)[::-1][:top_n]

        return [
            (feature_names[idx], float(importances[idx]))
            for idx in top_indices
            if importances[idx] > 0
        ]

    def tree_to_json(
        self,
        max_depth: Optional[int] = None,
        min_samples: int = 1,
        include_samples: bool = True
    ) -> Dict[str, Any]:
        """Extract the decision tree structure as JSON.

        Args:
            max_depth: Maximum depth to export (None for full tree)
            min_samples: Minimum samples at a node to include it
            include_samples: Include sample counts at each node

        Returns:
            JSON-serializable dict representing the decision tree
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        tree = self.classifier.tree_
        feature_names = self.vectorizer.get_feature_names_out()

        def build_node(node_id: int, depth: int = 0) -> Optional[Dict[str, Any]]:
            if max_depth is not None and depth > max_depth:
                return None

            n_samples = int(tree.n_node_samples[node_id])
            if n_samples < min_samples:
                return None

            class_counts = tree.value[node_id][0]
            majority_idx = int(np.argmax(class_counts))
            majority_count = int(class_counts[majority_idx])
            confidence = majority_count / n_samples if n_samples > 0 else 0

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            is_leaf = left_child == right_child

            if is_leaf:
                node = {
                    'type': 'leaf',
                    'prediction': self.taxa_mapping.get(
                        majority_idx, f'class_{majority_idx}'
                    ),
                    'confidence': round(confidence, 4)
                }
                if include_samples:
                    node['samples'] = n_samples
                return node

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            if 0 <= feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
            else:
                feature_name = f'feature_{feature_idx}'

            # Parse feature name (format: "key=value")
            if '=' in feature_name:
                key, value = feature_name.split('=', 1)
                question = f'Is "{key}" = "{value}"?'
            else:
                question = f'Is "{feature_name}" present?'

            node = {
                'type': 'decision',
                'feature': feature_name,
                'threshold': round(float(threshold), 6),
                'question': question
            }

            if include_samples:
                node['samples'] = n_samples

            node['no'] = build_node(left_child, depth + 1)
            node['yes'] = build_node(right_child, depth + 1)

            return node

        tree_structure = build_node(0)

        return {
            'metadata': {
                'n_classes': int(len(self.taxa_mapping)),
                'n_features': int(len(feature_names)),
                'tree_depth': int(self.classifier.get_depth()),
                'tree_leaves': int(self.classifier.get_n_leaves()),
                'exported_max_depth': max_depth
            },
            'tree': tree_structure
        }

    def tree_to_rules(
        self,
        max_rules: int = 100,
        min_confidence: float = 0.5,
        min_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """Extract decision rules from the tree as a list.

        Args:
            max_rules: Maximum number of rules to return
            min_confidence: Minimum confidence for a rule
            min_samples: Minimum samples at leaf

        Returns:
            List of rules with conditions, prediction, confidence
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        tree = self.classifier.tree_
        feature_names = self.vectorizer.get_feature_names_out()
        rules: List[Dict[str, Any]] = []

        def extract_rules(node_id: int, conditions: List[str]):
            if len(rules) >= max_rules:
                return

            n_samples = int(tree.n_node_samples[node_id])
            if n_samples < min_samples:
                return

            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            is_leaf = left_child == right_child

            if is_leaf:
                class_counts = tree.value[node_id][0]
                majority_idx = int(np.argmax(class_counts))
                majority_count = int(class_counts[majority_idx])
                confidence = majority_count / n_samples if n_samples > 0 else 0

                if confidence >= min_confidence:
                    rules.append({
                        'conditions': conditions.copy() or ['(no conditions)'],
                        'prediction': self.taxa_mapping.get(
                            majority_idx, f'class_{majority_idx}'
                        ),
                        'confidence': round(confidence, 4),
                        'samples': n_samples
                    })
            else:
                feature_idx = tree.feature[node_id]

                if 0 <= feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                else:
                    feature_name = f'feature_{feature_idx}'

                if '=' in feature_name:
                    key, value = feature_name.split('=', 1)
                    left_cond = f'"{key}" ≠ "{value}"'
                    right_cond = f'"{key}" = "{value}"'
                else:
                    left_cond = f'"{feature_name}" absent'
                    right_cond = f'"{feature_name}" present'

                extract_rules(left_child, conditions + [left_cond])
                extract_rules(right_child, conditions + [right_cond])

        extract_rules(0, [])
        rules.sort(key=lambda r: (-r['confidence'], -r['samples']))

        return rules[:max_rules]

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'taxa_mapping': self.taxa_mapping,
            'reverse_mapping': self.reverse_mapping,
            'train_stats': self._train_stats,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'criterion': self.criterion,
                'random_state': self.random_state
            },
            'version': '1.0'
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        if self.verbosity >= 1:
            print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.taxa_mapping = model_data['taxa_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        self._train_stats = model_data.get('train_stats')

        config = model_data.get('config', {})
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.verbosity >= 1:
            print(f"Model loaded from {filepath}")


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='TF-IDF + Decision Tree classifier on flattened JSON'
    )
    parser.add_argument('--database', default='skol_taxa_full_dev')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--max-features', type=int, default=5000)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--save-model', type=str, default=None)
    parser.add_argument('--export-tree', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='count', default=1)

    args = parser.parse_args()

    classifier = TaxaJsonClassifier(
        database=args.database,
        max_depth=args.max_depth,
        max_features=args.max_features,
        verbosity=args.verbose
    )

    classifier.fit(limit=args.limit, test_size=args.test_size)

    if args.verbose >= 2:
        print("\nTop Feature Importances:")
        for feat, imp in classifier.get_feature_importances(20):
            print(f"  {feat}: {imp:.4f}")

    if args.save_model:
        classifier.save_model(args.save_model)

    if args.export_tree:
        tree_json = classifier.tree_to_json()
        with open(args.export_tree, 'w') as f:
            json.dump(tree_json, f, indent=2)
        print(f"Tree exported to {args.export_tree}")


if __name__ == '__main__':
    main()
