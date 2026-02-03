"""Taxa Decision Tree Classifier using TF-IDF encoding.

This module builds a Decision Tree classifier for taxa identification
based on description text, using TF-IDF vectorization for text encoding.

Example usage:

    from taxa_classifier import TaxaDecisionTreeClassifier

    # Initialize with CouchDB connection
    classifier = TaxaDecisionTreeClassifier(
        couchdb_url='http://localhost:5984',
        database='skol_taxa_dev',
        username='admin',
        password='password'
    )

    # Train on specific taxa IDs
    taxa_ids = ['taxon_001...', 'taxon_002...']
    classifier.fit(taxa_ids)

    # Predict taxa from new descriptions
    predictions = classifier.predict(['A fungus with brown spores...'])

    # Save/load model
    classifier.save_model('taxa_model.pkl')
    classifier.load_model('taxa_model.pkl')
"""

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


class TaxaDecisionTreeClassifier:
    """Classifier for taxa identification using TF-IDF + Decision Tree.

    This classifier takes taxa descriptions and learns to identify which
    taxon a description belongs to based on its text content.

    Attributes:
        vectorizer: TF-IDF vectorizer for text encoding
        classifier: Decision Tree classifier
        taxa_mapping: Dict mapping internal indices to taxon IDs
        reverse_mapping: Dict mapping taxon IDs to internal indices
    """

    def __init__(
        self,
        couchdb_url: Optional[str] = None,
        database: str = 'skol_taxa_dev',
        username: Optional[str] = None,
        password: Optional[str] = None,
        # TF-IDF parameters
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        # Decision Tree parameters
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        random_state: int = 42,
        verbosity: int = 1
    ):
        """Initialize the TaxaDecisionTreeClassifier.

        Args:
            couchdb_url: CouchDB server URL (default: from env_config)
            database: CouchDB database name
            username: CouchDB username (default: from env_config)
            password: CouchDB password (default: from env_config)
            max_features: Maximum vocabulary size for TF-IDF
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            ngram_range: N-gram range for TF-IDF (e.g., (1, 2) for unigrams and bigrams)
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

        self.couchdb_url = couchdb_url or config.get('couchdb_url', 'http://localhost:5984')
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
        self.taxa_mapping: Dict[int, str] = {}  # index -> taxa_id
        self.reverse_mapping: Dict[str, int] = {}  # taxa_id -> index

        # Training statistics
        self._train_stats: Optional[Dict[str, Any]] = None

    def _connect_db(self):
        """Connect to CouchDB and return database handle."""
        import couchdb

        server = couchdb.Server(self.couchdb_url)
        if self.username and self.password:
            server.resource.credentials = (self.username, self.password)

        if self.database not in server:
            raise ValueError(f"Database '{self.database}' not found in CouchDB")

        return server[self.database]

    def fetch_taxa(
        self,
        taxa_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch taxa documents from CouchDB.

        Args:
            taxa_ids: List of specific taxon IDs to fetch. If None, fetches all.
            limit: Maximum number of documents to fetch (applies when taxa_ids is None)

        Returns:
            List of taxa documents with _id, taxon, and description fields
        """
        db = self._connect_db()
        documents = []

        if taxa_ids:
            # Fetch specific documents
            if self.verbosity >= 1:
                print(f"Fetching {len(taxa_ids)} taxa documents...")

            for i, doc_id in enumerate(taxa_ids):
                try:
                    doc = db[doc_id]
                    documents.append({
                        '_id': doc['_id'],
                        'taxon': doc.get('taxon', ''),
                        'description': doc.get('description', '')
                    })
                    if self.verbosity >= 2 and (i + 1) % 100 == 0:
                        print(f"  Fetched {i + 1}/{len(taxa_ids)} documents")
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  Warning: Could not fetch {doc_id}: {e}")
        else:
            # Fetch all documents
            if self.verbosity >= 1:
                print(f"Fetching all taxa documents from {self.database}...")

            count = 0
            for doc_id in db:
                if doc_id.startswith('_'):
                    continue

                try:
                    doc = db[doc_id]
                    # Only include documents with description field
                    if 'description' in doc and doc.get('description'):
                        documents.append({
                            '_id': doc['_id'],
                            'taxon': doc.get('taxon', ''),
                            'description': doc.get('description', '')
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
            print(f"Fetched {len(documents)} taxa documents")

        return documents

    def _prepare_data(
        self,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Prepare descriptions and labels from documents.

        Args:
            documents: List of taxa documents

        Returns:
            Tuple of (descriptions, taxa_ids)
        """
        descriptions = []
        taxa_ids = []

        for doc in documents:
            desc = doc.get('description', '')
            taxon = doc.get('taxon', '')

            # Combine taxon name and description for richer features
            text = f"{taxon} {desc}".strip()

            if text:
                descriptions.append(text)
                taxa_ids.append(doc['_id'])

        return descriptions, taxa_ids

    def fit(
        self,
        taxa_ids: Optional[List[str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        test_size: float = 0.2,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the classifier on taxa descriptions.

        Args:
            taxa_ids: List of specific taxon IDs to train on. If None, uses all.
            documents: Pre-fetched documents (alternative to taxa_ids)
            test_size: Fraction of data to use for testing (0.0-1.0)
            limit: Maximum number of documents to use (when taxa_ids is None)

        Returns:
            Dictionary with training statistics including accuracy, classification report
        """
        # Fetch documents if not provided
        if documents is None:
            documents = self.fetch_taxa(taxa_ids, limit=limit)

        if len(documents) < 2:
            raise ValueError(f"Need at least 2 taxa documents for training, got {len(documents)}")

        # Prepare data
        if self.verbosity >= 1:
            print("Preparing training data...")

        descriptions, ids = self._prepare_data(documents)

        if len(descriptions) < 2:
            raise ValueError(f"Need at least 2 valid descriptions, got {len(descriptions)}")

        # Create label mappings
        unique_ids = list(set(ids))
        self.taxa_mapping = {i: taxa_id for i, taxa_id in enumerate(unique_ids)}
        self.reverse_mapping = {taxa_id: i for i, taxa_id in enumerate(unique_ids)}

        # Convert taxa_ids to numeric labels
        y = np.array([self.reverse_mapping[taxa_id] for taxa_id in ids])

        if self.verbosity >= 1:
            print(f"Training on {len(descriptions)} descriptions from {len(unique_ids)} unique taxa")

        # Create and fit TF-IDF vectorizer
        if self.verbosity >= 1:
            print(f"Building TF-IDF vocabulary (max_features={self.max_features}, ngrams={self.ngram_range})...")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents='unicode'
        )

        X = self.vectorizer.fit_transform(descriptions)

        if self.verbosity >= 1:
            print(f"TF-IDF matrix shape: {X.shape}")

        # Split data for evaluation
        if test_size > 0 and len(unique_ids) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y if len(unique_ids) <= len(y) * test_size else None
            )
        else:
            X_train, X_test, y_train, y_test = X, None, y, None

        # Create and fit Decision Tree classifier
        if self.verbosity >= 1:
            tree_params = f"max_depth={self.max_depth}, criterion={self.criterion}"
            print(f"Training Decision Tree classifier ({tree_params})...")

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
            'n_samples': len(descriptions),
            'n_classes': len(unique_ids),
            'n_features': X.shape[1],
            'tree_depth': self.classifier.get_depth(),
            'tree_n_leaves': self.classifier.get_n_leaves()
        }

        # Training accuracy
        train_predictions = self.classifier.predict(X_train)
        stats['train_accuracy'] = accuracy_score(y_train, train_predictions)

        # Test accuracy (if we have test data)
        if X_test is not None:
            test_predictions = self.classifier.predict(X_test)
            stats['test_accuracy'] = accuracy_score(y_test, test_predictions)
            stats['test_size'] = len(y_test)

            if self.verbosity >= 1:
                print(f"\nTraining Results:")
                print(f"  Train accuracy: {stats['train_accuracy']:.4f}")
                print(f"  Test accuracy:  {stats['test_accuracy']:.4f}")
                print(f"  Tree depth:     {stats['tree_depth']}")
                print(f"  Tree leaves:    {stats['tree_n_leaves']}")

            if self.verbosity >= 2:
                # Get string labels for classification report
                y_test_labels = [self.taxa_mapping[idx] for idx in y_test]
                pred_labels = [self.taxa_mapping[idx] for idx in test_predictions]
                report = classification_report(y_test_labels, pred_labels, zero_division=0)
                print(f"\nClassification Report:\n{report}")
                stats['classification_report'] = report
        else:
            if self.verbosity >= 1:
                print(f"\nTraining Results (no test split):")
                print(f"  Train accuracy: {stats['train_accuracy']:.4f}")
                print(f"  Tree depth:     {stats['tree_depth']}")
                print(f"  Tree leaves:    {stats['tree_n_leaves']}")

        self._train_stats = stats
        return stats

    def predict(
        self,
        descriptions: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Predict taxa IDs for given descriptions.

        Args:
            descriptions: Single description or list of descriptions

        Returns:
            List of dicts with 'predicted_id', 'confidence', and 'probabilities'
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Handle single description
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # Transform descriptions to TF-IDF features
        X = self.vectorizer.transform(descriptions)

        # Get predictions and probabilities
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)

        results = []
        for i, pred_idx in enumerate(predictions):
            # Get top-k predictions by probability
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
                    if probs[idx] > 0.01  # Only include if probability > 1%
                ]
            }
            results.append(result)

        return results

    def predict_taxa_id(
        self,
        description: str
    ) -> str:
        """Predict a single taxa ID for a description.

        Args:
            description: Description text

        Returns:
            Predicted taxa ID string
        """
        results = self.predict(description)
        return results[0]['predicted_id']

    def get_feature_importances(
        self,
        top_n: int = 20
    ) -> List[Tuple[str, float]]:
        """Get the most important features (words/ngrams) for classification.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")

        importances = self.classifier.feature_importances_
        feature_names = self.vectorizer.get_feature_names_out()

        # Get top features by importance
        top_indices = np.argsort(importances)[::-1][:top_n]

        return [
            (feature_names[idx], float(importances[idx]))
            for idx in top_indices
            if importances[idx] > 0
        ]

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
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
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.taxa_mapping = model_data['taxa_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        self._train_stats = model_data.get('train_stats')

        # Optionally restore config
        config = model_data.get('config', {})
        if config:
            self.max_features = config.get('max_features', self.max_features)
            self.min_df = config.get('min_df', self.min_df)
            self.max_df = config.get('max_df', self.max_df)
            self.ngram_range = config.get('ngram_range', self.ngram_range)
            self.max_depth = config.get('max_depth', self.max_depth)
            self.min_samples_split = config.get('min_samples_split', self.min_samples_split)
            self.min_samples_leaf = config.get('min_samples_leaf', self.min_samples_leaf)
            self.criterion = config.get('criterion', self.criterion)
            self.random_state = config.get('random_state', self.random_state)

        if self.verbosity >= 1:
            print(f"Model loaded from {filepath}")
            if self._train_stats:
                print(f"  Classes: {self._train_stats.get('n_classes', 'unknown')}")
                print(f"  Features: {self._train_stats.get('n_features', 'unknown')}")

    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the last training run.

        Returns:
            Dictionary with training statistics, or None if not trained
        """
        return self._train_stats


def main():
    """Command-line interface for the taxa classifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train and use a TF-IDF + Decision Tree classifier for taxa identification'
    )

    parser.add_argument(
        '--database',
        default='skol_taxa_dev',
        help='CouchDB database name'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to fetch'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum vocabulary size for TF-IDF'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum depth of decision tree (None for unlimited)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing'
    )

    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained model'
    )

    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to load pre-trained model'
    )

    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help='Description text to predict taxa for'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=1,
        help='Increase verbosity (can be used multiple times)'
    )

    args = parser.parse_args()

    # Initialize classifier
    classifier = TaxaDecisionTreeClassifier(
        database=args.database,
        max_features=args.max_features,
        max_depth=args.max_depth,
        verbosity=args.verbose
    )

    # Load or train model
    if args.load_model:
        classifier.load_model(args.load_model)
    else:
        # Train model
        stats = classifier.fit(
            limit=args.limit,
            test_size=args.test_size
        )

        # Show feature importances
        if args.verbose >= 2:
            print("\nTop Feature Importances:")
            for feature, importance in classifier.get_feature_importances(top_n=20):
                print(f"  {feature}: {importance:.4f}")

        # Save model if requested
        if args.save_model:
            classifier.save_model(args.save_model)

    # Make prediction if requested
    if args.predict:
        results = classifier.predict(args.predict)
        print("\nPrediction Results:")
        for result in results:
            print(f"  Predicted ID: {result['predicted_id']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            if result['top_predictions']:
                print("  Top predictions:")
                for pred in result['top_predictions']:
                    print(f"    {pred['taxa_id']}: {pred['probability']:.4f}")


if __name__ == '__main__':
    main()
