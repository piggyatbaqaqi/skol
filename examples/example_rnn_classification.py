"""
Example: Using RNN model for line-level classification with context.

This example demonstrates how to use the RNN (Bidirectional LSTM) model
for taxonomic text classification. The RNN model uses surrounding lines
as context to improve classification accuracy.

Prerequisites:
    pip install tensorflow elephas

Usage:
    python example_rnn_classification.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def main():
    """Example of RNN-based line classification."""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("RNN Line Classification") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    print("=" * 80)
    print("RNN Model for Line-Level Classification with Context")
    print("=" * 80)

    # Example 1: Train RNN model from CouchDB
    print("\n1. TRAINING RNN MODEL FROM COUCHDB")
    print("-" * 80)

    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='annotated_taxa',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt.ann',
        model_storage='disk',
        model_path='models/rnn_model.pkl',
        line_level=True,  # IMPORTANT: RNN requires line-level mode
        use_suffixes=True,
        model_type='rnn',  # Use RNN model
        # RNN-specific parameters
        input_size=1000,      # Feature vector size (matches TF-IDF output)
        hidden_size=128,      # LSTM hidden state size
        num_layers=2,         # Number of LSTM layers
        num_classes=3,        # Number of classes (Nomenclature, Description, Misc-exposition)
        dropout=0.3,          # Dropout rate
        window_size=50,       # Maximum sequence length
        batch_size=32,        # Batch size
        epochs=10,            # Training epochs
        num_workers=4         # Spark workers for distributed training
    )

    print("Training RNN model...")
    print(f"  Model type: {classifier.model_type}")
    print(f"  Line-level mode: {classifier.line_level}")
    print(f"  Using suffixes: {classifier.use_suffixes}")

    # Train the model
    metrics = classifier.fit()

    # Save the model to disk
    classifier.save_model()

    print("\nTraining completed!")
    print(f"  Model saved to: {classifier.model_path}")
    print(f"  Metrics: {metrics}")

    # Example 2: Predict using trained RNN model
    print("\n2. MAKING PREDICTIONS WITH RNN MODEL")
    print("-" * 80)

    predictor = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='raw_taxa',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt',
        output_dest='couchdb',
        output_couchdb_suffix='.ann',
        model_storage='disk',
        model_path='models/rnn_model.pkl',
        line_level=True,
        coalesce_labels=True,
        auto_load_model=True
    )

    print("Loading raw data...")
    raw_df = predictor.load_raw()
    print(f"  Loaded {raw_df.count()} lines")

    print("\nMaking predictions with RNN context...")
    predictions_df = predictor.predict(raw_df)

    print("\nSample predictions:")
    predictions_df.select('doc_id', 'value', 'predicted_label').show(10, truncate=50)

    print("\nSaving predictions to CouchDB...")
    predictor.save_annotated(predictions_df)

    # Example 3: Compare RNN vs Logistic Regression
    print("\n3. COMPARING RNN VS LOGISTIC REGRESSION")
    print("-" * 80)

    # Train logistic regression model
    print("\nTraining Logistic Regression baseline...")
    lr_classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='annotated_taxa',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt.ann',
        model_storage='disk',
        model_path='models/lr_model.pkl',
        line_level=True,
        use_suffixes=True,
        model_type='logistic',
        maxIter=10,
        regParam=0.01
    )

    lr_metrics = lr_classifier.fit()

    # Save the logistic regression model
    lr_classifier.save_model()

    print("\nComparison:")
    print(f"  RNN Model: {metrics}")
    print(f"  Logistic Regression: {lr_metrics}")

    # Example 4: Advanced RNN configuration
    print("\n4. ADVANCED RNN CONFIGURATION")
    print("-" * 80)
    print("""
    For better performance, consider:

    1. Larger hidden size for complex patterns:
        hidden_size=256

    2. More layers for deeper understanding:
        num_layers=3

    3. Longer context windows:
        window_size=100

    4. More training epochs:
        epochs=20

    5. Adjust dropout to prevent overfitting:
        dropout=0.5

    Example configuration:
    """)

    print("""
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='annotated_taxa',
        model_type='rnn',
        line_level=True,

        # Advanced RNN configuration
        hidden_size=256,      # Larger hidden state
        num_layers=3,         # Deeper network
        window_size=100,      # Longer context
        epochs=20,            # More training
        dropout=0.5,          # Higher dropout
        batch_size=64,        # Larger batches
        num_workers=8         # More parallelism
    )
    """)

    print("\n" + "=" * 80)
    print("RNN Model Benefits:")
    print("  • Uses surrounding lines as context")
    print("  • Better handles sequential patterns")
    print("  • Improved accuracy on ambiguous cases")
    print("  • Learns document structure")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
