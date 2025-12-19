"""
Example: Train and evaluate a hybrid model (logistic + RNN).

This script demonstrates how to:
1. Train a hybrid model that combines logistic and RNN
2. Tune the nomenclature threshold
3. Compare performance against individual models
"""

import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def main():
    # Initialize Spark and Redis
    spark = SparkSession.builder.appName("Hybrid Model Example").getOrCreate()
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Paths to annotated data
    annotated_files = ['data/annotated/*.ann']

    print("="*70)
    print("Training Hybrid Model (Logistic + RNN)")
    print("="*70)

    # Configuration
    config = {
        'spark': spark,
        'input_source': 'files',
        'file_paths': annotated_files,
        'auto_load_model': False,
        'model_storage': 'redis',
        'redis_client': redis_client,
        'verbosity': 1
    }

    # Train hybrid model
    print("\n[1/1] Training Hybrid Model")
    print("-" * 70)

    hybrid_classifier = SkolClassifierV2(
        model_type='hybrid',
        redis_key='hybrid_model',

        # Hybrid threshold
        nomenclature_threshold=0.6,

        # Logistic parameters
        logistic_params={
            'maxIter': 20,
            'regParam': 0.01
        },

        # RNN parameters
        rnn_params={
            'window_size': 35,
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.4,
            'epochs': 15,
            'batch_size': 4000,
            'prediction_stride': 1,
            'class_weights': {
                'Nomenclature': 80.0,
                'Description': 8.0,
                'Misc-exposition': 0.2
            }
        },

        **config
    )

    hybrid_results = hybrid_classifier.fit()

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nHybrid Model Performance:")
    print(f"  Nomenclature F1: {hybrid_results['test_stats']['Nomenclature_f1']:.4f}")
    print(f"  Description F1:  {hybrid_results['test_stats']['Description_f1']:.4f}")
    print(f"  Misc F1:         {hybrid_results['test_stats']['Misc-exposition_f1']:.4f}")
    print(f"  Overall F1:      {hybrid_results['test_stats']['f1_score']:.4f}")
    print(f"  Overall Accuracy: {hybrid_results['test_stats']['accuracy']:.4f}")

    # Save model
    hybrid_classifier.save_model()
    print(f"\n✓ Model saved to Redis: hybrid_model")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


def compare_models():
    """
    Optional: Compare hybrid against individual models.
    """
    spark = SparkSession.builder.appName("Model Comparison").getOrCreate()
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    annotated_files = ['data/annotated/*.ann']

    config = {
        'spark': spark,
        'input_source': 'files',
        'file_paths': annotated_files,
        'auto_load_model': False,
        'model_storage': 'redis',
        'redis_client': redis_client,
        'verbosity': 1
    }

    results = {}

    # 1. Logistic only
    print("\n[1/3] Training Logistic Regression...")
    logistic = SkolClassifierV2(
        model_type='logistic',
        redis_key='logistic_only',
        maxIter=20,
        regParam=0.01,
        **config
    )
    results['logistic'] = logistic.fit()

    # 2. RNN only
    print("\n[2/3] Training RNN...")
    rnn = SkolClassifierV2(
        model_type='rnn',
        redis_key='rnn_only',
        window_size=35,
        hidden_size=256,
        num_layers=3,
        dropout=0.4,
        epochs=15,
        batch_size=4000,
        class_weights={
            'Nomenclature': 80.0,
            'Description': 8.0,
            'Misc-exposition': 0.2
        },
        **config
    )
    results['rnn'] = rnn.fit()

    # 3. Hybrid
    print("\n[3/3] Training Hybrid...")
    hybrid = SkolClassifierV2(
        model_type='hybrid',
        redis_key='hybrid_comparison',
        nomenclature_threshold=0.6,
        logistic_params={'maxIter': 20, 'regParam': 0.01},
        rnn_params={
            'window_size': 35,
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.4,
            'epochs': 15,
            'batch_size': 4000,
            'class_weights': {
                'Nomenclature': 80.0,
                'Description': 8.0,
                'Misc-exposition': 0.2
            }
        },
        **config
    )
    results['hybrid'] = hybrid.fit()

    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    headers = ["Model", "Nomenclature", "Description", "Misc", "Overall"]
    print(f"\n{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10}")
    print("-" * 70)

    for model_name, result in results.items():
        stats = result['test_stats']
        print(f"{model_name:<15} "
              f"{stats['Nomenclature_f1']:<15.4f} "
              f"{stats['Description_f1']:<15.4f} "
              f"{stats['Misc-exposition_f1']:<10.4f} "
              f"{stats['f1_score']:<10.4f}")

    print("\n" + "="*70)


def tune_threshold():
    """
    Optional: Find optimal threshold for hybrid model.
    """
    spark = SparkSession.builder.appName("Threshold Tuning").getOrCreate()
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    annotated_files = ['data/annotated/*.ann']

    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    print("\n" + "="*70)
    print("THRESHOLD TUNING")
    print("="*70)

    results = []

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")

        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=annotated_files,
            model_type='hybrid',
            nomenclature_threshold=threshold,
            logistic_params={'maxIter': 20, 'regParam': 0.01},
            rnn_params={
                'window_size': 35,
                'hidden_size': 256,
                'num_layers': 3,
                'epochs': 10,  # Fewer epochs for speed
                'batch_size': 4000,
                'class_weights': {
                    'Nomenclature': 80.0,
                    'Description': 8.0,
                    'Misc-exposition': 0.2
                }
            },
            verbosity=0  # Quiet
        )

        result = classifier.fit()
        results.append({
            'threshold': threshold,
            'nomenclature_f1': result['test_stats']['Nomenclature_f1'],
            'description_f1': result['test_stats']['Description_f1'],
            'overall_f1': result['test_stats']['f1_score']
        })

    # Print results
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)

    headers = ["Threshold", "Nomenclature F1", "Description F1", "Overall F1"]
    print(f"\n{headers[0]:<12} {headers[1]:<18} {headers[2]:<18} {headers[3]:<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['threshold']:<12.2f} "
              f"{r['nomenclature_f1']:<18.4f} "
              f"{r['description_f1']:<18.4f} "
              f"{r['overall_f1']:<12.4f}")

    # Find best threshold
    best = max(results, key=lambda x: x['overall_f1'])
    print(f"\n✓ Best threshold: {best['threshold']} (Overall F1: {best['overall_f1']:.4f})")
    print("="*70)


if __name__ == '__main__':
    # Basic usage: train a hybrid model
    main()

    # Optional: Uncomment to compare models
    # compare_models()

    # Optional: Uncomment to tune threshold
    # tune_threshold()
