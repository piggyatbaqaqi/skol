#!/usr/bin/env python3
"""
Test script to verify GPU compatibility detection and automatic CPU fallback.

Run this script to test if the RTX 5090 compute capability detection works.
"""

import sys
import os

# Make sure we can import skol_classifier
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("Testing GPU Compatibility Detection")
print("=" * 70)
print()

try:
    from skol_classifier.rnn_model import KERAS_AVAILABLE, build_bilstm_model

    if KERAS_AVAILABLE:
        print("✓ TensorFlow imported successfully")
        print("  Auto-detection logic has run during import")
        print()

        print("Attempting to build BiLSTM model...")
        print("-" * 70)

        try:
            model = build_bilstm_model(
                input_shape=(50, 1000),
                num_classes=3,
                hidden_size=128,
                num_layers=2,
                dropout=0.3
            )
            print()
            print("✓ Model built successfully!")
            print(f"  Model has {len(model.layers)} layers")
            print()
            print("NOTE: If you see 'Forced CPU-only mode' above, your GPU")
            print("      (compute capability 12.0) was detected as incompatible")
            print("      and CPU mode was automatically activated.")

        except RuntimeError as e:
            print()
            print("✗ Model build failed with RuntimeError")
            print()
            print("Error message:")
            print(str(e))
            print()
            print("SOLUTION: Restart your Python session/kernel and add this")
            print("          BEFORE any imports:")
            print()
            print("    import os")
            print("    os.environ['CUDA_VISIBLE_DEVICES'] = ''")

        except Exception as e:
            print()
            print(f"✗ Unexpected error: {type(e).__name__}")
            print(str(e))
    else:
        print("✗ TensorFlow/Keras not available")
        print("  Install with: pip install tensorflow[and-cuda]")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print()
    print("Make sure you have all dependencies installed:")
    print("  pip install pyspark numpy pandas tensorflow[and-cuda]")

print()
print("=" * 70)
