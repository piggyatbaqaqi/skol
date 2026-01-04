# Quick RNN Testing

## TL;DR

```bash
# From the skol directory
python test_rnn_synthetic.py
```

**Time:** 30-60 seconds
**Purpose:** Rapidly test RNN model changes without restarting Jupyter

## Files Created

1. **[test_rnn_synthetic.py](test_rnn_synthetic.py)** - The test script
2. **[RNN_TESTING_GUIDE.md](RNN_TESTING_GUIDE.md)** - Complete documentation

## What It Tests

✓ RNN model initialization
✓ Synthetic data generation (3 label classes)
✓ Training with minimal parameters
✓ Prediction on test set
✓ Statistics calculation
✓ Complete end-to-end pipeline

## Key Features

- **Fast**: 30-60 seconds vs 5-10 minutes for full data
- **Isolated**: Tests only RNN, not entire pipeline
- **No Restart**: Modify code and re-run immediately
- **Synthetic Data**: Consistent, reproducible results
- **CPU-Only**: Automatic GPU bypass for RTX 5090 compatibility
- **Instrumented**: Detailed logging shows exactly where failures occur

## Command Line Options

```bash
# Faster test (minimal data/training)
python test_rnn_synthetic.py --epochs 1 --num-docs 5

# More detailed output
python test_rnn_synthetic.py --verbosity 3

# Larger test
python test_rnn_synthetic.py --num-docs 20 --epochs 5
```

## Example Output

```
======================================================================
RNN Model Synthetic Data Test
======================================================================
...
[RNN Fit] Training completed successfully
[Classifier Fit] Making predictions on test set
[RNN Predict] Starting prediction
[RNN Predict] Applying prediction UDF to sequences
[RNN Predict] Joining predictions with labels
[RNN Predict] Prediction completed successfully
...
======================================================================
TRAINING COMPLETE
======================================================================

Training time: 45.23 seconds

Results:
  accuracy: 0.6500
  precision: 0.6234
  recall: 0.6500
  f1_score: 0.6112

✓✓✓ SUCCESS ✓✓✓
```

## Iteration Workflow

1. Run test → Note failure point
2. Edit `rnn_model.py`
3. Re-run test (no restart!)
4. Repeat until success
5. Test with real data in Jupyter

## See Also

- **[RNN_TESTING_GUIDE.md](RNN_TESTING_GUIDE.md)** - Full documentation
- **[INSTRUMENTATION_GUIDE.md](INSTRUMENTATION_GUIDE.md)** - Logging details
- **[GPU_COMPATIBILITY.md](GPU_COMPATIBILITY.md)** - GPU/CPU mode guide
