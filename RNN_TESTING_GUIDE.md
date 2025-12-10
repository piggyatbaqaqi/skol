# RNN Testing Guide

Quick testing workflow for iterating on RNN model fixes.

## Quick Start

```bash
cd /data/piggy/src/github.com/piggyatbaqaqi/skol
python test_rnn_synthetic.py
```

Expected time: **30-60 seconds**

## What It Does

The `test_rnn_synthetic.py` script:

1. **Creates synthetic data** (10 documents, 20 lines each)
2. **Generates 3 label classes** with distinct word patterns
3. **Trains a minimal RNN** (1 layer, 32 hidden units, 2 epochs)
4. **Evaluates and reports** accuracy/precision/recall
5. **Times the entire process**

## Usage

### Basic Usage

```bash
python test_rnn_synthetic.py
```

### Adjust Verbosity

```bash
# Minimal output
python test_rnn_synthetic.py --verbosity 1

# Detailed output (default)
python test_rnn_synthetic.py --verbosity 2

# Maximum debug output
python test_rnn_synthetic.py --verbosity 3
```

### Adjust Data Size

```bash
# Tiny dataset (faster)
python test_rnn_synthetic.py --num-docs 5 --lines-per-doc 10

# Larger dataset (more realistic)
python test_rnn_synthetic.py --num-docs 20 --lines-per-doc 30
```

### Adjust Training

```bash
# Faster training (single epoch)
python test_rnn_synthetic.py --epochs 1

# Longer training
python test_rnn_synthetic.py --epochs 5

# Smaller window size
python test_rnn_synthetic.py --window-size 5
```

## Expected Output

### Success

```
======================================================================
RNN Model Synthetic Data Test
======================================================================

Environment:
  CUDA_VISIBLE_DEVICES:
  Python: 3.13.x

Configuration:
  Verbosity: 2
  Documents: 10
  Lines per doc: 20
  Epochs: 2
  Window size: 10

----------------------------------------------------------------------
Initializing Spark...
----------------------------------------------------------------------
✓ Spark version: 3.x.x

----------------------------------------------------------------------
Generating synthetic data...
----------------------------------------------------------------------
Generating 10 documents with 20 lines each...
✓ Generated 200 total lines
  Label distribution:
    Misc-exposition: 70 (35.0%)
    Description: 66 (33.0%)
    Nomenclature: 64 (32.0%)

✓ Created DataFrame with 200 rows

----------------------------------------------------------------------
Initializing RNN Classifier...
----------------------------------------------------------------------
Model parameters:
  input_size: 100
  hidden_size: 32
  num_layers: 1
  num_classes: 3
  dropout: 0.2
  window_size: 10
  batch_size: 8
  epochs: 2
  num_workers: 2
  verbosity: 2

✓ Classifier initialized

======================================================================
TRAINING
======================================================================

[RNN Fit] Starting RNN model training
[... training output ...]
[RNN Fit] Training completed successfully
[Classifier Fit] Model training completed, starting evaluation
[Classifier Fit] Making predictions on test set
[RNN Predict] Starting prediction
[RNN Predict] Applying prediction UDF to sequences
[RNN Predict] Joining predictions with labels
[RNN Predict] Prediction completed successfully
[Classifier Fit] Calculating statistics

======================================================================
TRAINING COMPLETE
======================================================================

Training time: 45.23 seconds

Results:
  accuracy: 0.6500
  precision: 0.6234
  recall: 0.6500
  f1_score: 0.6112
  train_size: 160
  test_size: 40

✓✓✓ SUCCESS ✓✓✓

The RNN model trained and evaluated successfully!
You can now iterate on rnn_model.py with confidence.

Stopping Spark session...
✓ Done
```

### Common Failure Points

If the script fails, you'll see exactly where:

**Failure during UDF execution:**
```
[RNN Predict] Applying prediction UDF to sequences
ERROR ArrowPythonRunner: Python worker exited unexpectedly (crashed)
```
→ Problem in the predict_sequence UDF

**Failure during join:**
```
[RNN Predict] Joining predictions with labels
EOFError / PythonException
```
→ Problem with prediction array shapes

**Failure during stats calculation:**
```
[Classifier Fit] Calculating statistics
ERROR / Exception
```
→ Problem in calculate_stats method

## Iterating on Fixes

### Workflow

1. **Run the test:**
   ```bash
   python test_rnn_synthetic.py --verbosity 2
   ```

2. **Note the failure point** from the last logged message

3. **Edit `skol_classifier/rnn_model.py`** to fix the issue

4. **Re-run immediately** (no restart needed for this script)
   ```bash
   python test_rnn_synthetic.py --verbosity 2
   ```

5. **Repeat** until success

### Fast Iteration Tips

- Use `--verbosity 2` to see detailed progress
- Use `--epochs 1` for faster testing
- Use `--num-docs 5` for minimal dataset
- Watch for the **last message** before failure

### Debugging a Specific Stage

**To debug UDF execution:**
```bash
# Add more logging in the predict_sequence UDF
# Look for executor logs in Spark logs
```

**To debug join operation:**
```bash
# Set verbosity to 3 to see data schemas
python test_rnn_synthetic.py --verbosity 3
```

**To debug statistics calculation:**
```bash
# Check the RNN Stats output
# Verify prediction column types
```

## Advantages Over Full Testing

Compared to running your full Jupyter notebook:

| Aspect | Full Notebook | Synthetic Test |
|--------|---------------|----------------|
| **Time** | 5-10 minutes | 30-60 seconds |
| **Data** | Real data | Synthetic |
| **Restart** | Required | Not required |
| **Isolation** | Full pipeline | RNN-focused |
| **Iteration** | Slow | Fast |

## After Success

Once `test_rnn_synthetic.py` passes:

1. **Test with real data** in your Jupyter notebook
2. **Restart your kernel** (required for CPU mode)
3. **Set `CUDA_VISIBLE_DEVICES=''`** before imports
4. **Run your full training**

## Synthetic Data Details

The script generates 3 label classes with distinct word patterns:

- **Misc-exposition**: "this", "is", "about", "generally", "overall", "introduction"
- **Description**: "describes", "shows", "defines", "represents", "indicates"
- **Nomenclature**: "name", "called", "term", "symbol", "notation", "denoted"

Each document has roughly equal distribution of labels, with some randomness.

The data is simple enough to train quickly but realistic enough to catch bugs in:
- Sequence processing
- Prediction formatting
- Label alignment
- Statistics calculation

## Environment

The script automatically:
- Sets `CUDA_VISIBLE_DEVICES=''` (CPU-only mode)
- Reduces TensorFlow logging
- Uses minimal Spark resources
- Cleans up on exit

No manual environment setup required!
