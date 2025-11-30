# Line-Level Training Fix

## Problem

The `line_level` parameter was added to the model comparison configurations, but it wasn't actually being used during training. The `fit()` method was calling `load_annotated_data()` without passing the `line_level` parameter, so all models were training on the same paragraph-level data regardless of the setting.

## Root Cause

1. `load_annotated_data()` didn't have a `line_level` parameter
2. `fit()` didn't accept or pass through `line_level`
3. No extraction logic existed for parsing YEDDA annotations at line level

## Solution

### 1. Updated `load_annotated_data()` Method

Added `line_level` parameter with two different extraction paths:

**Paragraph-level (line_level=False - original behavior):**
- Uses `ParagraphExtractor.extract_annotated_paragraphs()`
- Groups lines between `[@` and `#Label*]` markers into single paragraphs
- Each training sample is a complete paragraph

**Line-level (line_level=True - new behavior):**
- Uses custom `extract_yedda_lines()` function
- Parses YEDDA blocks and extracts each line individually
- Each training sample is a single line
- Preserves line numbers within blocks

```python
def load_annotated_data(
    self,
    file_paths: List[str],
    collapse_labels: bool = True,
    line_level: bool = False  # NEW PARAMETER
) -> DataFrame:
```

### 2. Updated `fit()` Method

Added `line_level` parameter and passes it to `load_annotated_data()`:

```python
def fit(
    self,
    annotated_file_paths: List[str],
    model_type: str = "logistic",
    use_suffixes: bool = True,
    test_size: float = 0.2,
    line_level: bool = False,  # NEW PARAMETER
    **model_params
) -> Dict[str, Any]:
    # Load annotated data with line_level setting
    annotated_df = self.load_annotated_data(
        annotated_file_paths,
        line_level=line_level  # PASSED THROUGH
    )
```

### 3. Updated Model Comparison

Modified `examples/model_comparison.py` to show training mode:

```python
print(f"  Mode:      {'Line-level' if stats.get('line_level', False) else 'Paragraph'}")
print(f"  Train:     {stats['train_size']} samples")
print(f"  Test:      {stats['test_size']} samples")
```

## Impact

### Training Data Differences

**Before (all paragraph-level):**
- Input: `[@ Line 1\nLine 2\n#Label*]`
- Training samples: 1 (entire paragraph)
- Sample text: "Line 1 Line 2"

**After (with line_level=True):**
- Input: `[@ Line 1\nLine 2\n#Label*]`
- Training samples: 2 (individual lines)
- Sample 1: "Line 1"
- Sample 2: "Line 2"

### Expected Results

**Line-level training should:**
- Produce more training samples (one per line vs one per paragraph)
- Have different accuracy/precision/recall metrics
- Learn patterns at finer granularity
- Be useful for documents with inconsistent paragraph boundaries

**Model comparison will now show:**
- 8 distinct configurations (4 paragraph + 4 line-level)
- Different train/test sizes for line-level vs paragraph models
- Actual performance differences between the approaches

## Example Output

```
Testing: Logistic Regression (words only)
  Mode:      Paragraph
  Train:     1200 samples
  Test:      300 samples
  Accuracy:  0.8500
  ...

Testing: Logistic Regression (line-level, words only)
  Mode:      Line-level
  Train:     4800 samples
  Test:      1200 samples
  Accuracy:  0.8200
  ...
```

## Technical Details

### YEDDA Line Extraction Pattern

```python
pattern = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'
```

This regex:
- Matches YEDDA blocks: `[@ content #Label*]`
- Captures content and label separately
- Uses DOTALL to match across newlines
- Splits content by `\n` to get individual lines

### Schema Differences

**Paragraph-level DataFrame:**
- `filename`: str
- `label`: str
- `value`: str (paragraph text)

**Line-level DataFrame:**
- `filename`: str
- `label`: str
- `value`: str (single line text)
- `line_number`: int (position within YEDDA block)

## Testing

The fix can be verified by:

1. Running model comparison and checking train/test sizes differ
2. Verifying line-level models have more samples
3. Checking that performance metrics are different
4. Inspecting the loaded DataFrame to confirm line vs paragraph extraction

```python
# Test line-level loading
classifier = SkolClassifier()
df = classifier.load_annotated_data(['data/sample.txt.ann'], line_level=True)
df.show()  # Should show individual lines, not paragraphs
```

## Files Modified

- [skol_classifier/classifier.py](classifier.py:92) - Updated `load_annotated_data()` and `fit()`
- [examples/model_comparison.py](../examples/model_comparison.py:109) - Added mode/sample count output

## Backward Compatibility

✅ **Fully backward compatible**
- Default `line_level=False` maintains original behavior
- Existing code continues to work unchanged
- Only affects users who explicitly set `line_level=True`
