# Line-by-Line Classification with YEDDA Output

## Summary

Extended the SKOL classifier to support line-by-line classification in addition to paragraph-based classification. The new functionality produces YEDDA-formatted output where consecutive lines with the same label are coalesced into blocks.

## Changes Made

### 1. New Methods in `classifier.py`

#### `load_raw_data_lines(file_paths: List[str]) -> DataFrame`
- Loads raw text files as individual lines (not paragraphs)
- Adds line numbers within each file
- Returns DataFrame with columns: `value`, `filename`, `line_number`

#### `predict_lines(file_paths: List[str], output_format: str = "yedda") -> DataFrame`
- Classifies individual lines instead of paragraphs
- Supports three output formats:
  - `"yedda"`: Returns predictions for further processing
  - `"annotated"`: Wraps each line in YEDDA format
  - `"simple"`: Returns basic predictions
- Returns DataFrame with predictions and labels

#### `coalesce_consecutive_labels(lines_data: List[Dict[str, Any]]) -> str`
- Static method that merges consecutive lines with the same label
- Creates YEDDA blocks with format: `[@ <lines>\n#<label>*]`
- Handles label transitions automatically

#### `save_yedda_output(predictions: DataFrame, output_path: str) -> None`
- Saves line-level predictions in YEDDA format
- Automatically coalesces consecutive same-label lines into blocks
- Outputs files partitioned by source filename

### 2. Test Files

#### `test_line_classifier.py`
Unit tests for the coalescence functionality:
- Tests basic coalescence of consecutive labels
- Tests empty data handling
- Tests single line handling
- All tests pass ✓

#### `example_line_classification.py`
Example script demonstrating:
- How to use the new API
- YEDDA format output example
- Comparison with paragraph-based classification

### 3. Documentation Updates

#### `README.md`
- Added new "Line-by-Line Classification with YEDDA Output" section
- Documented all new API methods
- Included usage examples
- Updated features list

## Usage Comparison

### Paragraph-Based (Original)
```python
classifier = SkolClassifier()
predictions = classifier.predict_raw_text(['file.txt'])
classifier.save_annotated_output(predictions, 'output_dir')
```

Output: Each paragraph as a separate YEDDA block

### Line-Based (New)
```python
classifier = SkolClassifier()
predictions = classifier.predict_lines(['file.txt'])
classifier.save_yedda_output(predictions, 'output_dir')
```

Output: Consecutive lines with same label coalesced into YEDDA blocks

## YEDDA Format Example

Input lines with labels:
```
[Nomenclature] Glomus mosseae Nicolson & Gerdemann, 1963.
[Nomenclature] ≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker
[Description] Key characters: Spores formed singly.
[Description] Spore wall: mono- to multiple-layered.
[Misc-exposition] This species is common in temperate regions.
```

Output YEDDA blocks:
```
[@ Glomus mosseae Nicolson & Gerdemann, 1963.
≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker
#Nomenclature*]
[@ Key characters: Spores formed singly.
Spore wall: mono- to multiple-layered.
#Description*]
[@ This species is common in temperate regions.
#Misc-exposition*]
```

## Benefits

1. **Granular Control**: Classify at line level instead of relying on paragraph detection
2. **Better Segmentation**: More accurate for documents with poor paragraph structure
3. **YEDDA Integration**: Output compatible with the `yedda_parser` module
4. **Automatic Coalescence**: Reduces output size by merging consecutive same-label lines
5. **Flexible Output**: Multiple output format options

## Integration with yedda_parser

The YEDDA output from `save_yedda_output()` can be directly parsed by the `yedda_parser` module:

```python
from yedda_parser import yedda_file_to_spark_df

# After classification
df = yedda_file_to_spark_df('output_dir/file.txt', spark)
df.show()
```

This creates a complete pipeline:
1. Classify text line-by-line
2. Coalesce into YEDDA blocks
3. Parse YEDDA into structured DataFrame
4. Perform downstream analysis

## Files Modified

- `skol_classifier/classifier.py`: Added 4 new methods (~175 lines)
- `skol_classifier/README.md`: Added documentation section
- `skol_classifier/test_line_classifier.py`: New test file
- `skol_classifier/example_line_classification.py`: New example script

## Testing

Run tests:
```bash
python skol_classifier/test_line_classifier.py
```

Run example:
```bash
python skol_classifier/example_line_classification.py
```

All tests pass successfully.
