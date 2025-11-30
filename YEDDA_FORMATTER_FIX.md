# YeddaFormatter Coalesce Support

## Issue

**Error**: `TypeError: YeddaFormatter() takes no arguments`

**Location**: [skol_classifier/classifier_v2.py:551](skol_classifier/classifier_v2.py:551)

**Cause**: The code tried to instantiate `YeddaFormatter` with constructor arguments (`coalesce_labels`), but the class only had static methods and didn't accept any parameters.

## Solution

Refactored `YeddaFormatter` to support instance-based configuration with coalesce support, while maintaining backward compatibility with static methods.

### Key Changes

1. **Instance Constructor**: Added `__init__()` method to accept configuration
2. **Instance Method**: Added `format()` method that applies coalescing when configured
3. **Coalesce Support**: Automatically coalesces consecutive labels when enabled
4. **Backward Compatibility**: Kept static methods so existing code still works

## Implementation Details

### YeddaFormatter Class ([output_formatters.py:15-72](skol_classifier/output_formatters.py:15-72))

**Before (Static Only)**:
```python
class YeddaFormatter:
    @staticmethod
    def format_predictions(predictions: DataFrame) -> DataFrame:
        return predictions.withColumn(
            "annotated_value",
            concat(lit("[@ "), col("value"), lit(" #"),
                   col("predicted_label"), lit("*]"))
        )
```

**After (Instance + Static)**:
```python
class YeddaFormatter:
    def __init__(self, coalesce_labels: bool = False, line_level: bool = False):
        """
        Initialize the formatter.

        Args:
            coalesce_labels: Whether to coalesce consecutive labels
            line_level: Whether data is line-level
        """
        self.coalesce_labels = coalesce_labels
        self.line_level = line_level

    def format(self, predictions: DataFrame) -> DataFrame:
        """
        Format predictions in YEDDA annotation format.

        Applies coalescing if configured.
        """
        # Add annotated_value column
        formatted = self.format_predictions(predictions)

        # Coalesce if requested
        if self.coalesce_labels and self.line_level:
            return self.coalesce_consecutive_labels(
                formatted,
                line_level=self.line_level
            )

        return formatted

    @staticmethod
    def format_predictions(predictions: DataFrame) -> DataFrame:
        """Static method for basic formatting."""
        return predictions.withColumn(
            "annotated_value",
            concat(lit("[@ "), col("value"), lit(" #"),
                   col("predicted_label"), lit("*]"))
        )
```

### Classifier V2 Usage ([classifier_v2.py:549-555](skol_classifier/classifier_v2.py:549-555))

**Before (Broken)**:
```python
def _format_as_annotated(self, predictions_df: DataFrame) -> DataFrame:
    formatter = YeddaFormatter(coalesce_labels=self.coalesce_labels)  # ❌ Error!
    return formatter.format(predictions_df)
```

**After (Fixed)**:
```python
def _format_as_annotated(self, predictions_df: DataFrame) -> DataFrame:
    formatter = YeddaFormatter(
        coalesce_labels=self.coalesce_labels,
        line_level=self.line_level  # ✅ Now passes line_level too
    )
    return formatter.format(predictions_df)
```

## Coalesce Behavior

### Without Coalesce (coalesce_labels=False)

Each line/paragraph gets its own annotation:
```
[@ line 1 text #Label1*]
[@ line 2 text #Label1*]
[@ line 3 text #Label2*]
```

### With Coalesce (coalesce_labels=True, line_level=True)

Consecutive lines with the same label are merged:
```
[@ line 1 text
line 2 text #Label1*]
[@ line 3 text #Label2*]
```

## Benefits

1. **Coalesce Support**: Users can now merge consecutive predictions with the same label
2. **Cleaner Output**: Reduces annotation noise for line-level predictions
3. **Backward Compatible**: Static methods still work for existing code
4. **Configurable**: Users control behavior via constructor parameters

## Usage Examples

### Instance Usage (New Way)

```python
from skol_classifier.output_formatters import YeddaFormatter

# With coalescing
formatter = YeddaFormatter(coalesce_labels=True, line_level=True)
formatted = formatter.format(predictions_df)

# Without coalescing
formatter = YeddaFormatter(coalesce_labels=False)
formatted = formatter.format(predictions_df)
```

### Static Usage (Old Way - Still Works)

```python
from skol_classifier.output_formatters import YeddaFormatter

# Basic formatting only
formatted = YeddaFormatter.format_predictions(predictions_df)

# Manual coalescing
coalesced = YeddaFormatter.coalesce_consecutive_labels(
    formatted,
    line_level=True
)
```

### SkolClassifierV2 Usage

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.getOrCreate()

# With coalescing enabled
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    line_level=True,
    coalesce_labels=True,  # Enable coalescing
    output_format='annotated',
    model_type='logistic'
)

results = classifier.fit()
predictions = classifier.predict()  # Automatically coalesces
```

## Files Modified

1. [skol_classifier/output_formatters.py](skol_classifier/output_formatters.py)
   - Lines 15-72: Added `__init__()` and `format()` methods to YeddaFormatter

2. [skol_classifier/classifier_v2.py](skol_classifier/classifier_v2.py)
   - Lines 549-555: Updated `_format_as_annotated()` to pass both parameters

## Testing

To test the coalesce functionality:

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.getOrCreate()

# Test without coalescing
classifier1 = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    line_level=True,
    coalesce_labels=False,  # Each line separate
    model_type='logistic'
)
results1 = classifier1.fit()
predictions1 = classifier1.predict()

# Test with coalescing
classifier2 = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    line_level=True,
    coalesce_labels=True,  # Merge consecutive
    model_type='logistic'
)
results2 = classifier2.fit()
predictions2 = classifier2.predict()
```

## Related Issues Fixed

This fix resolves:
- ✅ YeddaFormatter instantiation error
- ✅ Adds coalesce support as recommended
- ✅ Maintains backward compatibility
- ✅ Supports both line-level and paragraph-level data

## Notes

- Coalescing only works when `line_level=True` (paragraph-level doesn't need it)
- The `coalesce_consecutive_labels()` static method is still available for manual use
- The instance `format()` method automatically handles coalescing when configured
- Coalescing preserves document structure by grouping on filename or doc_id
