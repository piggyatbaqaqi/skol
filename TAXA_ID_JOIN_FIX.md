# Fix: Join on _id Instead of taxon in translate_descriptions_batch()

## Overview

Updated `TaxaJSONTranslator.translate_descriptions_batch()` to join results back to the original DataFrame using the `_id` field (CouchDB document ID) instead of the `taxon` field.

## Problem

Previously, the batch translation method joined on `taxon`:

```python
# Old approach
descriptions = taxa_df.select("taxon", description_col).collect()
# ... process ...
results.append({'taxon': taxon, output_col: json_obj})
enriched_df = taxa_df.join(results_df, on="taxon", how="left")
```

**Issue**: The `taxon` field is not guaranteed to be unique. Multiple taxa could have the same name if they appear in different documents or contexts.

## Solution

Join on `_id` (the CouchDB document ID) which is guaranteed to be unique:

```python
# New approach
descriptions = taxa_df.select("_id", "taxon", description_col).collect()
# ... process ...
results.append({'_id': doc_id, output_col: json_obj})
enriched_df = taxa_df.join(results_df, on="_id", how="left")
```

## Changes Made

### 1. Updated Schema in TaxonExtractor

**File**: [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py)

**Change**: Added `_id` field to schema

```python
self._extract_schema = StructType([
    StructField("_id", StringType(), True),  # CouchDB document ID (nullable for extract_taxa)
    StructField("taxon", StringType(), False),
    StructField("description", StringType(), False),
    # ... rest of fields
])
```

**Rationale**:
- `_id` is nullable because newly extracted taxa don't have CouchDB IDs yet
- Once saved and loaded from CouchDB, `_id` will be populated

### 2. Updated convert_taxa_to_rows()

**File**: [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py)

**Change**: Set `_id` to `None` for newly extracted taxa

```python
for taxon in partition:
    taxon_dict = taxon.as_row()
    # Add _id as None for newly extracted taxa (they don't have CouchDB IDs yet)
    taxon_dict['_id'] = None
    yield Row(**taxon_dict)
```

### 3. Updated load_taxa()

**File**: [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py)

**Change**: Include `_id` when loading from CouchDB

```python
taxon_data = {
    '_id': doc.get('_id', doc_id),  # Include CouchDB _id
    'taxon': doc.get('taxon', ''),
    'description': doc.get('description', ''),
    # ... rest of fields
}
```

### 4. Updated translate_descriptions_batch()

**File**: [taxa_json_translator.py](taxa_json_translator.py)

**Change**: Collect `_id` and join on it

```python
# Collect descriptions with _id for joining
descriptions = taxa_df.select("_id", "taxon", description_col).collect()

for row in batch:
    doc_id = row['_id']
    taxon = row['taxon']
    description = row[description_col]

    json_obj = self.generate_json(description)

    results.append({
        '_id': doc_id,
        output_col: json_obj
    })

# Join back to original DataFrame on _id
enriched_df = taxa_df.join(results_df, on="_id", how="left")
```

## Benefits

### 1. Correctness

**Before**: If two taxa have the same name, the join would be ambiguous
```
taxon: "Agaricus bisporus"  →  Which one gets which JSON?
taxon: "Agaricus bisporus"  →  Potentially incorrect assignment
```

**After**: Each taxon has a unique `_id`, ensuring 1:1 mapping
```
_id: "taxon_abc123"  →  JSON for this specific taxon
_id: "taxon_def456"  →  JSON for that specific taxon
```

### 2. Idempotency

The `_id` is generated deterministically from:
- Source document ID
- Source URL
- Line number

This ensures the same taxon from the same location always gets the same `_id`.

### 3. CouchDB Integration

The `_id` field matches the CouchDB document ID, making it easy to:
- Update translated taxa back to CouchDB
- Track which document a taxon came from
- Maintain referential integrity

## Schema Changes

### Before

```
root
 |-- taxon: string (nullable = false)
 |-- description: string (nullable = false)
 |-- source: map (nullable = false)
 |-- line_number: integer (nullable = true)
 |-- paragraph_number: integer (nullable = true)
 |-- page_number: integer (nullable = true)
 |-- empirical_page_number: string (nullable = true)
```

### After

```
root
 |-- _id: string (nullable = true)           ← NEW
 |-- taxon: string (nullable = false)
 |-- description: string (nullable = false)
 |-- source: map (nullable = false)
 |-- line_number: integer (nullable = true)
 |-- paragraph_number: integer (nullable = true)
 |-- page_number: integer (nullable = true)
 |-- empirical_page_number: string (nullable = true)
```

## Usage Impact

### For Users

**No breaking changes** for most common usage patterns:

```python
# Still works the same
taxa_df = extractor.load_taxa()
enriched_df = translator.translate_descriptions_batch(taxa_df)
```

**New capability**: Can now filter by `_id`:

```python
# Get specific taxon by ID
specific_taxon = enriched_df.filter(col("_id") == "taxon_abc123")
```

### For Developers

**When using extract_taxa()**: `_id` will be `None` until taxa are saved to CouchDB

```python
# Extract taxa
extracted_df = extractor.extract_taxa(annotated_df)
extracted_df.select("_id", "taxon").show()
# +----+------------------+
# | _id|             taxon|
# +----+------------------+
# |null|Agaricus bisporus|  ← _id is None
# |null|Boletus edulis   |  ← _id is None
# +----+------------------+
```

**After saving and loading**: `_id` will be populated

```python
# Save to CouchDB
extractor.save_taxa(extracted_df)

# Load back
loaded_df = extractor.load_taxa()
loaded_df.select("_id", "taxon").show()
# +----------------+------------------+
# |            _id|             taxon|
# +----------------+------------------+
# |taxon_abc123...|Agaricus bisporus|  ← _id populated
# |taxon_def456...|Boletus edulis   |  ← _id populated
# +----------------+------------------+
```

## Edge Cases

### 1. Newly Extracted Taxa (No _id)

**Scenario**: Using `translate_descriptions_batch()` on freshly extracted taxa

**Behavior**: Will fail because `_id` is `None`

**Solution**: Save taxa to CouchDB first, then load and translate:

```python
# Extract
extracted_df = extractor.extract_taxa(annotated_df)

# Save to CouchDB (assigns _id)
extractor.save_taxa(extracted_df)

# Load with _id
taxa_df = extractor.load_taxa()

# Now translate (works because _id is populated)
enriched_df = translator.translate_descriptions_batch(taxa_df)
```

### 2. Using translate_descriptions() (UDF mode)

**Note**: The UDF-based method doesn't use joins, so it's unaffected:

```python
# Works regardless of _id
enriched_df = translator.translate_descriptions(taxa_df)
```

## Testing

### Test Case 1: Duplicate Taxa Names

```python
# Create test data with duplicate taxon names
test_data = [
    {"_id": "taxon_1", "taxon": "Species A", "description": "Desc 1"},
    {"_id": "taxon_2", "taxon": "Species A", "description": "Desc 2"}  # Same name!
]

taxa_df = spark.createDataFrame(test_data)

# Translate
enriched_df = translator.translate_descriptions_batch(taxa_df)

# Verify correct assignment
result1 = enriched_df.filter(col("_id") == "taxon_1").first()
result2 = enriched_df.filter(col("_id") == "taxon_2").first()

assert result1['features_json'] != result2['features_json']  # Different translations
```

### Test Case 2: Round-Trip

```python
# Extract and save
extracted_df = extractor.extract_taxa(annotated_df)
extractor.save_taxa(extracted_df)

# Load with _id
loaded_df = extractor.load_taxa()
assert loaded_df.filter("_id IS NOT NULL").count() == loaded_df.count()

# Translate
enriched_df = translator.translate_descriptions_batch(loaded_df)
assert enriched_df.count() == loaded_df.count()  # No rows lost in join
```

## Migration Guide

### If You're Using extract_taxa() Directly

**Before**: Could translate immediately
```python
extracted_df = extractor.extract_taxa(annotated_df)
enriched_df = translator.translate_descriptions_batch(extracted_df)  # Now fails
```

**After**: Must save and load first
```python
extracted_df = extractor.extract_taxa(annotated_df)
extractor.save_taxa(extracted_df)
loaded_df = extractor.load_taxa()
enriched_df = translator.translate_descriptions_batch(loaded_df)  # Works
```

**Alternative**: Use UDF mode (no join)
```python
extracted_df = extractor.extract_taxa(annotated_df)
enriched_df = translator.translate_descriptions(extracted_df)  # Still works
```

### If You're Using load_taxa()

**No changes needed** - this is the recommended workflow:

```python
# Load taxa (has _id)
taxa_df = extractor.load_taxa()

# Translate (uses _id for join)
enriched_df = translator.translate_descriptions_batch(taxa_df)
```

## Related Changes

This fix enables future enhancements:

1. **Save translations back to CouchDB**: Use `_id` to update existing documents
2. **Incremental translation**: Track which taxa have been translated by `_id`
3. **Cross-reference**: Link translations back to source documents

## Summary

✅ **More Correct**: Joins on unique `_id` instead of potentially duplicate `taxon`
✅ **CouchDB Aligned**: `_id` matches CouchDB document ID
✅ **Future Proof**: Enables saving translations back to CouchDB
✅ **Backward Compatible**: Existing `load_taxa() → translate()` workflow unchanged

The recommended workflow is now explicitly:
1. Extract taxa: `extract_taxa()`
2. Save to CouchDB: `save_taxa()`
3. Load with IDs: `load_taxa()`
4. Translate: `translate_descriptions_batch()`
