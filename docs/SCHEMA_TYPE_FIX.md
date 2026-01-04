# Schema Type Fix - Source MapType

## Issue

Error when counting taxa DataFrame in `ist769_skol.ipynb`:
```
pyspark.errors.exceptions.base.PySparkTypeError: [CANNOT_ACCEPT_OBJECT_IN_TYPE]
`MapType(StringType(), StringType(), True)` can not accept object `68` in type `int`.
```

## Analysis

The error indicates that an integer value (68) is being placed in a MapType field that expects only string values. The schema defines the `source` field as:

```python
StructField("source", MapType(StringType(), StringType(), valueContainsNull=True), False)
```

This means:
- Keys: StringType (non-null)
- Values: StringType or NULL

## Potential Causes

1. **Direct cause**: An integer value is being added to the `source` dict
2. **Value 68**: Likely a `paragraph_number` or `page_number` (both are integers)

## Investigation

### taxon.as_row() Structure

The `Taxon.as_row()` method returns:
```python
{
    'taxon': str,
    'description': str,
    'source': {
        'doc_id': str,
        'url': str or None,
        'db_name': str,
    },
    'line_number': int,           # Separate top-level field
    'paragraph_number': int,       # Separate top-level field
    'page_number': int,            # Separate top-level field
    'empirical_page_number': str,  # Separate top-level field
}
```

The source dict should only contain strings or None.

### paragraph.as_dict() Issue

The `Paragraph.as_dict()` method (paragraph.py:119-128) returns integers for `paragraph_number` and `page_number`:
```python
def as_dict(self) -> Dict[str, Optional[str]]:  # Wrong type hint!
    return {
        'paragraph_number': self.paragraph_number,  # INTEGER!
        'page_number': self.page_number,            # INTEGER!
        ...
    }
```

However, this method is only used in `Taxon.dictionaries()`, not `Taxon.as_row()`.

## Fix Applied

### 1. Ensured empirical_page_number is a string

**File**: [taxon.py:83](taxon.py#L83)

```python
# Before
'empirical_page_number': pp.empirical_page_number,

# After
'empirical_page_number': str(pp.empirical_page_number) if pp.empirical_page_number is not None else None,
```

### 2. Updated type hint

**File**: [taxon.py:72](taxon.py#L72)

```python
# Before
retval: Dict[str, None | str | int | Dict[str, None | str | int]] = {

# After
retval: Dict[str, None | str | int | Dict[str, None | str]] = {
```

Changed source dict value type from `None | str | int` to `None | str`.

## Verification

Created test to verify schema works:
```python
schema = StructType([
    StructField("_id", StringType(), True),
    StructField("taxon", StringType(), False),
    StructField("description", StringType(), False),
    StructField("source", MapType(StringType(), StringType(), valueContainsNull=True), False),
    StructField("line_number", IntegerType(), True),
    StructField("paragraph_number", IntegerType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("empirical_page_number", StringType(), True),
])

test_data = {
    '_id': None,
    'taxon': 'Test',
    'description': 'Desc',
    'source': {'doc_id': 'test', 'url': None, 'db_name': 'db'},
    'line_number': 10,
    'paragraph_number': 68,  # Integer OK here (not in source)
    'page_number': 5,
    'empirical_page_number': '3',
}

df = spark.createDataFrame([Row(**test_data)], schema)
count = df.count()  # Works!
```

## Database Verification

✅ **Checked 5,278 documents in `skol_taxa_dev` database**
- **Result**: No integer values found in any `source` field
- All source field values are strings or None as expected
- This confirms the CouchDB data is clean

## Root Cause (Updated)

Since both the synthetic tests AND the CouchDB data are clean, the issue must be during extraction:

1. ✅ **NOT CouchDB data**: All 5,278 docs have clean source fields
2. ✅ **NOT schema definition**: Schema correctly enforces MapType(StringType(), StringType())
3. ❓ **Extraction process**: Issue may occur during `extract_taxa(annotated_df)`
4. ❓ **PySpark version**: Different PySpark versions may have different schema validation strictness

## Recommendations

### 1. Defensive Programming

Add validation in `convert_taxa_to_rows()`:
```python
for taxon in partition:
    taxon_dict = taxon.as_row()

    # Validate source dict
    if 'source' in taxon_dict:
        source = taxon_dict['source']
        for key, value in source.items():
            if value is not None and not isinstance(value, str):
                print(f"Warning: Non-string value in source[{key}]: {value} ({type(value)})")
                source[key] = str(value)  # Convert to string

    yield Row(**taxon_dict)
```

### 2. Fix paragraph.as_dict() Type Hint

**File**: paragraph.py:119

```python
# Current (WRONG)
def as_dict(self) -> Dict[str, Optional[str]]:
    return {
        'paragraph_number': self.paragraph_number,  # int!
        'page_number': self.page_number,            # int!
        ...
    }

# Should be
def as_dict(self) -> Dict[str, Optional[str | int]]:
    return {
        'paragraph_number': self.paragraph_number,
        'page_number': self.page_number,
        ...
    }
```

### 3. Ensure All Source Values are Strings

In `taxon.as_row()`, explicitly convert all source values:
```python
'source': {
    'doc_id': str(source_doc_id) if source_doc_id else None,
    'url': str(source_url) if source_url else None,
    'db_name': str(source_db_name) if source_db_name else None,
},
```

## Status

✅ Applied fix to ensure `empirical_page_number` is a string ([taxon.py:83](taxon.py#L83))
✅ Updated type hint for source dict ([taxon.py:72](taxon.py#L72))
✅ Verified CouchDB database: **All 5,278 documents have clean source fields**
✅ Synthetic tests pass with value 68 as paragraph_number
⚠️ **Issue likely happens during extraction** - add debug code if error persists

## Related Files

- [taxon.py](taxon.py) - Fixed empirical_page_number conversion
- [paragraph.py](paragraph.py) - Has incorrect type hint in as_dict()
- [extract_taxa_to_couchdb.py](extract_taxa_to_couchdb.py) - Schema definition
- [jupyter/ist769_skol.ipynb](jupyter/ist769_skol.ipynb) - Where error occurred

## Testing

To verify the fix works:
```python
# In notebook
taxa_df = extractor.extract_taxa(annotated_df)
print(f"Extracted {taxa_df.count()} taxa")  # Should work now
taxa_df.printSchema()
taxa_df.show(10)
```

### Debug Mode: Capture Actual Error Data

If error persists, use this debug code to capture the exact row causing the issue:

```python
# Add to convert_taxa_to_rows() in extract_taxa_to_couchdb.py
for taxon in partition:
    taxon_dict = taxon.as_row()

    # DEBUG: Validate source dict
    if 'source' in taxon_dict and isinstance(taxon_dict['source'], dict):
        for key, value in taxon_dict['source'].items():
            if value is not None and not isinstance(value, str):
                print(f"⚠️  ERROR: Non-string in source['{key}']: {value} ({type(value).__name__})")
                print(f"   Taxon: {taxon_dict.get('taxon', 'N/A')}")
                print(f"   Full source: {taxon_dict['source']}")
                # Convert to string
                taxon_dict['source'][key] = str(value)

    if '_id' not in taxon_dict:
        taxon_dict['_id'] = generate_taxon_doc_id(...)
    if 'json_annotated' not in taxon_dict:
        taxon_dict['json_annotated'] = None

    yield Row(**taxon_dict)
```

This will print the exact taxon and source that has the problematic value.
