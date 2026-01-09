# Python 3.11+ Compatibility

## The Problem

Python 3.11 removed `inspect.formatargspec()`, but several libraries in the ML stack still depend on it:

```
Django → search/views.py → sota_search.py → sentence_transformers
  → transformers → tensorflow → wrapt → inspect.formatargspec() ❌
```

The error you saw:
```
ImportError: cannot import name 'formatargspec' from 'inspect'
```

## The Solution

We implemented a **two-part compatibility strategy**:

### 1. Lazy Imports (Performance + Safety)

Heavy ML dependencies are imported **only when needed** (inside the API request handler), not at Django module load time.

**Benefits:**
- Fast Django startup (no TensorFlow loading)
- `/api/embeddings/` works immediately
- First search request loads ML stack (1-2 second delay)
- Subsequent requests are fast (libraries stay cached)

**Implementation:** [search/views.py:144](search/views.py#L144)
```python
def post(self, request):
    # ... validation ...

    try:
        # Lazy import - only loads when endpoint is called
        from sota_search import Experiment

        # ... use Experiment ...
```

### 2. Compatibility Shim (Cross-Version Support)

Provides `inspect.formatargspec()` for Python 3.11+ before importing libraries that need it.

**Benefits:**
- ✅ Works on Python 3.10 and earlier (uses native implementation)
- ✅ Works on Python 3.11+ (uses compatibility shim)
- ✅ Zero impact on older Python versions (shim not applied if native exists)

**Implementation:** [search/views.py:117-142](search/views.py#L117-L142)
```python
# Check if formatargspec exists
import inspect
if not hasattr(inspect, 'formatargspec'):
    # Define minimal implementation
    def formatargspec(args, varargs=None, varkw=None, defaults=None, ...):
        # Build argument list string: "(a, b=1, *args, **kwargs)"
        # ...
        return f"({', '.join(specs)})"

    # Monkey-patch it back into inspect module
    inspect.formatargspec = formatargspec

# NOW safe to import libraries that depend on formatargspec
from sota_search import Experiment
```

## How It Works

### On Python 3.10 and Earlier:
1. `inspect.formatargspec` already exists ✓
2. Shim check: `hasattr(inspect, 'formatargspec')` → `True`
3. Shim is **not applied** (native implementation used)
4. Libraries import normally

### On Python 3.11+:
1. `inspect.formatargspec` doesn't exist ✗
2. Shim check: `hasattr(inspect, 'formatargspec')` → `False`
3. Shim is **applied** (monkey-patched into `inspect`)
4. Libraries import successfully using shim

## Testing

Run the compatibility test:
```bash
cd skol/django
python3 test_compatibility.py
```

Expected output:
```
======================================================================
Python Compatibility Test
======================================================================
Python version: 3.13.7

Test 1: Native formatargspec availability
----------------------------------------------------------------------
Native formatargspec available: False
✓ Running on Python 3.11+ (formatargspec removed)

Test 2: Apply compatibility shim
----------------------------------------------------------------------
✓ Shim applied
formatargspec available now: True

Test 3: Test formatargspec functionality
----------------------------------------------------------------------
✓ formatargspec works correctly

Test 4: Test wrapt import (used by TensorFlow)
----------------------------------------------------------------------
✓ wrapt imported successfully

======================================================================
All compatibility tests passed!
======================================================================
```

## Verification

Test Django:
```bash
# Check configuration
python3 manage.py check

# Run migrations
python3 manage.py migrate

# Start server
./run_server.sh
```

All should work without import errors!

## Why This Approach?

### Alternative 1: Downgrade Python ❌
- Loses access to Python 3.11+ features
- Not future-proof
- Requires different environments

### Alternative 2: Wait for Library Updates ❌
- TensorFlow/wrapt updates are slow
- May never fully support Python 3.13
- Blocks development

### Our Approach: Compatibility Shim ✅
- Works on **all Python versions** (3.10, 3.11, 3.12, 3.13+)
- Minimal code (~15 lines)
- No external dependencies
- Future-proof
- Industry-standard technique (monkey-patching)

## Technical Details

The shim provides a minimal implementation that covers wrapt's usage:
```python
formatargspec(['self', 'x', 'y'], 'args', 'kwargs', (0, 0))
# Returns: "(self, x=0, y=0, *args, **kwargs)"
```

This is all `wrapt.decorators` needs to introspect function signatures for creating decorators.

## References

- [PEP 570 - Python Positional-Only Parameters](https://www.python.org/dev/peps/pep-0570/) (motivation for removal)
- [Python 3.11 Release Notes](https://docs.python.org/3/whatsnew/3.11.html#removed) (lists formatargspec removal)
- [wrapt Issue #179](https://github.com/GrahamDumpleton/wrapt/issues/179) (formatargspec dependency)
- [TensorFlow Python 3.11 Support](https://github.com/tensorflow/tensorflow/issues/58681)

## License

Part of the SKOL (Synoptic Key of Life) project.
