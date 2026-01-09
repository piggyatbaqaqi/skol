# skol_compat: Python 3.11+ Compatibility Module

## Overview

The `skol_compat.py` module provides a centralized compatibility shim for Python 3.11+ to work with older ML libraries that haven't been updated for newer Python versions.

## Location

```
skol/skol_compat.py
```

## Problem Solved

Python 3.11 removed `inspect.formatargspec()`, but several libraries in the ML stack still depend on it:

```
TensorFlow → wrapt → inspect.formatargspec() ❌
```

This causes an `ImportError` when trying to use TensorFlow-dependent libraries on Python 3.11+.

## Solution

The `skol_compat` module provides a monkey-patch that adds `formatargspec()` back to the `inspect` module before it's needed.

### Key Features

1. **Automatic Application**: The shim is applied automatically when the module is imported
2. **Safe for All Python Versions**: No-op on Python 3.10 and earlier (native implementation used)
3. **Centralized**: Used by multiple scripts/modules in the SKOL project
4. **Minimal Implementation**: Only implements what `wrapt` actually needs

## Usage

### In Scripts (bin/)

```python
import sys
from pathlib import Path

# Add skol to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Apply compatibility shim (imported for side effects)
import skol_compat  # noqa: F401

# Now safe to import TensorFlow-dependent libraries
from dr_drafts_mycosearch.compute_embeddings import EmbeddingsComputer
```

### In Django Views

```python
import sys
from pathlib import Path

# Add skol to path
skol_path = Path(__file__).resolve().parent.parent.parent.parent / 'skol'
if str(skol_path) not in sys.path:
    sys.path.insert(0, str(skol_path))

# Inside request handler (lazy import)
def post(self, request):
    # Apply compatibility shim before importing ML libraries
    import skol_compat  # noqa: F401
    from src.sota_search import Experiment
    # ...
```

## Current Usage

The module is currently used by:

1. **bin/embed_taxa.py** - Embedding computation script
2. **django/search/views.py** - Django REST API search endpoint

## Implementation Details

### formatargspec() Shim

The shim provides a minimal implementation that covers the subset of functionality needed by `wrapt`:

```python
def formatargspec(args, varargs=None, varkw=None, defaults=None, ...):
    """Format function signature as a string."""
    # Example: ['a', 'b'], None, None, (1,) → "(a, b=1)"
    specs = []

    # Add regular arguments with defaults
    if defaults:
        firstdefault = len(args) - len(defaults)
    else:
        firstdefault = len(args)

    for i, arg in enumerate(args):
        if i >= firstdefault:
            specs.append(f"{arg}={repr(defaults[i - firstdefault])}")
        else:
            specs.append(arg)

    # Add *args and **kwargs if present
    if varargs:
        specs.append(f"*{varargs}")
    if varkw:
        specs.append(f"**{varkw}")

    return f"({', '.join(specs)})"
```

### Auto-Application

The shim is applied automatically at module import time:

```python
# At end of skol_compat.py
apply_formatargspec_shim()
```

This means you just need to import the module and the shim is applied.

## Testing

Test the compatibility module:

```bash
cd skol
python3 -c "
import skol_compat
import inspect
print(f'formatargspec available: {hasattr(inspect, \"formatargspec\")}')
result = inspect.formatargspec(['a', 'b'], 'args', 'kwargs', (1,))
print(f'Test result: {result}')
"
```

Expected output:
```
formatargspec available: True
Test result: (a, b=1, *args, **kwargs)
```

## Python Version Compatibility

| Python Version | Behavior |
|----------------|----------|
| 3.10 and earlier | Uses native `formatargspec` (shim not applied) |
| 3.11 | Uses compatibility shim |
| 3.12 | Uses compatibility shim |
| 3.13 | Uses compatibility shim |
| Future versions | Will use compatibility shim until libraries are updated |

## Alternative Approaches Considered

### 1. Downgrade Python ❌
- Loses access to Python 3.11+ features
- Not future-proof
- Requires different environments

### 2. Wait for Library Updates ❌
- TensorFlow/wrapt updates are slow
- May never fully support Python 3.13
- Blocks development

### 3. Patch Each Script Individually ❌
- Code duplication
- Hard to maintain
- Easy to forget in new scripts

### 4. Centralized Compatibility Module ✅ (Chosen)
- Works on all Python versions (3.10, 3.11, 3.12, 3.13+)
- Minimal code (~50 lines)
- No external dependencies
- Easy to maintain
- Industry-standard technique (monkey-patching)

## Maintenance

### When to Update

This module should be reviewed when:

1. **TensorFlow/wrapt update**: Check if they've added Python 3.11+ support
2. **New Python version**: Test compatibility with new Python releases
3. **New error patterns**: If different import errors occur

### How to Test

```bash
# Test on Python 3.13
python3.13 -m pytest tests/

# Test embedding script
python3.13 bin/embed_taxa.py --help

# Test Django
cd django
python3.13 manage.py check
```

### When to Remove

This module can be removed when:
- TensorFlow and wrapt both support Python 3.11+ natively
- All dependencies have been updated
- Tests pass without the shim

Monitor these issues:
- [wrapt Issue #179](https://github.com/GrahamDumpleton/wrapt/issues/179)
- [TensorFlow Python 3.11 Support](https://github.com/tensorflow/tensorflow/issues/58681)

## License

Part of the SKOL (Synoptic Key of Life) project.
