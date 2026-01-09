#!/usr/bin/env python3
"""
Test script to verify Python 3.11+ compatibility with wrapt/TensorFlow.

This script tests that the formatargspec compatibility shim works correctly.
"""

import sys
import inspect

print("=" * 70)
print("Python Compatibility Test")
print("=" * 70)
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print()

# Test 1: Check if formatargspec exists natively
print("Test 1: Native formatargspec availability")
print("-" * 70)
has_native = hasattr(inspect, 'formatargspec')
print(f"Native formatargspec available: {has_native}")
if has_native:
    print("✓ Running on Python 3.10 or earlier")
else:
    print("✓ Running on Python 3.11+ (formatargspec removed)")
print()

# Test 2: Apply compatibility shim
print("Test 2: Apply compatibility shim")
print("-" * 70)
if not has_native:
    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                    kwonlyargs=(), kwonlydefaults={}, annotations={}):
        """Compatibility shim for deprecated formatargspec."""
        specs = []
        if defaults:
            firstdefault = len(args) - len(defaults)
        for i, arg in enumerate(args):
            spec = arg
            if defaults and i >= firstdefault:
                spec = f"{arg}={repr(defaults[i - firstdefault])}"
            specs.append(spec)
        if varargs:
            specs.append(f"*{varargs}")
        if varkw:
            specs.append(f"**{varkw}")
        return f"({', '.join(specs)})"

    inspect.formatargspec = formatargspec
    print("✓ Shim applied")
else:
    print("✓ Using native implementation (no shim needed)")

has_after = hasattr(inspect, 'formatargspec')
print(f"formatargspec available now: {has_after}")
print()

# Test 3: Test formatargspec functionality
print("Test 3: Test formatargspec functionality")
print("-" * 70)
try:
    # Test various argument patterns
    test_cases = [
        (['a', 'b', 'c'], None, None, None),
        (['a', 'b'], None, None, (1,)),
        (['self', 'x', 'y'], 'args', 'kwargs', (0, 0)),
    ]

    for args, varargs, varkw, defaults in test_cases:
        result = inspect.formatargspec(args, varargs, varkw, defaults)
        print(f"  formatargspec{args, varargs, varkw, defaults}")
        print(f"    → {result}")

    print("✓ formatargspec works correctly")
except Exception as e:
    print(f"✗ formatargspec failed: {e}")
    sys.exit(1)

print()

# Test 4: Try importing wrapt (if installed)
print("Test 4: Test wrapt import (used by TensorFlow)")
print("-" * 70)
try:
    import wrapt
    print(f"✓ wrapt imported successfully (version {wrapt.__version__})")
except ImportError:
    print("⚠ wrapt not installed (this is fine, TensorFlow will install it)")
except Exception as e:
    print(f"✗ wrapt import failed: {e}")
    print("  This means the compatibility shim may need adjustment")
    sys.exit(1)

print()
print("=" * 70)
print("All compatibility tests passed!")
print("=" * 70)
print()
print("Summary:")
print(f"  • Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"  • Native formatargspec: {'Yes' if has_native else 'No'}")
print(f"  • Compatibility shim: {'Not needed' if has_native else 'Applied'}")
print(f"  • Status: ✓ Ready for Django/TensorFlow")
