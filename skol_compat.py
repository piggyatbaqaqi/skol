"""
Python 3.11+ compatibility utilities for SKOL project.

This module provides compatibility shims for libraries that haven't been
updated for Python 3.11+ changes.
"""

import inspect


def apply_formatargspec_shim():
    """
    Apply formatargspec compatibility shim for Python 3.11+.

    Python 3.11 removed inspect.formatargspec(), but some libraries in the
    ML stack (notably wrapt, used by TensorFlow) still depend on it.

    This function monkey-patches a minimal implementation back into the
    inspect module if it's missing. It's safe to call multiple times and
    on older Python versions (it will be a no-op if formatargspec already exists).

    Usage:
        import skol_compat
        skol_compat.apply_formatargspec_shim()

        # Now safe to import TensorFlow-dependent libraries
        from dr_drafts_mycosearch.compute_embeddings import EmbeddingsComputer
    """
    if hasattr(inspect, 'formatargspec'):
        # Already exists (Python 3.10 or earlier, or already patched)
        return

    def formatargspec(args, varargs=None, varkw=None, defaults=None,
                     kwonlyargs=(), kwonlydefaults={}, annotations={}):
        """
        Compatibility shim for deprecated inspect.formatargspec().

        Formats a function signature as a string, e.g. "(a, b=1, *args, **kwargs)".

        This implementation covers the subset of functionality needed by wrapt.
        It does not support all features of the original formatargspec.

        Args:
            args: List of argument names
            varargs: Name of *args parameter (if any)
            varkw: Name of **kwargs parameter (if any)
            defaults: Tuple of default values for trailing arguments
            kwonlyargs: Ignored (kwonly args not fully supported)
            kwonlydefaults: Ignored (kwonly args not fully supported)
            annotations: Ignored (annotations not supported)

        Returns:
            String representation of the argument spec
        """
        specs = []

        # Add regular arguments with defaults
        if defaults:
            firstdefault = len(args) - len(defaults)
        else:
            firstdefault = len(args)

        for i, arg in enumerate(args):
            if i >= firstdefault:
                # Argument has a default value
                default_val = defaults[i - firstdefault]
                specs.append(f"{arg}={repr(default_val)}")
            else:
                # Argument has no default
                specs.append(arg)

        # Add *args
        if varargs:
            specs.append(f"*{varargs}")

        # Add **kwargs
        if varkw:
            specs.append(f"**{varkw}")

        return f"({', '.join(specs)})"

    # Monkey-patch it into the inspect module
    inspect.formatargspec = formatargspec


# Automatically apply the shim when this module is imported
# This makes it transparent - just import skol_compat and you're protected
apply_formatargspec_shim()
