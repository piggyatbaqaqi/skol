"""Python 3.14 / Django 4.2 compatibility shim for template.Context.__copy__.

**Remove this module when Django is upgraded to 5.2+.**

Django 4.2 doesn't officially support Python 3.14.  Symptom: every
admin page (and any view that uses ``simple_tag``, including some of
DRF's browsable API) renders with::

    AttributeError: 'super' object has no attribute 'dicts' and
    no __dict__ for setting new attributes

The trigger is ``django.template.context.BaseContext.__copy__``::

    def __copy__(self):
        duplicate = super().__copy__()
        duplicate.dicts = self.dicts[:]
        return duplicate

In Python 3.13 and earlier, ``super().__copy__()`` returned a fresh
instance with the right type.  In Python 3.14 the resolution went
to ``object`` and the call returns a ``super`` proxy with no
``__dict__``, so the next line (``duplicate.dicts = ...``) raises.

The fix here is what Django 5.x already does internally:
construct the duplicate directly via ``cls.__new__(cls)``, then copy
``__dict__`` and the ``dicts`` list.  This restores the contract
that ``Context.__copy__`` and ``RequestContext.__copy__`` (both
subclass overrides that call ``super().__copy__()``) depend on.

The patch is applied at module import time, so it must be imported
from ``settings.py`` (or earlier) — before any template rendering
happens.
"""

from django.template.context import BaseContext


def _basecontext_copy_py314(self):
    cls = type(self)
    duplicate = cls.__new__(cls)
    duplicate.__dict__.update(self.__dict__)
    duplicate.dicts = self.dicts[:]
    return duplicate


# Idempotent: re-importing the module doesn't double-patch.
if getattr(BaseContext.__copy__, '_skol_py314_patched', False) is not True:
    _basecontext_copy_py314._skol_py314_patched = True
    BaseContext.__copy__ = _basecontext_copy_py314
