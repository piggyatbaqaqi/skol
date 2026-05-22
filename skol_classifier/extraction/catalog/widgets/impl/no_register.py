"""File without a register method.

Used by ``test_load`` to verify the autoloader skips ``impl/`` —
without that exclusion the missing ``register`` raises
``CatalogNoRegister``.
"""
