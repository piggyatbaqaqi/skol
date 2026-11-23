"""
Move stored "Total Results (k)" settings from the old default to the new
one (user, 2026-11-23).

Every signed-in user's settings row stored the old default, 20, so
changing the model default reached only new accounts.  A stored 20
cannot be told apart from a chosen one; the user decided to move them
all.  Called by migration 0026 with the historical model.
"""
from typing import Any

OLD_DEFAULT = 20
NEW_DEFAULT = 200


def move_stored_default(user_settings_model: Any) -> int:
    """Set ``default_k`` to NEW_DEFAULT on every row holding OLD_DEFAULT.

    Returns the number of rows changed.  Other values are left alone.
    """
    return int(user_settings_model.objects.filter(
        default_k=OLD_DEFAULT).update(default_k=NEW_DEFAULT))
