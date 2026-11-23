"""Move stored default_k settings of 20, the old default, to 200
(user, 2026-11-23).  The reverse does nothing: once moved, nothing
records which users had 20."""
from typing import Any

from django.db import migrations

from search.default_k import move_stored_default


def forwards(apps: Any, schema_editor: Any) -> None:
    move_stored_default(apps.get_model('search', 'UserSettings'))


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0025_usersettings_default_k_200'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
