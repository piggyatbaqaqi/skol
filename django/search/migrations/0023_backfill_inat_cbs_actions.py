"""Backfill IdentifierType.actions for iNat and CBS.

Generalises the previously-hardcoded ``identifier_type_code === 'inat'``
branch in collection_detail.html into a data-driven action list. Two
action kinds are emitted by this backfill:

* ``external_post_button`` — for iNat: render a button that POSTs the
  collection's description to the iNaturalist comment endpoint.
* ``clipboard_on_link_click`` — for CBS: the wi.knaw.nl SPA can't be
  linked to a strain page directly, so clicking the URL also copies a
  ``"CBS {value}"`` string for the user to paste into the on-site
  search form.

Reversible: clears the actions list on both records.
"""

from django.db import migrations


INAT_ACTIONS = [
    {
        "kind": "external_post_button",
        "endpoint": "post_inat_comment",
        "label": "Post to iNat",
        "requires_owner": True,
        "requires_description": True,
    },
]

CBS_ACTIONS = [
    {
        "kind": "clipboard_on_link_click",
        "format": "{name} {value}",
    },
]


def backfill_actions(apps, schema_editor):
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code='inat').update(actions=INAT_ACTIONS)
    IdentifierType.objects.filter(code='cbs').update(actions=CBS_ACTIONS)


def reverse_backfill(apps, schema_editor):
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code__in=('inat', 'cbs')).update(actions=[])


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0022_identifier_type_actions'),
    ]

    operations = [
        migrations.RunPython(backfill_actions, reverse_backfill),
    ]
