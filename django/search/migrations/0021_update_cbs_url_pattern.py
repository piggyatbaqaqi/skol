from django.db import migrations

CBS_URL = 'https://wi.knaw.nl/fungal_table/CBS+{id}'
OLD_CBS_URL = 'https://www.westerdijkinstitute.nl/Collections/?id={id}'


def update_cbs_url(apps, schema_editor):
    """Update CBS URL pattern to Westerdijk wi.knaw.nl search page."""
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code='cbs').update(url_pattern=CBS_URL)


def reverse_cbs_url(apps, schema_editor):
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code='cbs').update(url_pattern=OLD_CBS_URL)


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0020_add_cbs_identifier_type'),
    ]

    operations = [
        migrations.RunPython(update_cbs_url, reverse_cbs_url),
    ]
