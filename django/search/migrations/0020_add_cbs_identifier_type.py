from django.db import migrations


def add_cbs_identifier_type(apps, schema_editor):
    """Add CBS (Westerdijk Institute) culture collection identifier type."""
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.get_or_create(
        code='cbs',
        defaults={
            'name': 'CBS',
            'url_pattern': 'https://www.westerdijkinstitute.nl/Collections/?id={id}',
            'description': 'CBS (Westerdijk Fungal Biodiversity Institute) culture collection number',
        },
    )


def reverse_cbs_identifier_type(apps, schema_editor):
    """Remove CBS identifier type."""
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code='cbs').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0019_usage_event'),
    ]

    operations = [
        migrations.RunPython(add_cbs_identifier_type, reverse_cbs_identifier_type),
    ]
