# Generated migration for Fungarium identifier type

from django.db import migrations, models


def create_fungarium_identifier_type(apps, schema_editor):
    """Create the Fungarium identifier type."""
    IdentifierType = apps.get_model('search', 'IdentifierType')

    # The url_pattern is a marker - actual URLs are built dynamically from Redis data
    IdentifierType.objects.create(
        code='fungarium',
        name='Fungarium',
        url_pattern='__FUNGARIUM__:{id}',  # Special marker, URL built from Redis
        description='Herbarium/Fungarium accession number (Index Herbariorum)'
    )


def reverse_fungarium_identifier_type(apps, schema_editor):
    """Remove the Fungarium identifier type."""
    IdentifierType = apps.get_model('search', 'IdentifierType')
    IdentifierType.objects.filter(code='fungarium').delete()


class Migration(migrations.Migration):

    dependencies = [
        ("search", "0003_add_collection_notes"),
    ]

    operations = [
        # Add fungarium_code field to ExternalIdentifier
        migrations.AddField(
            model_name='externalidentifier',
            name='fungarium_code',
            field=models.CharField(
                blank=True,
                default='',
                max_length=20,
                help_text='Index Herbariorum code for fungarium identifiers'
            ),
        ),
        # Create the Fungarium identifier type
        migrations.RunPython(
            create_fungarium_identifier_type,
            reverse_fungarium_identifier_type
        ),
    ]
