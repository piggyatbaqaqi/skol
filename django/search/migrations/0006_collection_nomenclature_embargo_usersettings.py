# Generated manually for collection nomenclature, embargo, and user settings

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.core.validators


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('search', '0005_alter_externalidentifier_unique_together'),
    ]

    operations = [
        # Add nomenclature field to Collection
        migrations.AddField(
            model_name='collection',
            name='nomenclature',
            field=models.CharField(
                blank=True,
                default='',
                help_text='Best guess taxon name for this collection',
                max_length=500
            ),
        ),
        # Add embargo_until field to Collection
        migrations.AddField(
            model_name='collection',
            name='embargo_until',
            field=models.DateTimeField(
                blank=True,
                null=True,
                help_text='Collection is private until this date (null=public)'
            ),
        ),
        # Create UserSettings model
        migrations.CreateModel(
            name='UserSettings',
            fields=[
                ('id', models.BigAutoField(
                    auto_created=True,
                    primary_key=True,
                    serialize=False,
                    verbose_name='ID'
                )),
                ('default_embargo_days', models.PositiveIntegerField(
                    default=0,
                    help_text='Default embargo period in days (0=public immediately)'
                )),
                ('default_embedding', models.CharField(
                    blank=True,
                    default='',
                    help_text='Preferred embedding model for search',
                    max_length=255
                )),
                ('default_k', models.PositiveIntegerField(
                    default=3,
                    help_text='Default number of search results',
                    validators=[
                        django.core.validators.MinValueValidator(1),
                        django.core.validators.MaxValueValidator(100)
                    ]
                )),
                ('feature_taxa_count', models.PositiveIntegerField(
                    default=6,
                    help_text='Number of taxa to retrieve for feature lists',
                    validators=[
                        django.core.validators.MinValueValidator(2),
                        django.core.validators.MaxValueValidator(50)
                    ]
                )),
                ('feature_max_tree_depth', models.PositiveIntegerField(
                    default=10,
                    help_text='Maximum depth for feature tree display',
                    validators=[
                        django.core.validators.MinValueValidator(1),
                        django.core.validators.MaxValueValidator(20)
                    ]
                )),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='settings',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'verbose_name': 'User Settings',
                'verbose_name_plural': 'User Settings',
            },
        ),
    ]
