"""
Django signals for Collection model.

Handles:
- Automatic sync to CouchDB when collections are created, updated, or deleted.
- Auto-adding new collections to the owner's default projects.
"""
import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from .models import Collection

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Collection)
def on_collection_save(sender, instance, created, **kwargs):
    """
    Sync collection to CouchDB after save.

    Runs asynchronously to avoid blocking the request.
    """
    from .couchdb_sync import sync_collection_to_couchdb

    try:
        sync_collection_to_couchdb(instance.collection_id)
    except Exception as e:
        # Log but don't fail the save
        logger.error(f"Failed to sync collection {instance.collection_id}: {e}")


@receiver(post_delete, sender=Collection)
def on_collection_delete(sender, instance, **kwargs):
    """
    Delete collection from CouchDB after deletion.
    """
    from .couchdb_sync import delete_collection_from_couchdb

    try:
        delete_collection_from_couchdb(instance.collection_id)
    except Exception as e:
        # Log but don't fail the delete
        logger.error(f"Failed to delete collection {instance.collection_id}: {e}")


@receiver(post_save, sender=Collection)
def on_collection_created_add_to_default_projects(
    sender, instance, created, **kwargs
):
    """Add newly created collections to the owner's default projects."""
    if not created:
        return
    from .models import UserSettings, CollectionProject

    try:
        user_settings = UserSettings.objects.get(user=instance.owner)
    except UserSettings.DoesNotExist:
        return

    for project in user_settings.default_projects.all():
        try:
            CollectionProject.objects.get_or_create(
                collection=instance,
                project=project,
                defaults={'added_by': instance.owner},
            )
        except Exception as e:
            logger.error(
                "Failed to add collection %s to default project %s: %s",
                instance.collection_id,
                project,
                e,
            )
