from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User, Group


@receiver(post_save, sender=User)
def create_user_group(sender, instance, created, **kwargs):
    """
    Create a group with the same name as the username when user is activated.
    Only runs when user becomes active (email verified).
    """
    # Check if user just became active (not created, but is_active changed)
    if instance.is_active and not created:
        # Check if the group already exists
        group_name = instance.username
        group, group_created = Group.objects.get_or_create(name=group_name)

        # Add user to their group if not already in it
        if not instance.groups.filter(name=group_name).exists():
            instance.groups.add(group)
