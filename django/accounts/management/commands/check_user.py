from django.core.management.base import BaseCommand
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'Check user details for debugging authentication issues'

    def add_arguments(self, parser):
        parser.add_argument('identifier', type=str, help='Username or email to check')

    def handle(self, *args, **options):
        identifier = options['identifier']

        # Try to find user by username or email
        users = User.objects.filter(username=identifier) | User.objects.filter(email=identifier)

        if not users.exists():
            self.stdout.write(self.style.ERROR(f'No user found with username or email: {identifier}'))
            return

        for user in users:
            self.stdout.write(self.style.SUCCESS('\n=== User Details ==='))
            self.stdout.write(f'Username: {user.username}')
            self.stdout.write(f'Email: {user.email}')
            self.stdout.write(f'First Name: {user.first_name}')
            self.stdout.write(f'Last Name: {user.last_name}')
            self.stdout.write(f'Is Active: {user.is_active}')
            self.stdout.write(f'Is Staff: {user.is_staff}')
            self.stdout.write(f'Is Superuser: {user.is_superuser}')
            self.stdout.write(f'Date Joined: {user.date_joined}')
            self.stdout.write(f'Last Login: {user.last_login}')

            # Show groups
            groups = user.groups.all()
            if groups:
                self.stdout.write(f'\nGroups: {", ".join([g.name for g in groups])}')
            else:
                self.stdout.write('\nGroups: None')

            # Show permissions
            if user.user_permissions.exists():
                self.stdout.write(f'\nPermissions: {user.user_permissions.count()} custom permissions')
            else:
                self.stdout.write('\nPermissions: None (using group permissions only)')

            self.stdout.write('')
