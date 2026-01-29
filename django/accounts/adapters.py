"""
Custom allauth adapters for SKOL.

These adapters customize the social account authentication behavior,
particularly for linking social accounts to existing users.
"""

from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.contrib.auth import get_user_model


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Custom adapter to automatically link social accounts to existing users
    when they have matching email addresses.

    This allows existing users to sign in with OAuth providers (GitHub, Google, ORCID)
    and have their social account automatically linked to their existing account.
    """

    def pre_social_login(self, request, sociallogin):
        """
        Called after a user successfully authenticates via a social provider,
        but before the login is completed.

        If a user with the same email already exists, connect the social account
        to that existing user instead of creating a new account.
        """
        # If the social account is already linked to a user, nothing to do
        if sociallogin.is_existing:
            return

        # Get email from the social account
        email = None

        # Try to get email from account data
        if sociallogin.account.extra_data:
            email = sociallogin.account.extra_data.get('email')

        # Also check email addresses from the social login
        if not email and sociallogin.email_addresses:
            # Get the primary email if available
            for email_address in sociallogin.email_addresses:
                if email_address.primary:
                    email = email_address.email
                    break
            # Fall back to first email if no primary
            if not email and sociallogin.email_addresses:
                email = sociallogin.email_addresses[0].email

        if not email:
            return

        # Find existing user with this email
        User = get_user_model()
        try:
            user = User.objects.get(email__iexact=email)
            # Connect the social account to the existing user
            sociallogin.connect(request, user)
        except User.DoesNotExist:
            # No existing user with this email, proceed with normal signup
            pass
        except User.MultipleObjectsReturned:
            # Multiple users with same email (shouldn't happen, but handle it)
            # Let allauth handle this case normally
            pass
