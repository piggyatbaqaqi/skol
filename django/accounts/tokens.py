from django.contrib.auth.tokens import PasswordResetTokenGenerator


class EmailVerificationTokenGenerator(PasswordResetTokenGenerator):
    """
    Token generator for email verification.
    Uses Django's PasswordResetTokenGenerator as base for security.
    """
    def _make_hash_value(self, user, timestamp):
        # Include user's active state to invalidate token after verification
        return f"{user.pk}{timestamp}{user.is_active}{user.email}"


email_verification_token = EmailVerificationTokenGenerator()
