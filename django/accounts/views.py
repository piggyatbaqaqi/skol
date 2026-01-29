from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.contrib import messages
from django.contrib.auth.views import PasswordResetView
from .forms import CustomUserCreationForm
from .tokens import email_verification_token
import logging

logger = logging.getLogger(__name__)


def register(request):
    """User registration view with email verification."""
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            # Create user but set inactive until email verified
            user = form.save(commit=False)
            user.is_active = False
            user.save()

            # Generate verification token
            current_site = get_current_site(request)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = email_verification_token.make_token(user)

            # Build verification URL
            verify_url = f"http://{current_site.domain}/accounts/verify-email/{uid}/{token}/"

            # Render email templates
            context = {
                'user': user,
                'domain': current_site.domain,
                'verify_url': verify_url,
            }
            text_content = render_to_string('emails/verify_email.txt', context)
            html_content = render_to_string('emails/verify_email.html', context)

            # Send email
            email = EmailMultiAlternatives(
                subject='Verify your SKOL account',
                body=text_content,
                to=[user.email],
            )
            email.attach_alternative(html_content, "text/html")
            email.send()

            return redirect('accounts:register_complete')
    else:
        form = CustomUserCreationForm()

    return render(request, 'accounts/register.html', {'form': form})


def register_complete(request):
    """Show 'check your email' message after registration."""
    return render(request, 'accounts/register_complete.html')


def verify_email(request, uidb64, token):
    """Verify email address and activate user account."""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and email_verification_token.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, 'Your email has been verified! You can now log in.')
        return render(request, 'accounts/verify_email.html', {'success': True})
    else:
        messages.error(request, 'The verification link is invalid or has expired.')
        return render(request, 'accounts/verify_email.html', {'success': False})


class CustomPasswordResetView(PasswordResetView):
    """Custom password reset view with detailed logging."""

    def form_valid(self, form):
        """Override to add logging before sending email."""
        email = form.cleaned_data['email']
        logger.info(f"[PasswordReset] Password reset requested for email: {email}")

        # Check if any users with this email exist
        users = User.objects.filter(email=email)
        logger.info(f"[PasswordReset] Found {users.count()} user(s) with email: {email}")

        for user in users:
            logger.info(f"[PasswordReset] User: {user.username}, Active: {user.is_active}, Email: {user.email}")

        # Call parent form_valid which sends the email
        result = super().form_valid(form)

        logger.info(f"[PasswordReset] Email sending completed for: {email}")

        return result


def resend_verification(request):
    """Resend email verification link for inactive users."""
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()

        if not email:
            messages.error(request, 'Please enter your email address.')
            return redirect('accounts:resend_verification')

        # Rate limiting: Check if user has requested too recently
        last_request_key = f'resend_verification_{email}'
        last_request_time = request.session.get(last_request_key)

        if last_request_time:
            from datetime import datetime, timedelta
            last_time = datetime.fromisoformat(last_request_time)
            if datetime.now() - last_time < timedelta(minutes=5):
                messages.error(request, 'Please wait 5 minutes before requesting another verification email.')
                return redirect('accounts:resend_verification')

        # Find inactive user with this email
        try:
            user = User.objects.get(email=email, is_active=False)

            logger.info(f"[ResendVerification] Resending verification email for: {user.username} ({email})")

            # Generate new verification token
            current_site = get_current_site(request)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = email_verification_token.make_token(user)

            # Build verification URL
            verify_url = f"http://{current_site.domain}/accounts/verify-email/{uid}/{token}/"

            # Render email templates
            context = {
                'user': user,
                'domain': current_site.domain,
                'verify_url': verify_url,
            }
            text_content = render_to_string('emails/verify_email.txt', context)
            html_content = render_to_string('emails/verify_email.html', context)

            # Send email
            email_msg = EmailMultiAlternatives(
                subject='Verify your SKOL account',
                body=text_content,
                to=[user.email],
            )
            email_msg.attach_alternative(html_content, "text/html")
            email_msg.send()

            # Update rate limiting timestamp
            from datetime import datetime
            request.session[last_request_key] = datetime.now().isoformat()

            messages.success(request, 'Verification email sent! Please check your inbox.')
            logger.info(f"[ResendVerification] Successfully sent verification email to: {email}")

        except User.DoesNotExist:
            # Don't reveal whether user exists (security best practice)
            messages.success(request, 'If an inactive account exists with this email, a verification link has been sent.')
            logger.info(f"[ResendVerification] No inactive user found for email: {email}")
        except Exception as e:
            logger.error(f"[ResendVerification] Error sending verification email: {e}")
            messages.error(request, 'An error occurred. Please try again later.')

        return redirect('accounts:login')

    return render(request, 'accounts/resend_verification.html')


@login_required
def social_connections(request):
    """
    Display and manage connected social accounts.

    Shows which OAuth providers (GitHub, Google, ORCID) are connected
    to the user's account, and allows them to connect/disconnect providers.
    """
    # Import here to avoid circular imports and handle case when allauth not installed
    try:
        from allauth.socialaccount.models import SocialAccount
        social_accounts = SocialAccount.objects.filter(user=request.user)
    except ImportError:
        social_accounts = []

    return render(request, 'accounts/social_connections.html', {
        'social_accounts': social_accounts,
    })
