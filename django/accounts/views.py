from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives
from django.contrib import messages
from .forms import CustomUserCreationForm
from .tokens import email_verification_token


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
