"""Views for contact and feedback forms."""
import json
import logging

from django.conf import settings
from django.core.mail import EmailMessage
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils.html import mark_safe

from .forms import ContactForm, FeedbackForm
from .github import get_github_token, create_github_issue, format_feedback_issue

logger = logging.getLogger(__name__)


def contact_view(request):
    """Handle general contact form submissions."""
    initial = {}

    # Pre-populate email if user is logged in
    if request.user.is_authenticated and request.user.email:
        initial['email'] = request.user.email

    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            reply_to = form.cleaned_data.get('email')

            # Build email body
            body_parts = [message, '', '---']
            if request.user.is_authenticated:
                body_parts.append(f'User: {request.user.username}')
            if reply_to:
                body_parts.append(f'Reply-To: {reply_to}')
            body = '\n'.join(body_parts)

            try:
                # Debug: log email settings
                logger.debug(
                    f"Email config: HOST={settings.EMAIL_HOST}, PORT={settings.EMAIL_PORT}, "
                    f"USER={settings.EMAIL_HOST_USER}, TLS={settings.EMAIL_USE_TLS}, "
                    f"BACKEND={settings.EMAIL_BACKEND}"
                )

                email = EmailMessage(
                    subject=f'[SKOL Contact] {subject}',
                    body=body,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[settings.CONTACT_EMAIL],
                    reply_to=[reply_to] if reply_to else None,
                )
                email.send(fail_silently=False)

                logger.info(
                    f"Contact email sent: subject='{subject}', "
                    f"user={request.user.username if request.user.is_authenticated else 'anonymous'}"
                )
                messages.success(request, 'Your message has been sent. Thank you!')
                return redirect('contact:contact_success')

            except Exception as e:
                logger.error(f"Failed to send contact email: {e}")
                messages.error(request, 'Failed to send message. Please try again later.')
    else:
        form = ContactForm(initial=initial)

    return render(request, 'contact/contact.html', {'form': form})


def feedback_view(request):
    """Handle feedback/bug report form submissions."""
    initial = {}

    # Pre-populate email if user is logged in
    if request.user.is_authenticated and request.user.email:
        initial['email'] = request.user.email

    # Get referring page URL
    referer = request.GET.get('url') or request.META.get('HTTP_REFERER', '')
    if referer:
        initial['page_url'] = referer

    # Check if user has GitHub credentials for issue creation
    github_token = get_github_token(request.user)
    has_github = github_token is not None

    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback_type = form.cleaned_data['feedback_type']
            message = form.cleaned_data['message']
            page_url = form.cleaned_data.get('page_url', '')
            reply_to = form.cleaned_data.get('email')
            browser_info_raw = form.cleaned_data.get('browser_info', '')

            # Parse browser info JSON
            browser_info = None
            if browser_info_raw:
                try:
                    browser_info = json.loads(browser_info_raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse browser_info JSON")

            # Try to create GitHub issue if user has linked GitHub account
            if github_token:
                title, body, labels = format_feedback_issue(
                    feedback_type=feedback_type,
                    message=message,
                    page_url=page_url,
                    user=request.user,
                    email=reply_to,
                    browser_info=browser_info,
                )

                result = create_github_issue(github_token, title, body, labels)

                if result['success']:
                    logger.info(
                        f"Feedback submitted as GitHub issue #{result['issue_number']}: "
                        f"type={feedback_type}, user={request.user.username}"
                    )
                    messages.success(
                        request,
                        mark_safe(
                            f'Your feedback has been submitted as GitHub issue. '
                            f'<a href="{result["issue_url"]}" target="_blank">View issue</a>'
                        )
                    )
                    return redirect('contact:feedback_success')
                else:
                    # Fall back to email if GitHub fails
                    logger.warning(
                        f"GitHub issue creation failed, falling back to email: {result['error']}"
                    )

            # Fall back to email submission
            body_parts = [
                f'Feedback Type: {feedback_type}',
                f'Page URL: {page_url or "Not specified"}',
                '',
                'Description:',
                message,
                '',
                '---',
            ]
            if request.user.is_authenticated:
                body_parts.append(f'User: {request.user.username}')
            if reply_to:
                body_parts.append(f'Reply-To: {reply_to}')

            # Add browser environment info
            if browser_info:
                body_parts.extend(['', 'Browser Environment:'])
                if browser_info.get('browser'):
                    body_parts.append(f"  Browser: {browser_info.get('browser')}")
                if browser_info.get('os'):
                    os_str = browser_info.get('os')
                    if browser_info.get('osVersion'):
                        os_str += f" {browser_info.get('osVersion')}"
                    body_parts.append(f"  OS: {os_str}")
                body_parts.append(f"  Mobile: {'Yes' if browser_info.get('isMobile') else 'No'}")
                body_parts.append(f"  Screen: {browser_info.get('screenWidth')}x{browser_info.get('screenHeight')}")
                body_parts.append(f"  Viewport: {browser_info.get('viewportWidth')}x{browser_info.get('viewportHeight')}")
                body_parts.append(f"  Touch: {'Yes' if browser_info.get('touchSupport') else 'No'}")
                if browser_info.get('connectionType'):
                    body_parts.append(f"  Connection: {browser_info.get('connectionType')}")
                body_parts.append(f"  User Agent: {browser_info.get('userAgent', 'N/A')}")

            body = '\n'.join(body_parts)

            # Create subject based on feedback type
            type_labels = {
                'bug': 'Bug Report',
                'feature': 'Feature Request',
                'usability': 'Usability Issue',
                'other': 'Feedback',
            }
            subject = f'[SKOL Feedback] {type_labels.get(feedback_type, "Feedback")}'

            try:
                logger.debug(
                    f"Email config: HOST={settings.EMAIL_HOST}, PORT={settings.EMAIL_PORT}, "
                    f"USER={settings.EMAIL_HOST_USER}, TLS={settings.EMAIL_USE_TLS}, "
                    f"BACKEND={settings.EMAIL_BACKEND}"
                )

                email = EmailMessage(
                    subject=subject,
                    body=body,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[settings.FEEDBACK_EMAIL],
                    reply_to=[reply_to] if reply_to else None,
                )
                email.send(fail_silently=False)

                logger.info(
                    f"Feedback email sent: type={feedback_type}, page={page_url}, "
                    f"user={request.user.username if request.user.is_authenticated else 'anonymous'}"
                )
                messages.success(request, 'Your feedback has been submitted. Thank you!')
                return redirect('contact:feedback_success')

            except Exception as e:
                logger.error(f"Failed to send feedback email: {e}")
                messages.error(request, 'Failed to send feedback. Please try again later.')
    else:
        form = FeedbackForm(initial=initial)

    return render(request, 'contact/feedback.html', {
        'form': form,
        'has_github': has_github,
    })


def contact_success_view(request):
    """Display success page after contact form submission."""
    return render(request, 'contact/success.html', {
        'title': 'Message Sent',
        'message': 'Your message has been sent successfully. We will get back to you if needed.'
    })


def feedback_success_view(request):
    """Display success page after feedback form submission."""
    return render(request, 'contact/success.html', {
        'title': 'Feedback Received',
        'message': 'Thank you for your feedback! We appreciate you helping us improve.'
    })
