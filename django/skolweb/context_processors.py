"""Custom context processors for skolweb."""

from django.conf import settings


def script_name(request):
    """
    Add SCRIPT_NAME to template context for JavaScript API calls.

    This allows the frontend to construct API URLs that work regardless
    of what URL prefix the app is deployed under (e.g., /skol).
    """
    return {
        'SCRIPT_NAME': getattr(settings, 'FORCE_SCRIPT_NAME', '') or '',
    }
