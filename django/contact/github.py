"""GitHub integration for feedback submission."""
import logging
import requests

logger = logging.getLogger(__name__)

# Target repository for feedback issues
GITHUB_REPO = 'piggyatbaqaqi/skol'
GITHUB_API_BASE = 'https://api.github.com'


def get_github_token(user):
    """
    Get the GitHub OAuth access token for a user.

    Returns the token string if the user has a linked GitHub account,
    or None if not linked or token unavailable.
    """
    if not user.is_authenticated:
        return None

    try:
        from allauth.socialaccount.models import SocialAccount, SocialToken

        # Find user's GitHub social account
        github_account = SocialAccount.objects.filter(
            user=user,
            provider='github'
        ).first()

        if not github_account:
            return None

        # Get the token for this account
        token = SocialToken.objects.filter(account=github_account).first()

        if token:
            return token.token

        return None

    except ImportError:
        logger.warning("allauth not installed, cannot get GitHub token")
        return None
    except Exception as e:
        logger.error(f"Error getting GitHub token: {e}")
        return None


def create_github_issue(token, title, body, labels=None):
    """
    Create a GitHub issue in the SKOL repository.

    Args:
        token: GitHub OAuth access token
        title: Issue title
        body: Issue body (markdown supported)
        labels: Optional list of label names

    Returns:
        dict with 'success' bool and either 'issue_url' or 'error' message
    """
    url = f'{GITHUB_API_BASE}/repos/{GITHUB_REPO}/issues'

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }

    data = {
        'title': title,
        'body': body,
    }

    if labels:
        data['labels'] = labels

    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)

        if response.status_code == 201:
            issue_data = response.json()
            logger.info(f"Created GitHub issue #{issue_data['number']}: {title}")
            return {
                'success': True,
                'issue_url': issue_data['html_url'],
                'issue_number': issue_data['number'],
            }
        else:
            error_msg = response.json().get('message', 'Unknown error')
            logger.error(f"Failed to create GitHub issue: {response.status_code} - {error_msg}")
            return {
                'success': False,
                'error': f"GitHub API error: {error_msg}",
            }

    except requests.RequestException as e:
        logger.error(f"Request error creating GitHub issue: {e}")
        return {
            'success': False,
            'error': str(e),
        }


def format_feedback_issue(feedback_type, message, page_url=None, user=None,
                          email=None, browser_info=None):
    """
    Format feedback data into a GitHub issue title and body.

    Args:
        feedback_type: Type of feedback (bug, feature, usability, other)
        message: The feedback message
        page_url: URL of the page where feedback originated
        user: Django user object (optional)
        email: Reply-to email address (optional)
        browser_info: Dict of browser/device environment info (optional)

    Returns:
        tuple of (title, body, labels)
    """
    # Map feedback type to title prefix and label
    type_config = {
        'bug': ('Bug Report', 'bug'),
        'feature': ('Feature Request', 'enhancement'),
        'usability': ('Usability Issue', 'usability'),
        'other': ('Feedback', 'feedback'),
    }

    prefix, label = type_config.get(feedback_type, ('Feedback', 'feedback'))
    title = f'[{prefix}] {message[:60]}{"..." if len(message) > 60 else ""}'

    # Build issue body
    body_parts = ['## Description', '', message, '']

    if page_url:
        body_parts.extend(['## Page URL', '', page_url, ''])

    body_parts.extend(['## Submitted by', ''])
    if user and user.is_authenticated:
        body_parts.append(f'- **User:** {user.username}')
    if email:
        body_parts.append(f'- **Email:** {email}')

    # Add browser environment section
    if browser_info:
        body_parts.extend(['', '## Environment', ''])

        # OS and browser
        if browser_info.get('browser'):
            body_parts.append(f"- **Browser:** {browser_info.get('browser')}")
        if browser_info.get('os'):
            os_str = browser_info.get('os')
            if browser_info.get('osVersion'):
                os_str += f" {browser_info.get('osVersion')}"
            body_parts.append(f"- **OS:** {os_str}")

        # Device type
        is_mobile = browser_info.get('isMobile', False)
        body_parts.append(f"- **Device:** {'Mobile' if is_mobile else 'Desktop'}")

        # Screen and viewport
        if browser_info.get('screenWidth') and browser_info.get('screenHeight'):
            body_parts.append(
                f"- **Screen:** {browser_info.get('screenWidth')}x"
                f"{browser_info.get('screenHeight')}"
            )
        if browser_info.get('viewportWidth') and browser_info.get('viewportHeight'):
            body_parts.append(
                f"- **Viewport:** {browser_info.get('viewportWidth')}x"
                f"{browser_info.get('viewportHeight')}"
            )

        # Touch support
        if browser_info.get('touchSupport'):
            touch_points = browser_info.get('maxTouchPoints', 0)
            body_parts.append(f"- **Touch:** Yes ({touch_points} touch points)")

        # Connection type (useful for mobile debugging)
        if browser_info.get('connectionType'):
            body_parts.append(f"- **Connection:** {browser_info.get('connectionType')}")

        # Pixel ratio (important for retina/mobile displays)
        if browser_info.get('devicePixelRatio'):
            body_parts.append(
                f"- **Pixel Ratio:** {browser_info.get('devicePixelRatio')}"
            )

        # Full user agent in collapsible section
        if browser_info.get('userAgent'):
            body_parts.extend([
                '',
                '<details>',
                '<summary>Full User Agent</summary>',
                '',
                f"`{browser_info.get('userAgent')}`",
                '',
                '</details>',
            ])

    body_parts.extend([
        '',
        '---',
        '*This issue was automatically created from the SKOL feedback form.*'
    ])

    return title, '\n'.join(body_parts), [label, 'user-feedback']
