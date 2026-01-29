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


def format_feedback_issue(feedback_type, message, page_url=None, user=None, email=None):
    """
    Format feedback data into a GitHub issue title and body.

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

    body_parts.extend([
        '',
        '---',
        '*This issue was automatically created from the SKOL feedback form.*'
    ])

    return title, '\n'.join(body_parts), [label, 'user-feedback']
