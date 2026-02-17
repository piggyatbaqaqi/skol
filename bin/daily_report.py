#!/usr/bin/env python3
"""
SKOL Daily Admin Report.

Generates a summary of events requiring admin attention:
flagged collections, flagged comments, new users, new collections,
search activity, hidden collections, and system totals.

Runs from cron; outputs to stdout (for MAILTO delivery) and/or
emails opted-in users.

Usage:
    daily_report.py [options]

Options:
    --stdout        Print report to stdout (default if no other output specified)
    --email ADDR    Send report to a specific email address
    --send          Send report to all users with receive_admin_summary=True
    --days N        Lookback window in days (default: 1)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bootstrap Django ORM
DJANGO_DIR = str(Path(__file__).resolve().parent.parent / 'django')
sys.path.insert(0, DJANGO_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skolweb.settings')
import django  # noqa: E402
django.setup()

from django.conf import settings  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402
from django.core.mail import EmailMessage  # noqa: E402
from django.db.models import Count  # noqa: E402

from search.models import Collection, SearchHistory, UserSettings  # noqa: E402


def get_site_url():
    """Get the site URL for generating links."""
    url = os.environ.get('SKOL_SITE_URL', '')
    if url:
        return url.rstrip('/')
    # Fallback: use FORCE_SCRIPT_NAME with placeholder host
    script_name = getattr(settings, 'FORCE_SCRIPT_NAME', '') or ''
    return f'https://synoptickeyof.life{script_name}'


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_flagged_collections(since, site_url):
    """Flagged collections requiring admin review."""
    collections = (
        Collection.objects
        .exclude(flagged_by=[])
        .filter(hidden=False)
        .select_related('owner')
    )
    if not collections.exists():
        return ''

    lines = [
        'ACTION REQUIRED: Flagged Collections',
        '-' * 40,
    ]
    for c in collections:
        flag_count = len(c.flagged_by) if c.flagged_by else 0
        lines.append(f'  - {c.name} (ID: {c.collection_id})')
        lines.append(f'    Owner: {c.owner.username}, Flags: {flag_count}')
        lines.append(f'    {site_url}/?collection={c.collection_id}')
    return '\n'.join(lines)


def section_flagged_comments(since, site_url):
    """Flagged comments requiring admin review (from CouchDB)."""
    try:
        from search.comment_service import get_comments_db, ensure_design_docs
        db = get_comments_db()
        ensure_design_docs(db)

        result = db.view('comments/flagged', include_docs=False)
        # Aggregate by collection_id
        by_collection = {}
        for row in result:
            cid = row.key
            count = row.value
            if cid not in by_collection:
                by_collection[cid] = {'count': 0, 'comments': 0}
            by_collection[cid]['count'] += count
            by_collection[cid]['comments'] += 1

        if not by_collection:
            return ''

        lines = [
            'ACTION REQUIRED: Flagged Comments',
            '-' * 40,
        ]
        for cid, info in sorted(by_collection.items()):
            lines.append(
                f'  - Collection {cid}: '
                f'{info["comments"]} flagged comment(s), '
                f'{info["count"]} total flag(s)'
            )
            lines.append(f'    {site_url}/?collection={cid}')
        return '\n'.join(lines)

    except Exception as e:
        return f'ACTION REQUIRED: Flagged Comments\n  (CouchDB error: {e})'


def section_new_users(since, site_url):
    """Users who signed up in the period."""
    users = User.objects.filter(date_joined__gte=since).order_by('-date_joined')
    if not users.exists():
        return ''

    lines = [
        'New Users',
        '-' * 40,
    ]
    for u in users:
        lines.append(
            f'  - {u.username} ({u.email or "no email"}) '
            f'— {u.date_joined.strftime("%Y-%m-%d %H:%M UTC")}'
        )
    return '\n'.join(lines)


def section_new_collections(since, site_url):
    """Collections created in the period."""
    collections = (
        Collection.objects
        .filter(created_at__gte=since)
        .select_related('owner')
        .order_by('-created_at')
    )
    if not collections.exists():
        return ''

    lines = [
        'New Collections',
        '-' * 40,
    ]
    for c in collections:
        lines.append(f'  - {c.name} (ID: {c.collection_id})')
        lines.append(f'    Owner: {c.owner.username}')
        lines.append(f'    {site_url}/?collection={c.collection_id}')
    return '\n'.join(lines)


def section_search_activity(since, site_url):
    """Aggregate search activity stats for the period."""
    searches = SearchHistory.objects.filter(
        created_at__gte=since,
        event_type='search',
    )
    total = searches.count()
    if total == 0:
        return ''

    unique_collections = searches.values('collection').distinct().count()

    # Most active collection
    top = (
        searches
        .values('collection__name', 'collection__collection_id')
        .annotate(cnt=Count('id'))
        .order_by('-cnt')
        .first()
    )

    lines = [
        'Search Activity',
        '-' * 40,
        f'  Total searches: {total}',
        f'  Active collections: {unique_collections}',
    ]
    if top:
        lines.append(
            f'  Most active: {top["collection__name"]} '
            f'(ID: {top["collection__collection_id"]}, '
            f'{top["cnt"]} searches)'
        )
    return '\n'.join(lines)


def section_hidden_collections(since, site_url):
    """Currently hidden collections."""
    collections = (
        Collection.objects
        .filter(hidden=True)
        .select_related('owner', 'hidden_by')
    )
    if not collections.exists():
        return ''

    lines = [
        'Hidden Collections',
        '-' * 40,
    ]
    for c in collections:
        hidden_by = c.hidden_by.username if c.hidden_by else 'unknown'
        lines.append(f'  - {c.name} (ID: {c.collection_id})')
        lines.append(f'    Owner: {c.owner.username}, Hidden by: {hidden_by}')
        lines.append(f'    {site_url}/?collection={c.collection_id}')
    return '\n'.join(lines)


def section_system_totals(since, site_url):
    """System-wide totals."""
    total_users = User.objects.filter(is_active=True).count()
    total_collections = Collection.objects.count()
    total_searches = SearchHistory.objects.filter(event_type='search').count()

    lines = [
        'System Totals',
        '-' * 40,
        f'  Active users: {total_users}',
        f'  Total collections: {total_collections}',
        f'  All-time searches: {total_searches}',
    ]
    return '\n'.join(lines)


# Ordered list of report section functions
REPORT_SECTIONS = [
    section_flagged_collections,
    section_flagged_comments,
    section_new_users,
    section_new_collections,
    section_search_activity,
    section_hidden_collections,
    section_system_totals,
]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(days=1):
    """Generate the full report as a string."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    site_url = get_site_url()

    header = (
        f'SKOL Daily Admin Report\n'
        f'Generated: {now.strftime("%Y-%m-%d %H:%M UTC")}\n'
        f'Period: last {days} day(s)\n'
        f'Site: {site_url}\n'
        f'{"=" * 60}'
    )

    sections = []
    for section_fn in REPORT_SECTIONS:
        text = section_fn(since, site_url)
        if text:
            sections.append(text)

    footer = (
        f'{"=" * 60}\n'
        f'Admin: {site_url}/admin/'
    )

    body = '\n\n'.join(sections) if sections else '  (no items to report)'
    return f'{header}\n\n{body}\n\n{footer}\n'


def send_report(report_text, recipients):
    """Send the report via Django email to a list of addresses."""
    for addr in recipients:
        try:
            email = EmailMessage(
                subject='[SKOL] Daily Admin Report',
                body=report_text,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[addr],
            )
            email.send(fail_silently=False)
            print(f'Sent report to {addr}', file=sys.stderr)
        except Exception as e:
            print(f'Failed to send to {addr}: {e}', file=sys.stderr)


def get_opted_in_recipients():
    """Get email addresses of users who opted in to receive admin summary."""
    return list(
        UserSettings.objects
        .filter(receive_admin_summary=True)
        .select_related('user')
        .exclude(user__email='')
        .values_list('user__email', flat=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SKOL Daily Admin Report',
    )
    parser.add_argument(
        '--stdout', action='store_true', default=False,
        help='Print report to stdout (default if no other output)',
    )
    parser.add_argument(
        '--email', type=str, default=None,
        help='Send report to a specific email address',
    )
    parser.add_argument(
        '--send', action='store_true', default=False,
        help='Send report to all opted-in users',
    )
    parser.add_argument(
        '--days', type=int, default=1,
        help='Lookback window in days (default: 1)',
    )
    args = parser.parse_args()

    # Default to stdout if no output specified
    use_stdout = args.stdout or (not args.email and not args.send)

    report = generate_report(days=args.days)

    if use_stdout:
        print(report)

    recipients = []
    if args.email:
        recipients.append(args.email)
    if args.send:
        recipients.extend(get_opted_in_recipients())

    if recipients:
        send_report(report, recipients)


if __name__ == '__main__':
    main()
