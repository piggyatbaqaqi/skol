"""
Management command to create the comments CouchDB database and design docs.

Usage:
    python manage.py setup_comment_views
    python manage.py setup_comment_views --warm
"""
from django.core.management.base import BaseCommand
from search.comment_service import get_comments_db, ensure_design_docs


class Command(BaseCommand):
    help = 'Create comments CouchDB database and push design documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--warm',
            action='store_true',
            help='Warm the view indexes after creating (runs a test query)',
        )

    def handle(self, *args, **options):
        self.stdout.write('Setting up comments database...')

        db = get_comments_db()
        self.stdout.write(
            self.style.SUCCESS(f'Database ready: {db.name}')
        )

        self.stdout.write('Pushing design documents...')
        ensure_design_docs(db)
        self.stdout.write(
            self.style.SUCCESS('Design documents up to date')
        )

        if options['warm']:
            self.stdout.write('Warming view indexes...')
            # Trigger a query to build each view index
            list(db.view(
                'comments/by_collection_path', limit=0
            ))
            list(db.view(
                'comments/count_by_collection', limit=0
            ))
            list(db.view(
                'comments/children_count', limit=0
            ))
            self.stdout.write(
                self.style.SUCCESS('View indexes warmed')
            )

        self.stdout.write(self.style.SUCCESS('Done.'))
