#!/usr/bin/env python3
"""
Fix CouchDB skol_training database permissions.

This script removes security restrictions from the skol_training database
to make it accessible with the same credentials as skol_dev.
"""

import couchdb
import os
import sys


def fix_database_permissions(
    couchdb_url: str,
    database_name: str,
    username: str,
    password: str
):
    """
    Fix database permissions by clearing security restrictions.

    Args:
        couchdb_url: CouchDB server URL
        database_name: Database name to fix
        username: Admin username
        password: Admin password
    """
    print(f"Connecting to {couchdb_url}...")

    # Build authenticated URL
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(couchdb_url)
    netloc = f"{username}:{password}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"

    auth_url = urlunparse((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment
    ))

    # Connect to CouchDB
    try:
        couch = couchdb.Server(auth_url)
        print(f"✓ Connected to CouchDB")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    # Check if database exists
    if database_name not in couch:
        print(f"✗ Database '{database_name}' does not exist")
        return False

    print(f"✓ Database '{database_name}' found")

    # Get database object
    db = couch[database_name]

    # Check current security settings
    try:
        import requests
        security_url = f"{auth_url}/{database_name}/_security"
        response = requests.get(security_url)

        print(f"\nCurrent security settings:")
        print(response.text)

        # Clear security settings (make it accessible to all authenticated users)
        new_security = {
            "admins": {
                "names": [],
                "roles": []
            },
            "members": {
                "names": [],
                "roles": []
            }
        }

        print(f"\nApplying new security settings:")
        print(f"  Admins: {new_security['admins']}")
        print(f"  Members: {new_security['members']}")

        response = requests.put(security_url, json=new_security)

        if response.status_code == 200:
            print(f"✓ Security settings updated successfully")

            # Verify the change
            response = requests.get(security_url)
            print(f"\nNew security settings:")
            print(response.text)

            return True
        else:
            print(f"✗ Failed to update security: {response.status_code} {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error updating security: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix CouchDB skol_training database permissions'
    )
    parser.add_argument(
        '--database',
        default='skol_training',
        help='Database name (default: skol_training)'
    )
    parser.add_argument(
        '--couchdb-url',
        default=os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        help='CouchDB server URL (default: from COUCHDB_URL env var)'
    )
    parser.add_argument(
        '--username',
        default=os.environ.get('COUCHDB_USER', 'admin'),
        help='CouchDB admin username (default: from COUCHDB_USER env var)'
    )
    parser.add_argument(
        '--password',
        default=os.environ.get('COUCHDB_PASSWORD', ''),
        help='CouchDB admin password (default: from COUCHDB_PASSWORD env var)'
    )

    args = parser.parse_args()

    if not args.password:
        print("Error: Password required. Set COUCHDB_PASSWORD environment variable or use --password")
        sys.exit(1)

    print("=" * 70)
    print("CouchDB Database Permission Fix")
    print("=" * 70)
    print(f"Database: {args.database}")
    print(f"URL: {args.couchdb_url}")
    print(f"Username: {args.username}")
    print("=" * 70)
    print()

    success = fix_database_permissions(
        args.couchdb_url,
        args.database,
        args.username,
        args.password
    )

    if success:
        print("\n" + "=" * 70)
        print("✅ Database permissions fixed successfully!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Failed to fix database permissions")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
