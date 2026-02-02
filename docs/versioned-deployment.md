# Versioned Deployment

SKOL packages use version-specific installation directories with atomic symlink switching for minimal downtime deployments. Each package version installs to its own directory, and switching versions is instantaneous via symlink updates.

## Overview

The versioned deployment system provides:

- **Zero-downtime upgrades**: Symlink switch is atomic
- **Instant rollback**: Switch to any previously installed version
- **Multiple versions coexist**: Old versions remain until explicitly removed
- **Automatic migration**: First install of versioned layout migrates old directories

## Directory Structure

After installation, the directory structure looks like:

```
/opt/skol/
├── versions/                           # skol package versions
│   ├── 0.1.0-5/
│   │   ├── bin/                        # Python scripts
│   │   └── skol_env.example
│   └── 0.1.0-6/
│       └── ...
├── bin -> versions/0.1.0-6/bin         # Symlink to active version
├── advanced-databases/                 # Docker configs (NOT versioned, persists)
├── config/                             # Shared config (persists across upgrades)
│   └── skol-django.env                 # Django environment config
├── wheels/                             # Shared wheel files
├── venv/                               # Shared Python virtual environment
├── data/                               # Shared data directory
│   ├── ontologies/
│   └── django/
│       └── db.sqlite3                  # Django database (persists across upgrades)
├── models/                             # Shared ML models
│
├── django-versions/                    # skol-django versions
│   ├── 0.1.0-3/
│   │   ├── manage.py
│   │   ├── skolweb/
│   │   ├── search/
│   │   ├── templates/
│   │   └── static/
│   └── 0.1.0-4/
│       └── ...
├── django -> django-versions/0.1.0-4   # Symlink to active version
├── django-venv/                        # Django virtual environment
└── staticfiles/                        # Collected static files

/opt/dr-drafts/
├── versions/                           # dr-drafts-mycosearch versions
│   ├── 0.3.0-2/
│   │   └── mycosearch/
│   └── 0.3.0-3/
│       └── mycosearch/
└── mycosearch -> versions/0.3.0-3/mycosearch  # Symlink to active version
```

## Version Switching

### Using skol-switch-version

The `skol-switch-version` command allows instant switching between installed versions:

```bash
# Switch skol to a specific version
skol-switch-version skol 0.1.0-5

# Switch django to a specific version
skol-switch-version django 0.1.0-3

# Switch dr-drafts to a specific version
skol-switch-version dr-drafts 0.3.0-2

# List available versions
skol-switch-version skol --list
skol-switch-version django --list
skol-switch-version dr-drafts --list

# Show current version
skol-switch-version skol
```

### Manual Symlink Switching

You can also switch versions manually:

```bash
# Switch skol (only bin is versioned; advanced-databases persists)
ln -sfn /opt/skol/versions/0.1.0-5/bin /opt/skol/bin

# Switch django (and restart service)
ln -sfn /opt/skol/django-versions/0.1.0-3 /opt/skol/django
systemctl restart skol-django

# Switch dr-drafts
ln -sfn /opt/dr-drafts/versions/0.3.0-2/mycosearch /opt/dr-drafts/mycosearch
```

## Building Packages

Each package's `build-deb.sh` script:

1. Increments the build number (stored in `.build-number`)
2. Creates version-specific staging directories
3. Injects the version into postinst/prerm templates
4. Builds the Debian package

```bash
# Build skol package
cd /path/to/skol
./build-deb.sh
# Creates: deb_dist/skol_0.1.0-N_all.deb

# Build skol-django package
cd /path/to/skol/django
./build-deb.sh
# Creates: deb_dist/skol-django_0.1.0-N_all.deb

# Build dr-drafts-mycosearch package
cd /path/to/dr-drafts-mycosearch
./build-deb.sh
# Creates: deb_dist/dr-drafts-mycosearch_0.3.0-N_all.deb
```

## Installation Behavior

### First Install (Migration)

When installing on a system with the old non-versioned layout:

1. Detects existing directories (e.g., `/opt/skol/bin` as a directory, not symlink)
2. Moves them to timestamped backups (e.g., `/opt/skol/bin.old-20240115120000`)
3. Installs to version-specific directory
4. Creates symlinks to the new version

### Upgrade

When installing a new version:

1. Installs files to a new version-specific directory (dpkg extracts the package)
2. Runs all setup tasks against the NEW version directory:
   - Creates/updates Python virtual environment
   - Installs packages from wheels
   - Runs database migrations
   - Collects static files
3. **Atomic switch**: Only after setup is complete, updates symlink to the new version
4. Restarts the service (for skol-django)
5. Previous version directory remains (for potential rollback)

The key insight is that the **old version continues serving requests** while the new version is being prepared. Downtime is minimized to just the service restart time.

### Removal

When removing a package version:

1. Removes the version-specific directory
2. If this was the active version, activates the most recent remaining version
3. If no versions remain, removes shared resources (venv, wrappers)

## Cleanup

Old versions accumulate and should be cleaned periodically:

```bash
# List all installed skol versions
ls /opt/skol/versions/

# Remove old versions (keep current and one previous)
# First, check which is current:
readlink /opt/skol/bin

# Then remove old ones:
rm -rf /opt/skol/versions/0.1.0-3
rm -rf /opt/skol/versions/0.1.0-4
# ... keep 0.1.0-5 (previous) and 0.1.0-6 (current)

# Same for django
ls /opt/skol/django-versions/
rm -rf /opt/skol/django-versions/0.1.0-1
rm -rf /opt/skol/django-versions/0.1.0-2

# Same for dr-drafts
ls /opt/dr-drafts/versions/
rm -rf /opt/dr-drafts/versions/0.3.0-1
```

## Rollback Procedure

If a new version has issues:

```bash
# 1. Check available versions
skol-switch-version skol --list

# 2. Switch to previous version
skol-switch-version skol 0.1.0-5

# 3. For Django, also restart the service
skol-switch-version django 0.1.0-3
systemctl restart skol-django

# 4. Verify services are working
curl -s http://localhost/skol/ | head -5
```

## Technical Details

### Template Files

Each package has template files for postinst and prerm scripts:

- `debian/postinst.template` - Contains `__VERSION__` placeholder
- `debian/prerm.template` - Contains `__VERSION__` placeholder

During build, `sed` replaces `__VERSION__` with the actual version string.

### Symlink Switching

Symlinks are updated using `ln -sfn` which:

- `-s`: Creates a symbolic link
- `-f`: Removes existing destination file
- `-n`: Treats destination as a normal file (doesn't follow if it's a symlink)

This combination ensures atomic switching even when the target is already a symlink.

### Shared vs. Versioned Resources

| Resource | Location | Versioned? |
|----------|----------|------------|
| Python scripts | `/opt/skol/versions/X/bin/` | Yes |
| Django app | `/opt/skol/django-versions/X/` | Yes |
| Docker configs | `/opt/skol/advanced-databases/` | No (shared) |
| Config files | `/opt/skol/config/` | No (shared) |
| Django database | `/opt/skol/data/django/db.sqlite3` | No (shared) |
| Wheels | `/opt/skol/wheels/` | No (shared) |
| Virtual envs | `/opt/skol/venv/`, `/opt/skol/django-venv/` | No (shared) |
| Data files | `/opt/skol/data/` | No (shared) |
| Database runtime | `/data/skol/{couchdb,redis,neo4j}/` | No (persistent) |
| Models | `/opt/skol/models/` | No (shared) |
| Ontologies | `/usr/share/skol/ontologies/` | No (system) |
| Cron jobs | `/etc/cron.d/skol` | No (system) |

### Shared Configuration

Configuration files are stored in `/opt/skol/config/` to persist across version upgrades. This directory is created during installation and user customizations are preserved.

**skol-django.env** - Django environment configuration:
```
/opt/skol/config/skol-django.env
```

The postinst script will:
1. Create the config directory if it doesn't exist
2. Migrate config from old version-specific locations (if present)
3. Create a default config only if no config file exists
4. Preserve existing user customizations on upgrade

Default settings include `FORCE_SCRIPT_NAME=/skol` for the common deployment pattern at a subpath.

To customize Django settings:
```bash
sudo -u skol nano /opt/skol/config/skol-django.env
systemctl restart skol-django
```

### Django Database

The Django SQLite database is stored in a shared location to persist across version upgrades:
```
/opt/skol/data/django/db.sqlite3
```

The postinst script will:
1. Create the data directory if it doesn't exist
2. Migrate the database from old version-specific locations (if present)
3. Preserve existing database on upgrade

This ensures Django users, sessions, and other database content survive version upgrades.

To use PostgreSQL instead (recommended for production), set `POSTGRES_HOST` in `/opt/skol/config/skol-django.env`.

### Docker Configuration (advanced-databases)

The `advanced-databases` directory is NOT versioned. It persists at `/opt/skol/advanced-databases/` across upgrades, preserving your Docker configuration customizations.

**Structure:**
```
/opt/skol/advanced-databases/
├── docker-compose.yaml      # Main Docker Compose file
├── redis.conf               # Redis configuration
├── redis-entrypoint.sh      # Redis startup script
├── neo4j/
│   ├── conf/                # Neo4j configuration
│   └── certificates/        # Neo4j TLS certificates
└── couchdb/
    └── etc/                 # CouchDB configuration templates
```

**Runtime data** is stored separately in `/data/skol/`:
```
/data/skol/
├── couchdb/
│   ├── data/               # CouchDB databases
│   └── etc/                # CouchDB runtime config (including credentials)
├── redis/
│   └── data/               # Redis persistence
└── neo4j/
    └── data/               # Neo4j database
```

On upgrade, new files from the package are copied to `advanced-databases/` only if they don't already exist (using `cp -n`). This ensures your customizations are preserved while new files are added.

To manually update a file from a new package version:
```bash
# Check what's in the new version
ls /opt/skol/versions/<version>/advanced-databases/

# Copy specific file if needed
cp /opt/skol/versions/<version>/advanced-databases/docker-compose.yaml /opt/skol/advanced-databases/
```

### Service Considerations

For `skol-django`, remember to restart the service after version switching:

```bash
systemctl restart skol-django
```

The prerm script automatically stops the service during removal and re-enables it if a fallback version is activated.
