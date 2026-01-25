# Database Configuration

SKOL Django supports two database backends:
- **SQLite** (default) - Zero configuration, suitable for development and small deployments
- **PostgreSQL** - Recommended for production with better concurrency and performance

## SQLite (Default)

SQLite is used by default with no configuration required. The database file is stored at:
- Development: `<project_root>/db.sqlite3`
- Production (deb package): `/opt/skol/django/db.sqlite3`

### Advantages
- No additional services to run
- Simple backup (just copy the file)
- Good for single-user or low-traffic deployments

### Limitations
- Limited concurrent write access (file locking)
- Not suitable for high-traffic production use

## PostgreSQL

PostgreSQL provides better performance and concurrency for production deployments.

### Prerequisites

1. **Install PostgreSQL** (if not using Docker):
   ```bash
   sudo apt install postgresql postgresql-contrib
   ```

2. **The psycopg2-binary package** is included in requirements.txt and will be installed automatically.

### Option 1: Docker PostgreSQL

The simplest way to run PostgreSQL for SKOL:

```bash
docker run -d \
  --name skol-postgres \
  -e POSTGRES_DB=skol \
  -e POSTGRES_USER=skol \
  -e POSTGRES_PASSWORD=your_secure_password \
  -p 5432:5432 \
  -v skol_postgres_data:/var/lib/postgresql/data \
  postgres:16
```

### Option 2: Native PostgreSQL

Create a database and user:

```bash
sudo -u postgres psql

CREATE USER skol WITH PASSWORD 'your_secure_password';
CREATE DATABASE skol OWNER skol;
GRANT ALL PRIVILEGES ON DATABASE skol TO skol;
\q
```

### Configuration

Edit `/opt/skol/django/skol-django.env` and uncomment/configure the PostgreSQL settings:

```bash
# PostgreSQL configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=skol
POSTGRES_USER=skol
POSTGRES_PASSWORD=your_secure_password
```

### Apply Migrations

After configuring PostgreSQL, run migrations:

```bash
# If installed via deb package
/opt/skol/bin/with_skol_django python manage.py migrate

# Or reinstall the package (migrations run automatically in postinst)
sudo dpkg -i --force-all ./skol-django_*.deb
```

### Restart the Service

```bash
sudo systemctl restart skol-django
```

## Switching Between Databases

### From SQLite to PostgreSQL

1. **Export data from SQLite** (optional, if you have existing data):
   ```bash
   /opt/skol/bin/with_skol_django python manage.py dumpdata \
     --natural-foreign --natural-primary \
     --exclude=contenttypes --exclude=auth.permission \
     > /tmp/skol_data.json
   ```

2. **Configure PostgreSQL** in `/opt/skol/django/skol-django.env`

3. **Run migrations** on the new database:
   ```bash
   /opt/skol/bin/with_skol_django python manage.py migrate
   ```

4. **Import data** (if you exported it):
   ```bash
   /opt/skol/bin/with_skol_django python manage.py loaddata /tmp/skol_data.json
   ```

5. **Restart the service**:
   ```bash
   sudo systemctl restart skol-django
   ```

### From PostgreSQL to SQLite

1. Export data using `dumpdata` as shown above
2. Comment out or remove the `POSTGRES_*` variables from the env file
3. Run migrations: `python manage.py migrate`
4. Import data using `loaddata`
5. Restart the service

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | (empty) | PostgreSQL host. If empty, SQLite is used |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
| `POSTGRES_DB` | skol | Database name |
| `POSTGRES_USER` | skol | Database user |
| `POSTGRES_PASSWORD` | (empty) | Database password |

## Database Comparison

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Setup complexity | None | Requires server |
| Concurrent writes | Limited | Excellent |
| Performance at scale | Moderate | Better |
| JSON field support | Basic | Native with indexing |
| Backup method | Copy file | pg_dump |
| Best for | Development, small deployments | Production |

## Troubleshooting

### Connection refused
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify the port is correct
- Check pg_hba.conf allows connections from localhost

### Authentication failed
- Verify username and password
- Check the user has access to the database

### Migrations fail
- Ensure the database exists
- Ensure the user has CREATE TABLE permissions
- Check for any pending migrations: `python manage.py showmigrations`

### Reset database (development only)
```bash
# SQLite - just delete the file
rm /opt/skol/django/db.sqlite3
/opt/skol/bin/with_skol_django python manage.py migrate

# PostgreSQL - drop and recreate
sudo -u postgres psql -c "DROP DATABASE skol;"
sudo -u postgres psql -c "CREATE DATABASE skol OWNER skol;"
/opt/skol/bin/with_skol_django python manage.py migrate
```
