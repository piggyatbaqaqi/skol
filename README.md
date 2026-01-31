# skol
Synoptic Key of Life
by La Monte Henry Piggy Yarroll <piggy.yarroll+skol@gmail.com>

The goal of this project is to find and understand all the species
descriptions in the biological literature, and automatically generate
a synoptic key for all the known species.

I'm starting with the Mycological (fungi) literature, as I am familiar
with it and know where to find a lot of it.

This is a project of the [Western Pennsylvania Mushroom
Club](http://wpamushroomclub.org) as a contribution to the [North
American Mycoflora Project](http://mycoflora.org).

## Installation

### Prerequisites

Before installing the SKOL package, ensure these dependencies are available:

#### System Packages

```bash
# Docker and Docker Compose (for database services)
sudo apt install docker.io docker-compose-v2

# Add your user to the docker group
sudo usermod -aG docker $USER

# Java 17+ (required for PySpark)
sudo apt install openjdk-21-jdk

# Python 3.13 (required)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv
```

#### TLS Certificates

SKOL uses TLS for Redis and CouchDB connections. We recommend using Let's Encrypt with certbot:

```bash
# Install certbot
sudo apt install certbot

# Obtain certificates (replace with your domain)
sudo certbot certonly --standalone -d yourdomain.example.com

# Certificates are stored in /etc/letsencrypt/live/yourdomain.example.com/
```

For detailed TLS setup instructions, see: https://certbot.eff.org/instructions

### Build the Package

Building the Debian package requires additional tools:

```bash
# Install build prerequisites
sudo apt install ruby ruby-dev build-essential python3-venv python3-pip python3-build
sudo gem install fpm

# Build the package
./build-deb.sh
```

### Install the Package

```bash
# Install the package
sudo dpkg -i deb_dist/skol_*.deb

# Or on a different machine, copy and install
scp deb_dist/skol_*.deb user@server:/tmp/
ssh user@server "sudo dpkg -i /tmp/skol_*.deb"
```

The package installs to:
- `/opt/skol/` - Application files, virtual environment, and static configs
- `/data/skol/` - Runtime database data (created by postinst)
- `/etc/cron.d/skol` - Cron jobs

### Post-Installation Setup

#### 1. Create the Credential File

Create `/home/skol/.skol_env` with your credentials. This file is sourced by all SKOL scripts and the Docker containers. A template is provided in `skol_env.example`.

```bash
# Option 1: Copy and edit the template (from installed package)
sudo cp /opt/skol/skol_env.example /home/skol/.skol_env
sudo chown skol:skol /home/skol/.skol_env
sudo chmod 600 /home/skol/.skol_env
sudo -u skol nano /home/skol/.skol_env  # Edit with your values

# Option 2: Create manually
sudo -u skol tee /home/skol/.skol_env << 'EOF'
# CouchDB Configuration
COUCHDB_USER=admin
COUCHDB_PASSWORD=your_secure_password_here
COUCHDB_URL=http://localhost:5984

# Database Names
INGEST_DATABASE=skol_dev
TAXON_DATABASE=taxa

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_password_here
REDIS_TLS=true

# Email Configuration (for notifications)
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@example.com
EMAIL_HOST_PASSWORD=your_email_password
EMAIL_USE_TLS=true
DEFAULT_FROM_EMAIL=skol@example.com
MAILTO=admin@example.com

# Logging
LOGDIR=/var/log/skol
VERBOSITY=1

# Model expiration (empty string = no expiration)
CLASSIFIER_MODEL_EXPIRE=
EOF

# Secure the file
sudo chmod 600 /home/skol/.skol_env
sudo chown skol:skol /home/skol/.skol_env
```

#### 2. Update TLS Certificate Paths

Edit `/opt/skol/advanced-databases/docker-compose.yaml` and update the certificate paths if your domain differs from the default:

```yaml
# In redis and couchdb service volumes, update:
- /etc/letsencrypt/live/yourdomain.example.com/...
```

Also update `/opt/skol/advanced-databases/redis.conf` with your certificate paths.

#### 3. Start Database Services

```bash
cd /opt/skol/advanced-databases
docker compose up -d
```

Verify services are running:
```bash
docker compose ps
```

#### 4. Initialize CouchDB Admin

On first start, CouchDB reads credentials from the environment. Verify the admin was created:

```bash
curl -u admin:your_password http://localhost:5984/_session
```

If you need to reset the password, delete `/data/skol/couchdb/etc/local.d/docker.ini` and restart the container.

#### 5. Create CouchDB Databases

```bash
# Using the with_skol wrapper ensures proper environment
/opt/skol/bin/with_skol python -c "
import couchdb
server = couchdb.Server('http://admin:your_password@localhost:5984')
for db in ['skol_dev', 'taxa']:
    if db not in server:
        server.create(db)
        print(f'Created database: {db}')
"
```

#### 6. Populate Redis Keys

After databases are running, populate the Redis cache:

```bash
# Rebuild all Redis keys (this may take a while for classifier training)
/opt/skol/bin/rebuild_redis

# Or skip the slow classifier training
/opt/skol/bin/rebuild_redis --skip-classifier

# List existing keys
/opt/skol/bin/rebuild_redis --list
```

### Directory Structure

```
/opt/skol/
├── bin/                    # Command scripts and wrappers
├── venv/                   # Python virtual environment
├── wheels/                 # Python wheel packages
├── data/ontologies/        # Ontology files (.obo)
├── models/                 # ML model files
├── advanced-databases/     # Docker Compose and database configs
│   ├── docker-compose.yaml
│   ├── redis.conf
│   ├── redis-entrypoint.sh
│   └── neo4j/conf/
└── .cargo/, .rustup/       # Rust toolchain (for outlines package)

/data/skol/                 # Runtime database data
├── couchdb/
│   ├── data/               # CouchDB data files
│   └── etc/                # CouchDB config (including credentials)
├── redis/data/             # Redis persistence (RDB/AOF)
└── neo4j/data/             # Neo4j graph data

/home/skol/.skol_env        # Master credential file (chmod 600)
/var/log/skol/              # Application logs
/etc/cron.d/skol            # Scheduled tasks
```

### Available Commands

All commands are available in `/opt/skol/bin/`:

| Command | Description |
|---------|-------------|
| `ingest` | Ingest documents into CouchDB |
| `train_classifier` | Train the text classifier model |
| `predict_classifier` | Run predictions with trained model |
| `extract_taxa_to_couchdb` | Extract taxonomic data to CouchDB |
| `embed_taxa` | Generate embeddings for taxa |
| `taxa_to_json` | Export taxa to JSON format |
| `build_vocab_tree` | Build vocabulary tree for UI menus |
| `manage_fungaria` | Manage Index Herbariorum data |
| `watch_install` | Watch for new documents to process |
| `rebuild_redis` | Rebuild all Redis keys |

Use the wrapper for custom commands:
```bash
/opt/skol/bin/with_skol python your_script.py
```

### Troubleshooting

#### CouchDB Authentication Errors

If you see "Unauthorized" errors:
1. Check that `/home/skol/.skol_env` has correct `COUCHDB_PASSWORD`
2. Verify the password matches what's in `/data/skol/couchdb/etc/local.d/docker.ini`
3. To reset: delete `docker.ini`, restart container (will re-read from env)

#### Redis Connection Errors

1. Verify Redis is running: `docker compose ps`
2. Check TLS settings match between `.skol_env` and `redis.conf`
3. Test connection: `redis-cli -h localhost -p 6380 --tls --cacert /etc/ssl/certs/ca-certificates.crt PING`

#### Permission Errors

Database directories have specific ownership requirements:
```bash
# CouchDB runs as uid 5984
sudo chown -R 5984:5984 /data/skol/couchdb

# Redis runs as uid 999
sudo chown -R 999:999 /data/skol/redis

# Neo4j runs as uid 7474
sudo chown -R 7474:7474 /data/skol/neo4j
```

### Development Setup

For development without installing the package:

```bash
# Clone the repository
git clone https://github.com/piggyatbaqaqi/skol.git
cd skol

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Set up environment (copy from production or create new)
cp /path/to/.skol_env ~/.skol_env
source ~/.skol_env

# Run tests
make test
```
