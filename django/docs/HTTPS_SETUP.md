# Setting Up HTTPS for SKOL Django

This guide explains how to configure SKOL Django to run over HTTPS using Apache as a reverse proxy.

## Overview

The recommended setup is:
- Apache handles HTTPS termination and serves as a reverse proxy
- Django runs on localhost:8000 (not exposed to the internet)
- Apache forwards requests to Django and handles SSL certificates

## Prerequisites

- SKOL Django deb package installed
- Apache2 installed (`sudo apt install apache2`)
- SSL certificate (from Let's Encrypt or other CA)
- Required Apache modules

## Step 1: Enable Apache Modules

```bash
sudo a2enmod proxy proxy_http ssl headers rewrite
sudo systemctl restart apache2
```

## Step 2: Obtain SSL Certificate

Using Let's Encrypt (recommended):

```bash
sudo apt install certbot python3-certbot-apache
sudo certbot --apache -d synoptickeyof.life
```

Or if you have existing certificates, note their paths (typically):
- Certificate: `/etc/letsencrypt/live/synoptickeyof.life/fullchain.pem`
- Private Key: `/etc/letsencrypt/live/synoptickeyof.life/privkey.pem`

## Deployment Options

You can deploy SKOL Django in two ways:

### Option A: At a Subpath (e.g., https://synoptickeyof.life/skol/)

This is useful when you have an existing website and want to add SKOL Django at a specific path.

### Option B: At a Subdomain (e.g., https://skol.synoptickeyof.life/)

This gives SKOL Django its own subdomain.

---

## Option A: Subpath Configuration

### Apache Configuration

Add to your existing site's VirtualHost in `/etc/apache2/sites-available/your-site-ssl.conf`:

```apache
<VirtualHost *:443>
    ServerName synoptickeyof.life

    # ... your existing SSL and site configuration ...

    # SKOL Django at /skol path
    <Location /skol>
        ProxyPass http://127.0.0.1:8000
        ProxyPassReverse http://127.0.0.1:8000
        RequestHeader set X-Forwarded-Proto "https"
        RequestHeader set X-Forwarded-For "%{REMOTE_ADDR}s"
        RequestHeader set X-Script-Name "/skol"
    </Location>

    # ... rest of your configuration ...
</VirtualHost>
```

### Django Configuration

Edit `/opt/skol/django/skol-django.env`:

```bash
# Set to True when running behind HTTPS reverse proxy
SKOL_HTTPS=True

# Add your domain to trusted origins
CSRF_TRUSTED_ORIGINS=https://synoptickeyof.life

# URL path prefix - must match Apache's Location path
FORCE_SCRIPT_NAME=/skol

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Restart Services

```bash
sudo systemctl restart skol-django
sudo systemctl reload apache2
```

### Verify

Visit https://synoptickeyof.life/skol/

---

## Option B: Subdomain Configuration

### Apache Configuration

Create `/etc/apache2/sites-available/skol-django-ssl.conf`:

```apache
<VirtualHost *:80>
    ServerName skol.synoptickeyof.life

    # Redirect all HTTP to HTTPS
    RewriteEngine On
    RewriteCond %{HTTPS} off
    RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
</VirtualHost>

<VirtualHost *:443>
    ServerName skol.synoptickeyof.life

    # SSL Configuration
    SSLEngine on
    SSLCertificateFile /etc/letsencrypt/live/skol.synoptickeyof.life/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/skol.synoptickeyof.life/privkey.pem

    # Modern SSL configuration
    SSLProtocol all -SSLv3 -TLSv1 -TLSv1.1
    SSLCipherSuite ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384
    SSLHonorCipherOrder off

    # Proxy to Django
    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/

    # Pass headers so Django knows about HTTPS
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Forwarded-For "%{REMOTE_ADDR}s"

    # Security headers
    Header always set Strict-Transport-Security "max-age=31536000; includeSubDomains"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-Frame-Options "SAMEORIGIN"

    # Logging
    ErrorLog ${APACHE_LOG_DIR}/skol-django-error.log
    CustomLog ${APACHE_LOG_DIR}/skol-django-access.log combined
</VirtualHost>
```

Enable the site:

```bash
sudo a2ensite skol-django-ssl
sudo apache2ctl configtest
sudo systemctl reload apache2
```

### Django Configuration

Edit `/opt/skol/django/skol-django.env`:

```bash
# Set to True when running behind HTTPS reverse proxy
SKOL_HTTPS=True

# Add your domain to trusted origins
CSRF_TRUSTED_ORIGINS=https://skol.synoptickeyof.life

# Leave empty for subdomain deployment (no path prefix)
FORCE_SCRIPT_NAME=

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Restart Services

```bash
sudo systemctl restart skol-django
sudo systemctl restart apache2
```

### Verify

Visit https://skol.synoptickeyof.life/

---

## Configuration File Location

The Django environment configuration file is at:

```
/opt/skol/django/skol-django.env
```

This file is read by the systemd service and sets environment variables for Django.

## Troubleshooting

### Check logs

```bash
# Django logs
sudo tail -f /var/log/skol/skol-django.log

# Apache logs
sudo tail -f /var/log/apache2/skol-django-error.log
sudo tail -f /var/log/apache2/skol-django-access.log
```

### Common issues

**502 Bad Gateway**
- Django not running: `sudo systemctl start skol-django`
- Check Django is listening: `ss -tlnp | grep 8000`

**CSRF verification failed**
- Ensure `CSRF_TRUSTED_ORIGINS` includes your domain with `https://` prefix
- Ensure `SKOL_HTTPS=True` is set
- Restart Django after changing environment: `sudo systemctl restart skol-django`

**404 on subpath deployment**
- Ensure `FORCE_SCRIPT_NAME` matches the Apache `<Location>` path exactly
- Restart Django after changes: `sudo systemctl restart skol-django`

**Static files not loading**
- Check `FORCE_SCRIPT_NAME` is set correctly (static URLs include the prefix)
- Django's development server serves static files automatically

### Certificate renewal (Let's Encrypt)

Certbot typically sets up automatic renewal. Test with:

```bash
sudo certbot renew --dry-run
```

## Production Considerations

For production deployments, also consider:

1. **Use Gunicorn instead of Django's dev server**:
   - Install in the venv: `/opt/skol/django-venv/bin/pip install gunicorn`
   - Change service ExecStart to use gunicorn

2. **Set DEBUG=False** in Django settings

3. **Use a proper SECRET_KEY** - set via environment variable

4. **Configure ALLOWED_HOSTS** to only allow your domain

5. **Serve static files via Apache** using `collectstatic`
