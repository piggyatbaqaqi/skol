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
sudo certbot --apache -d skol.synoptickeyof.life
```

Or if you have existing certificates, note their paths (typically):
- Certificate: `/etc/letsencrypt/live/skol.synoptickeyof.life/fullchain.pem`
- Private Key: `/etc/letsencrypt/live/skol.synoptickeyof.life/privkey.pem`

## Step 3: Configure Apache Virtual Host

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
sudo a2dissite 000-default  # Optional: disable default site
sudo apache2ctl configtest
sudo systemctl reload apache2
```

## Step 4: Configure Django for HTTPS

Edit `/opt/skol/django/skol-django.env`:

```bash
# Set to True when running behind HTTPS reverse proxy
SKOL_HTTPS=True

# Add your domain to trusted origins
CSRF_TRUSTED_ORIGINS=https://skol.synoptickeyof.life

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Step 5: Restart Services

```bash
sudo systemctl restart skol-django
sudo systemctl restart apache2
```

## Step 6: Verify Configuration

1. Check Django is running:
   ```bash
   sudo systemctl status skol-django
   curl http://127.0.0.1:8000/
   ```

2. Check Apache is proxying correctly:
   ```bash
   curl -I https://skol.synoptickeyof.life/
   ```

3. Verify HTTPS in browser - visit https://skol.synoptickeyof.life/

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

**Static files not loading**
- Django's development server serves static files
- For production, consider using `collectstatic` and serving via Apache

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
