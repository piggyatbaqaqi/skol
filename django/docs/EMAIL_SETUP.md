# Email Configuration Guide

This guide explains how to configure email sending for the SKOL authentication system.

## Quick Start

Copy the `.env.example` file and configure your SMTP settings:

```bash
cp .env.example .env
# Edit .env with your SMTP credentials
```

Then set the environment variables before starting the Django server:

```bash
# Source environment variables
export $(cat .env | xargs)

# Start Django server
python manage.py runserver
```

## Gmail Setup (Development/Testing)

Gmail is a good option for development and testing. You'll need to use an App Password (not your regular Gmail password).

### Step 1: Enable 2-Factor Authentication

1. Go to your [Google Account](https://myaccount.google.com/)
2. Navigate to Security
3. Enable 2-Step Verification if not already enabled

### Step 2: Generate App Password

1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
2. Select "Mail" as the app
3. Select "Other (Custom name)" as the device
4. Enter "SKOL Django" as the name
5. Click "Generate"
6. Copy the 16-character password (shown as `xxxx xxxx xxxx xxxx`)

### Step 3: Set Environment Variables

```bash
export EMAIL_HOST=smtp.gmail.com
export EMAIL_PORT=587
export EMAIL_HOST_USER=your.email@gmail.com
export EMAIL_HOST_PASSWORD=xxxx-xxxx-xxxx-xxxx  # App password from step 2
export EMAIL_USE_TLS=True
export DEFAULT_FROM_EMAIL=your.email@gmail.com
```

Or add to your `.env` file:

```bash
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your.email@gmail.com
EMAIL_HOST_PASSWORD=xxxx-xxxx-xxxx-xxxx
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=your.email@gmail.com
```

## SendGrid Setup (Production)

SendGrid is recommended for production environments as it provides better deliverability and analytics.

### Step 1: Create SendGrid Account

1. Sign up at [SendGrid](https://sendgrid.com)
2. Verify your email address
3. Complete the sender authentication process

### Step 2: Generate API Key

1. Go to Settings → [API Keys](https://app.sendgrid.com/settings/api_keys)
2. Click "Create API Key"
3. Choose "Restricted Access"
4. Grant "Mail Send" permission
5. Copy the API key (starts with `SG.`)

### Step 3: Set Environment Variables

```bash
export EMAIL_HOST=smtp.sendgrid.net
export EMAIL_PORT=587
export EMAIL_HOST_USER=apikey  # Literally the word "apikey"
export EMAIL_HOST_PASSWORD=SG.xxxxxxxxxxxxx  # Your API key from step 2
export EMAIL_USE_TLS=True
export DEFAULT_FROM_EMAIL=noreply@yourdomain.com
```

Or add to your `.env` file:

```bash
EMAIL_HOST=smtp.sendgrid.net
EMAIL_PORT=587
EMAIL_HOST_USER=apikey
EMAIL_HOST_PASSWORD=SG.xxxxxxxxxxxxx
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=noreply@yourdomain.com
```

## Testing Email Configuration

You can test your email configuration using the Django shell:

```bash
python manage.py shell
```

Then run:

```python
from django.core.mail import send_mail

send_mail(
    subject='Test Email from SKOL',
    message='This is a test email to verify SMTP configuration.',
    from_email='noreply@skol.example.com',
    recipient_list=['your.email@example.com'],
    fail_silently=False,
)
```

If the email sends successfully, you'll see `1` returned. Check your inbox (and spam folder) for the test email.

## Console Backend (Development)

For development without configuring a real SMTP server, you can use Django's console email backend which prints emails to the terminal instead of sending them.

### Option 1: Temporary Override

Start Django with:

```bash
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend python manage.py runserver
```

### Option 2: Settings Override

Edit `skolweb/settings.py` and temporarily change the EMAIL_BACKEND:

```python
# For development - emails print to console
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# For production - uncomment this and comment out above
# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
```

With console backend enabled, verification and password reset emails will appear in your terminal output.

## File Backend (Development)

Another option for development is the file backend, which saves emails as files in a directory:

```python
# In settings.py
EMAIL_BACKEND = 'django.core.mail.backends.filebased.EmailBackend'
EMAIL_FILE_PATH = '/tmp/skol-emails'  # Directory to store emails
```

Emails will be saved as `.eml` files that you can open with any email client.

## Troubleshooting

### Gmail: "Username and Password not accepted"

- Make sure you're using an App Password, not your regular Gmail password
- Verify that 2-Step Verification is enabled
- Check that the App Password is entered without spaces

### SendGrid: "Authentication failed"

- Ensure EMAIL_HOST_USER is exactly `apikey` (lowercase)
- Verify your API key is correct and has "Mail Send" permission
- Check that your SendGrid account is active and verified

### Emails Not Arriving

1. Check spam/junk folders
2. Verify EMAIL_USE_TLS is set to `True`
3. Test with the Django shell command above
4. Check Django logs for error messages
5. For Gmail, check [Less secure app access](https://myaccount.google.com/lesssecureapps) is not required (it's deprecated)

### Port Connection Errors

- Port 587: TLS (recommended) - set `EMAIL_USE_TLS=True`
- Port 465: SSL - set `EMAIL_USE_SSL=True` and `EMAIL_USE_TLS=False`
- Port 25: Usually blocked by ISPs for security reasons

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `EMAIL_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `EMAIL_PORT` | SMTP server port | `587` |
| `EMAIL_HOST_USER` | SMTP username | `your.email@gmail.com` |
| `EMAIL_HOST_PASSWORD` | SMTP password or API key | `xxxx-xxxx-xxxx-xxxx` |
| `EMAIL_USE_TLS` | Use TLS encryption | `True` |
| `DEFAULT_FROM_EMAIL` | Default sender address | `noreply@skol.example.com` |

## Security Notes

- **Never commit** your `.env` file to version control
- Use App Passwords for Gmail (never your main password)
- Rotate API keys regularly
- Use environment variables in production (not hardcoded values)
- Consider using secret management services for production

## Next Steps

After configuring email:

1. Start the Django development server
2. Navigate to `/accounts/register/`
3. Create a test account
4. Check your email for the verification link
5. Test the password reset flow

For production deployment, consider additional security measures like:
- Using HTTPS for all email links
- Implementing rate limiting on email sends
- Using a dedicated email service provider
- Setting up SPF, DKIM, and DMARC records for your domain
