"""
Django settings for skolweb project.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SKOL_DJANGO_ROOT is set when running via deb package (/opt/skol/django)
# Fall back to BASE_DIR for development
SKOL_DJANGO_ROOT = Path(os.environ.get('SKOL_DJANGO_ROOT', BASE_DIR))

LOG_FILE_PATH = "/var/log/skol/skolweb.log"

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-skol-dev-key-change-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'search',
    'accounts',
    'contact',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'skolweb.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [SKOL_DJANGO_ROOT / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'skolweb.context_processors.script_name',
            ],
        },
    },
]

WSGI_APPLICATION = 'skolweb.wsgi.application'


# Database
# Using SQLite for simplicity - no database models needed for this app
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# URL prefix when running behind a reverse proxy at a subpath
# e.g., FORCE_SCRIPT_NAME='/skol' for https://example.com/skol/
FORCE_SCRIPT_NAME = os.environ.get('FORCE_SCRIPT_NAME', None)

# Static files (CSS, JavaScript, Images)
# Prefix with FORCE_SCRIPT_NAME if set
_script_name = FORCE_SCRIPT_NAME or ''
STATIC_URL = f'{_script_name}/static/'
STATICFILES_DIRS = [SKOL_DJANGO_ROOT / 'static']

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS settings - allow all origins for development
CORS_ALLOW_ALL_ORIGINS = True

# HTTPS/Proxy settings
# When behind a reverse proxy (Apache/nginx), trust the X-Forwarded-Proto header
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Set these to True in production when using HTTPS
CSRF_COOKIE_SECURE = os.environ.get('SKOL_HTTPS', 'False') == 'True'
SESSION_COOKIE_SECURE = os.environ.get('SKOL_HTTPS', 'False') == 'True'

# CSRF trusted origins - add your domain here
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get('CSRF_TRUSTED_ORIGINS', '').split(',')
    if origin.strip()
]

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
    # Allow unauthenticated access to API endpoints (no CSRF required)
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}'

# CouchDB configuration
COUCHDB_HOST = os.environ.get('COUCHDB_HOST', 'localhost')
COUCHDB_PORT = int(os.environ.get('COUCHDB_PORT', '5984'))
COUCHDB_USERNAME = os.environ.get('COUCHDB_USER', 'admin')
COUCHDB_PASSWORD = os.environ.get('COUCHDB_PASSWORD', 'SU2orange!')
COUCHDB_URL = os.environ.get('COUCHDB_URL', f'http://{COUCHDB_HOST}:{COUCHDB_PORT}')

# Email Configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', '')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True') == 'True'
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'noreply@skol.example.com')

# Contact/Feedback Email Recipients
CONTACT_EMAIL = os.environ.get('CONTACT_EMAIL', 'piggy.yarroll+skol-contact@gmail.com')
FEEDBACK_EMAIL = os.environ.get('FEEDBACK_EMAIL', 'piggy.yarroll+skol-feedback@gmail.com')

# Authentication Configuration
LOGIN_URL = f'{_script_name}/accounts/login/'
LOGIN_REDIRECT_URL = f'{_script_name}/'
LOGOUT_REDIRECT_URL = f'{_script_name}/'
PASSWORD_RESET_TIMEOUT = 259200  # 3 days in seconds

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '[{levelname}] {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler', # A common file handler
            'filename': LOG_FILE_PATH,
            'maxBytes': 1024 * 1024 * 5, # 5 MB
            'backupCount': 5,
            'level': 'DEBUG',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'django.core.mail': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'accounts': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'search': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'contact': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
