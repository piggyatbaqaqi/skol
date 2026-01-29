from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .forms import CustomAuthenticationForm

app_name = 'accounts'

urlpatterns = [
    # Registration
    path('register/', views.register, name='register'),
    path('register/complete/', views.register_complete, name='register_complete'),
    path('verify-email/<uidb64>/<token>/', views.verify_email, name='verify_email'),
    path('resend-verification/', views.resend_verification, name='resend_verification'),

    # Login/Logout
    path('login/', auth_views.LoginView.as_view(
        template_name='accounts/login.html',
        redirect_authenticated_user=True,
        authentication_form=CustomAuthenticationForm,
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Password Reset
    path('password-reset/', views.CustomPasswordResetView.as_view(
        template_name='accounts/password_reset.html',
        email_template_name='emails/password_reset.txt',
        html_email_template_name='emails/password_reset.html',
        success_url='/accounts/password-reset/sent/',
    ), name='password_reset'),

    path('password-reset/sent/', auth_views.PasswordResetDoneView.as_view(
        template_name='accounts/password_reset_sent.html',
    ), name='password_reset_done'),

    path('password-reset/confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='accounts/password_reset_confirm.html',
        success_url='/accounts/password-reset/complete/',
    ), name='password_reset_confirm'),

    path('password-reset/complete/', auth_views.PasswordResetCompleteView.as_view(
        template_name='accounts/password_reset_complete.html',
    ), name='password_reset_complete'),

    # Social account connections
    path('connections/', views.social_connections, name='social_connections'),

    # Account settings
    path('settings/', views.account_settings, name='account_settings'),
]
