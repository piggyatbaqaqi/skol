"""URL configuration for contact app."""
from django.urls import path
from . import views

app_name = 'contact'

urlpatterns = [
    path('', views.contact_view, name='contact'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('success/', views.contact_success_view, name='contact_success'),
    path('feedback/success/', views.feedback_success_view, name='feedback_success'),
]
