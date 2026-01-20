"""
URL configuration for skolweb project.
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),
    path('accounts/', include('accounts.urls')),
    path('contact/', include('contact.urls')),
    path('about/', TemplateView.as_view(template_name='about.html'), name='about'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
