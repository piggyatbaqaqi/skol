from django.contrib import admin
from .models import API

class APIAdmin(admin.ModelAdmin):
    list_display = ('id', 'taxon', 'description', 'user')
    search_fields = ('taxon', 'description', 'user__username')
    list_filter = ('user',)

admin.site.register(API, APIAdmin)
