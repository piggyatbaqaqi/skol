from django.contrib.auth.models import User
from django.db import models

class API(models.Model):
    id = models.AutoField(primary_key=True)
    taxon = models.TextField()
    description = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='apis')

    def __str__(self):
        return self.taxon
