from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import viewsets
from .serializers import APISerializer
from .models import API

class APIView(viewsets.ModelViewSet):
    serializer_class = APISerializer
    queryset = API.objects.all()
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)