from django.urls import path
from . import views

app_name = 'interpolacion'

urlpatterns = [
    path('', views.interpolacion_view, name='interpolacion_form'),
    # Add other URLs for this app if needed
] 