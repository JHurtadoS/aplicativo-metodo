from django.urls import path
from . import views

app_name = 'sistemas_lineales'

urlpatterns = [
    path('', views.sistema_lineal_solver_view, name='solver'),
    # Add other URLs for this app if needed
] 