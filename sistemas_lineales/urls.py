from django.urls import path
from . import views

urlpatterns = [
    # Temporalmente, redirigimos a una vista placeholder
    path('', views.index, name='sistemas_index'),
] 