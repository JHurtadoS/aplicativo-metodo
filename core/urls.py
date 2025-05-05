from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('historial/', views.historial, name='historial'),
    path('ejemplos/', views.ejemplos, name='ejemplos'),
    path('pdf/<path:pdf_path>/', views.ver_pdf, name='ver_pdf'),
    path('api/limpiar-historial/', views.limpiar_historial, name='limpiar_historial'),
] 