from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    """
    Vista para el módulo de interpolación y ajuste de curvas
    """
    # Renderizar el formulario inicial para interpolación
    return render(request, 'interpolacion/form.html')
