from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    """
    Vista provisional para sistemas lineales
    """
    # Por ahora solo mostramos un mensaje
    return render(request, 'core/index.html')  # Redirigir a la página principal por ahora
