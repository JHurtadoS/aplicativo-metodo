from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    """
    Vista para el módulo de sistemas de ecuaciones lineales
    """
    # Renderizar el formulario inicial para sistemas lineales
    return render(request, 'sistemas_lineales/form.html')
