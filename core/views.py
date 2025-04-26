from django.shortcuts import render

# Create your views here.

def index(request):
    """
    Página principal que muestra el menú de acceso a todos los módulos
    """
    return render(request, 'core/index.html')

def historial(request):
    """
    Página que muestra el historial de operaciones realizadas
    """
    # Obtener historial de la sesión (o lista vacía si no existe)
    historial = request.session.get('historial', [])
    return render(request, 'core/historial.html', {'historial': historial})
