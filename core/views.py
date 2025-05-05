from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse
from django.conf import settings
import os
from pathlib import Path
import json

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
    return render(request, 'core/historial.html', {'historial': json.dumps(historial)})

def ver_pdf(request, pdf_path):
    """
    Sirve archivos PDF generados por la aplicación
    """
    # Construir la ruta completa al PDF
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Si el path ya incluye 'pdfs/', no lo añadimos nuevamente
    if pdf_path.startswith('pdfs/'):
        pdf_file_path = BASE_DIR / 'static' / pdf_path
    else:
        pdf_file_path = BASE_DIR / 'static' / 'pdfs' / pdf_path
    
    # Para depuración
    print(f"Intentando servir PDF: {pdf_file_path}")
    
    # Verificar que el archivo existe y está dentro del directorio permitido
    if pdf_file_path.exists() and pdf_file_path.is_file() and str(pdf_file_path).startswith(str(BASE_DIR / 'static')):
        return FileResponse(open(pdf_file_path, 'rb'), content_type='application/pdf')
    else:
        return HttpResponse(f"PDF no encontrado: {pdf_file_path}", status=404)

def ejemplos(request):
    """
    Muestra una página con ejemplos de uso
    """
    return render(request, 'core/ejemplos.html')

def limpiar_historial(request):
    """
    Endpoint para limpiar el historial de operaciones del usuario
    """
    if request.method == 'POST':
        # Limpiar el historial en la sesión
        request.session['historial'] = []
        request.session.modified = True
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'Método no permitido'}, status=405)
