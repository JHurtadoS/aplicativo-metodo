from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse
from django.conf import settings
import os
from pathlib import Path
import json
from django.contrib import messages
from .forms import DerivacionForm, IntegracionForm, EDOForm
from .utils import (
    NMResult, calcular_derivada, calcular_integral, calcular_edo,
    generar_grafico_derivada, generar_grafico_integral, generar_grafico_edo,
    save_to_history
)

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

# =============================================================================
# VISTAS PARA DERIVACIÓN NUMÉRICA
# =============================================================================

def derivacion_form(request):
    """Vista del formulario de derivación numérica"""
    if request.method == 'POST':
        form = DerivacionForm(request.POST)
        if form.is_valid():
            # Extraer datos del formulario
            function_str = form.cleaned_data['function']
            x0 = form.cleaned_data['x0']
            h = form.cleaned_data['h']
            method = form.cleaned_data['method']
            
            try:
                # Calcular derivada
                resultado, pasos, error, func, expr, error_info = calcular_derivada(
                    function_str, x0, h, method
                )
                
                # Generar gráfico
                grafico_path = generar_grafico_derivada(func, expr, x0, h, method, resultado)
                
                # Crear objeto resultado
                nm_result = NMResult(
                    tema='derivacion',
                    metodo=method,
                    entrada={
                        'function': function_str,
                        'x0': x0,
                        'h': h,
                        'method': method
                    },
                    pasos=pasos,
                    valor=resultado,
                    error=error,
                    grafico_path=grafico_path
                )
                nm_result.detalles = {
                    'expression': str(expr),
                    'error_info': error_info
                }
                
                # Guardar en historial
                save_to_history(request, 'derivacion', method, nm_result.entrada, nm_result.__dict__)
                
                return render(request, 'core/derivacion_resultado.html', {
                    'result': nm_result,
                    'form_data': form.cleaned_data
                })
                
            except Exception as e:
                messages.error(request, f"Error en el cálculo: {str(e)}")
                
    else:
        form = DerivacionForm()
    
    return render(request, 'core/derivacion_form.html', {'form': form})

# =============================================================================
# VISTAS PARA INTEGRACIÓN NUMÉRICA
# =============================================================================

def integracion_form(request):
    """Vista del formulario de integración numérica"""
    if request.method == 'POST':
        form = IntegracionForm(request.POST)
        if form.is_valid():
            # Extraer datos del formulario
            function_str = form.cleaned_data['function']
            a = form.cleaned_data['a']
            b = form.cleaned_data['b']
            n = form.cleaned_data['n']
            method = form.cleaned_data['method']
            
            try:
                # Calcular integral
                resultado, pasos, error, func, expr, x_vals, f_vals, error_info = calcular_integral(
                    function_str, a, b, n, method
                )
                
                # Generar gráfico
                grafico_path = generar_grafico_integral(func, expr, a, b, x_vals, f_vals, method, resultado)
                
                # Crear objeto resultado
                nm_result = NMResult(
                    tema='integracion',
                    metodo=method,
                    entrada={
                        'function': function_str,
                        'a': a,
                        'b': b,
                        'n': n,
                        'method': method
                    },
                    pasos=pasos,
                    valor=resultado,
                    error=error,
                    grafico_path=grafico_path
                )
                nm_result.detalles = {
                    'expression': str(expr),
                    'x_vals': x_vals.tolist(),
                    'f_vals': f_vals,
                    'error_info': error_info
                }
                
                # Guardar en historial
                save_to_history(request, 'integracion', method, nm_result.entrada, nm_result.__dict__)
                
                return render(request, 'core/integracion_resultado.html', {
                    'result': nm_result,
                    'form_data': form.cleaned_data
                })
                
            except Exception as e:
                messages.error(request, f"Error en el cálculo: {str(e)}")
                
    else:
        form = IntegracionForm()
    
    return render(request, 'core/integracion_form.html', {'form': form})

# =============================================================================
# VISTAS PARA EDOs
# =============================================================================

def edo_form(request):
    """Vista del formulario de EDOs"""
    if request.method == 'POST':
        form = EDOForm(request.POST)
        if form.is_valid():
            # Extraer datos del formulario
            function_str = form.cleaned_data['function']
            t0 = form.cleaned_data['t0']
            y0 = form.cleaned_data['y0']
            h = form.cleaned_data['h']
            n_steps = form.cleaned_data['n_steps']
            method = form.cleaned_data['method']
            
            try:
                # Resolver EDO con análisis estadístico
                t_vals, y_vals, pasos, func, expr, analisis = calcular_edo(
                    function_str, t0, y0, h, n_steps, method
                )
                
                # Generar gráfico
                grafico_path = generar_grafico_edo(t_vals, y_vals, expr, method, t0, y0)
                
                # Crear objeto resultado
                nm_result = NMResult(
                    tema='edo',
                    metodo=method,
                    entrada={
                        'function': function_str,
                        't0': t0,
                        'y0': y0,
                        'h': h,
                        'n_steps': n_steps,
                        'method': method
                    },
                    pasos=pasos,
                    valor=y_vals[-1],  # Valor final de y
                    error=None,  # Por ahora no calculamos error para EDOs
                    grafico_path=grafico_path
                )
                
                # Agregar detalles y análisis
                nm_result.detalles = {
                    'expression': str(expr),
                    't_vals': t_vals,
                    'y_vals': y_vals,
                    'final_time': t_vals[-1]
                }
                nm_result.analisis = analisis
                
                # Guardar en historial
                save_to_history(request, 'edo', method, nm_result.entrada, nm_result.__dict__)
                
                return render(request, 'core/edo_resultado.html', {
                    'result': nm_result,
                    'form_data': form.cleaned_data
                })
                
            except Exception as e:
                messages.error(request, f"Error en el cálculo: {str(e)}")
                
    else:
        form = EDOForm()
    
    return render(request, 'core/edo_form.html', {'form': form})
