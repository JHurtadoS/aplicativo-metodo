from django.shortcuts import render, redirect
from django.views import View
from .forms import InterpolacionForm
from . import algorithms
from datetime import datetime
from core.utils import save_to_history  # Importar función de historial

# Mapping from form choice value to algorithm function
METHOD_MAP = {
    'lagrange': algorithms.lagrange,
    'newton': algorithms.newton,
    'cubic_splines': algorithms.natural_cubic_splines,
    'linear_regression': algorithms.linear_regression,
    # 'polynomial_regression': algorithms.polynomial_regression, # Add if implemented
}

THEORY_TEXT = {
    'lagrange': "La **Interpolación Polinómica de Lagrange** construye un polinomio único de grado ≤ N-1 que pasa exactamente por N puntos dados. Utiliza polinomios base L_i(x) que valen 1 en x_i y 0 en los otros x_j.",
    'newton': "La **Interpolación Polinómica de Newton (Diferencias Divididas)** también encuentra el polinomio único de grado ≤ N-1. Usa una tabla de diferencias divididas para calcular los coeficientes de forma eficiente y permite añadir puntos fácilmente.",
    'cubic_splines': "Los **Splines Cúbicos Naturales** dividen el intervalo en segmentos, construyendo un polinomio cúbico para cada uno. Estos se unen suavemente (con continuidad C2), lo que produce una curva más natural que un polinomio único de alto grado.",
    'linear_regression': "La **Regresión Lineal por Mínimos Cuadrados** encuentra la recta (y = ax + b) que mejor se ajusta a un conjunto de puntos, minimizando la suma de los cuadrados de las distancias verticales entre los puntos y la recta. No requiere pasar exactamente por los puntos.",
}

class InterpolacionView(View):
    template_name = 'interpolacion/interpolacion_form.html'
    
    def get(self, request, *args, **kwargs):
        form = InterpolacionForm()
        context = {'form': form}
        return render(request, self.template_name, context)
    
    def post(self, request, *args, **kwargs):
        form = InterpolacionForm(request.POST)
        context = {'form': form}
        
        if form.is_valid():
            # Get points data - ya procesados por el método clean_points
            points = form.cleaned_data['points']  # Esto ya es una lista de tuplas [(x1,y1), (x2,y2), ...]
            method_key = form.cleaned_data['method']
            
            # No necesitamos procesar los puntos nuevamente
            try:
                # Verificar que hay suficientes puntos
                if len(points) < 2:
                    raise ValueError("Se requieren al menos 2 puntos para interpolación/regresión")
                
                # Verificación específica para splines
                if method_key == 'cubic_splines' and len(points) < 3:
                    raise ValueError("Se requieren al menos 3 puntos para splines cúbicos")
                
                # Check if method exists
                if method_key in METHOD_MAP:
                    algorithm_func = METHOD_MAP[method_key]
                    
                    # Para métodos que usan el solver
                    if method_key in ['lagrange', 'newton', 'cubic_splines'] and 'solver' in form.cleaned_data:
                        solver = form.cleaned_data['solver']
                        
                        try:
                            # Usar el método modernizado que recibe el solver como parámetro
                            result_data = algorithm_func(points, solver=solver)
                            
                            # Guardar en historial
                            input_data = {
                                'puntos': [(float(p[0]), float(p[1])) for p in points],
                                'solver': solver
                            }
                            save_to_history(
                                request, 
                                'interpolacion', 
                                method_key, 
                                input_data, 
                                result_data
                            )
                            
                            # Add results to context
                            context['results'] = result_data
                            context['theory'] = THEORY_TEXT.get(method_key, "")
                        except ValueError as e:
                            context['error_message'] = str(e)
                    else:
                        # Para otros métodos, mantener comportamiento original
                        try:
                            result_data = algorithm_func(points)
                            
                            # Guardar en historial
                            input_data = {
                                'puntos': [(float(p[0]), float(p[1])) for p in points]
                            }
                            save_to_history(
                                request, 
                                'interpolacion', 
                                method_key, 
                                input_data, 
                                result_data
                            )
                            
                            context['results'] = result_data
                            context['theory'] = THEORY_TEXT.get(method_key, "")
                        except ValueError as e:
                            context['error_message'] = str(e)
                else:
                    context['error_message'] = f"Método '{method_key}' no encontrado."
            
            except Exception as e:
                context['error_message'] = f"Error al procesar los datos: {str(e)}"
        
        return render(request, self.template_name, context)

# Alias for clarity if needed, matching the URL name perhaps
interpolacion_view = InterpolacionView.as_view()

def index(request):
    """
    Vista para el módulo de interpolación y ajuste de curvas
    """
    # Renderizar el formulario inicial para interpolación
    return render(request, 'interpolacion/form.html')
