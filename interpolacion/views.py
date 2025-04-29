from django.shortcuts import render, redirect
from django.views import View
from .forms import InterpolacionForm
from . import algorithms
# from core.utils import add_to_history # Assuming a history utility exists

# Mapping from form choice value to algorithm function
METHOD_MAP = {
    'lagrange': algorithms.lagrange_interpolation,
    'newton': algorithms.newton_interpolation,
    'linear_regression': algorithms.linear_regression,
    # 'polynomial_regression': algorithms.polynomial_regression, # Add if implemented
}

THEORY_TEXT = {
    'lagrange': "La **Interpolación Polinómica de Lagrange** construye un polinomio único de grado ≤ N-1 que pasa exactamente por N puntos dados. Utiliza polinomios base L_i(x) que valen 1 en x_i y 0 en los otros x_j.",
    'newton': "La **Interpolación Polinómica de Newton (Diferencias Divididas)** también encuentra el polinomio único de grado ≤ N-1. Usa una tabla de diferencias divididas para calcular los coeficientes de forma eficiente y permite añadir puntos fácilmente.",
    'linear_regression': "La **Regresión Lineal por Mínimos Cuadrados** encuentra la recta (y = ax + b) que mejor se ajusta a un conjunto de puntos, minimizando la suma de los cuadrados de las distancias verticales entre los puntos y la recta. No requiere pasar exactamente por los puntos.",
}

class InterpolacionView(View):
    template_name = 'interpolacion/interpolacion_form.html'

    def get(self, request, *args, **kwargs):
        form = InterpolacionForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = InterpolacionForm(request.POST)
        context = {'form': form}

        if form.is_valid():
            points = form.cleaned_data['points']
            method_key = form.cleaned_data['method']

            # Get the corresponding algorithm function
            algorithm_func = METHOD_MAP.get(method_key)

            if algorithm_func:
                # Call the algorithm
                result_data = algorithm_func(points)

                # Add results to context
                context['results'] = result_data
                context['theory'] = THEORY_TEXT.get(method_key, "")

                # Add to history (if history feature is implemented)
                # if 'error' not in result_data:
                #     add_to_history(request, 'interpolacion', method_key, {'points': points}, result_data)

            else:
                # Should not happen if form choices and METHOD_MAP match
                context['error_message'] = f"Método '{method_key}' no implementado."

        # Re-render the page with form (potentially with errors) and results/error
        return render(request, self.template_name, context)

# Alias for clarity if needed, matching the URL name perhaps
interpolacion_view = InterpolacionView.as_view()

def index(request):
    """
    Vista para el módulo de interpolación y ajuste de curvas
    """
    # Renderizar el formulario inicial para interpolación
    return render(request, 'interpolacion/form.html')
