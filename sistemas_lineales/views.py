from django.shortcuts import render, redirect
from django.views import View
from .forms import SistemaLinealForm
from . import algorithms
import logging # Import logging
import uuid
# from core.utils import add_to_history # Assuming a history utility exists

logger = logging.getLogger(__name__) # Get logger for this module

# Mapping form choice value to algorithm function (only Gauss for now)
METHOD_MAP = {
    'gauss': algorithms.gaussian_elimination,
    'lu': algorithms.lu_decomposition,
    'jacobi': algorithms.jacobi_method,
    'gauss_seidel': algorithms.gauss_seidel_method,
}

THEORY_TEXT = {
    'gauss': "La **Eliminación Gaussiana** transforma un sistema Ax=b en un sistema equivalente Ux=y (donde U es triangular superior) mediante operaciones elementales de fila (intercambio, escalado, suma). Luego, resuelve por sustitución regresiva. Se usa **pivoteo parcial** para mejorar la estabilidad numérica, intercambiando filas para usar el mayor pivote posible en cada columna.",
    'lu': "La **Descomposición LU** factoriza la matriz A en el producto de una matriz triangular inferior L y una triangular superior U (A=LU). Luego resuelve Ly=b y Ux=y. Es eficiente si se necesita resolver Ax=b para múltiples vectores b.",
    'jacobi': "El **Método de Jacobi** es iterativo. Despeja cada variable x_i de la i-ésima ecuación y calcula la siguiente aproximación x^(k+1) usando los valores de la iteración anterior x^(k). Requiere que la matriz sea diagonalmente dominante para garantizar convergencia.",
    'gauss_seidel': "El **Método de Gauss-Seidel** es similar a Jacobi, pero utiliza los valores de x_i más recientes tan pronto como se calculan dentro de la misma iteración. Generalmente converge más rápido que Jacobi si lo hace.",
}

class SistemaLinealView(View):
    template_name = 'sistemas_lineales/solver.html' # Adjusted template name

    def get(self, request, *args, **kwargs):
        form = SistemaLinealForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = SistemaLinealForm(request.POST)
        context = {'form': form}

        if form.is_valid():
            A = form.cleaned_data.get('matrix_A')
            b = form.cleaned_data.get('vector_b')
            method_key = form.cleaned_data.get('method')
            initial_guess = form.cleaned_data.get('initial_guess_vector')
            tolerance = form.cleaned_data.get('tolerance')
            max_iterations = form.cleaned_data.get('max_iterations')

            # Ensure A and b were successfully parsed in the clean method
            if A is None or b is None:
                context['error_message'] = "Error interno: No se pudieron obtener los datos procesados de la matriz/vector del formulario."
                logger.error("matrix_A or vector_b is None after form validation. Form errors: %s", form.errors.as_json())
                return render(request, self.template_name, context)

            # Get the corresponding algorithm function
            algorithm_func = None
            theory = ""
            if method_key:
                 algorithm_func = METHOD_MAP.get(method_key)
                 theory = THEORY_TEXT.get(method_key, "")
            else:
                 context['error_message'] = "No se seleccionó ningún método."
                 logger.warning("No method selected in POST request.")
                 return render(request, self.template_name, context)

            if algorithm_func:
                result_data = None
                logger.info("Calling algorithm '%s' for matrix A: %s", method_key, A.shape)
                try:
                    if method_key == 'gauss':
                        result_data = algorithm_func(A, b)
                    elif method_key == 'lu':
                        result_data = algorithm_func(A, b)
                    elif method_key in ['jacobi', 'gauss_seidel']:
                        # Ensure iterative params are not None before calling
                        if initial_guess is None or tolerance is None or max_iterations is None:
                             context['error_message'] = "Faltan parámetros requeridos (vector inicial, tolerancia, iteraciones) para el método iterativo."
                             logger.warning("Missing iterative parameters for method %s", method_key)
                        else:
                            result_data = algorithm_func(A, b, initial_guess, tolerance, max_iterations)
                            
                            # Generate plot for iterative methods if convergence data is available
                            if result_data and 'error' not in result_data and 'error_plot_data' in result_data:
                                error_data = result_data['error_plot_data']
                                if error_data and 'errors' in error_data and 'iterations' in error_data and error_data['errors']:
                                    # Generate unique ID for this plot
                                    plot_uuid = str(uuid.uuid4())
                                    # Create plot and get the path
                                    plot_path = algorithms.plot_error_convergence(
                                        error_data['errors'], 
                                        error_data['iterations'],
                                        f"Convergencia del Error - Método {method_key.capitalize()}",
                                        plot_uuid
                                    )
                                    # Add plot path to result data
                                    result_data['plot_path'] = plot_path
                                    logger.info("Generated error convergence plot: %s", plot_path)
                    else:
                         # This case should ideally not be reached if METHOD_MAP is correct
                         context['error_message'] = f"Lógica para método '{method_key}' no implementada en la vista."
                         logger.error("Method key '%s' has no corresponding view logic.", method_key)

                except Exception as e:
                    logger.exception("Error executing algorithm '%s'", method_key) # Log full exception
                    context['error_message'] = f"Error al ejecutar el algoritmo: {e}"
                    result_data = {'error': str(e), 'steps': [f"Error Inesperado: {e}"]}

                if result_data:
                    context['results'] = result_data
                    context['theory'] = theory
                    # Add to history (if history feature is implemented)
                    # if 'error' not in result_data:
                    #    input_data = {'matrix_input': form.cleaned_data['matrix_input']}
                    #    if method_key in ['jacobi', 'gauss_seidel']:
                    #        input_data.update({'initial_guess': initial_guess.tolist(), 'tolerance': tolerance, 'max_iterations': max_iterations})
                    #     add_to_history(request, 'sistemas_lineales', method_key, input_data, result_data)
            else:
                context['error_message'] = f"Método '{method_key}' no encontrado en el mapeo."
                logger.error("Method key '%s' not found in METHOD_MAP.", method_key)

        else:
             logger.warning("Form is invalid: %s", form.errors.as_json())
             context['error_message'] = "El formulario contiene errores. Por favor, revise los campos marcados."

        # Re-render the page with form (potentially with errors) and results/error
        return render(request, self.template_name, context)

# Alias for clarity
sistema_lineal_solver_view = SistemaLinealView.as_view()
