from django.shortcuts import render, redirect
from django.views import View
from .forms import SistemaLinealForm
from . import algorithms
import logging # Import logging
import uuid
from core.utils import save_to_history  # Importar función de historial

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
        context = {'form': form}
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        form = SistemaLinealForm(request.POST)
        context = {'form': form}

        if form.is_valid():
            # Get cleaned data - usar los nombres correctos de los campos
            matrix_data = form.cleaned_data.get('matrix_data')
            vector_data = form.cleaned_data.get('vector_data')
            method_key = form.cleaned_data.get('method')
            
            # Get theory text based on method
            theory = THEORY_TEXT.get(method_key, "")
            
            # Process matrix input to NumPy array
            from . import algorithms  # Import parser
            import numpy as np
            import json
            
            try:
                # Attempt to parse matrix and vector from JSON
                A = np.array(json.loads(matrix_data), dtype=float)
                b = np.array(json.loads(vector_data), dtype=float)
                
                logger.info(f"Successfully parsed matrix A of shape {A.shape} and vector b of shape {b.shape}")
                
                # Verify dimensions
                if A.shape[0] != A.shape[1]:
                    raise ValueError(f"Matrix must be square. Got {A.shape[0]}x{A.shape[1]}")
                
                if A.shape[0] != b.shape[0]:
                    raise ValueError(f"Matrix rows ({A.shape[0]}) must match vector length ({b.shape[0]})")
                
                # Check if method exists in the map
                if method_key in METHOD_MAP:
                    algorithm_func = METHOD_MAP[method_key]
                    result_data = None
                    
                    # Branch by method for additional parameters
                    if method_key in ['jacobi', 'gauss_seidel']:
                        # Additional parameters for iterative methods
                        try:
                            tolerance = float(form.cleaned_data.get('tolerance', 0.0001))
                            max_iterations = int(form.cleaned_data.get('max_iterations', 100))
                            
                            # Initial guess (default to zeros)
                            initial_guess = form.cleaned_data.get('initial_guess', None)
                            if initial_guess and initial_guess.strip():
                                try:
                                    # Parse as array with same shape as b
                                    initial_guess = np.array([float(x) for x in initial_guess.split()], dtype=np.float64)
                                    if initial_guess.shape[0] != b.shape[0]:
                                        raise ValueError(f"Initial guess length ({initial_guess.shape[0]}) doesn't match system size ({b.shape[0]})")
                                except Exception as e:
                                    logger.warning(f"Error parsing initial guess: {e}")
                                    initial_guess = np.zeros_like(b)  # Default to zeros
                            else:
                                initial_guess = np.zeros_like(b)  # Default to zeros
                            
                            # Execute iterative method
                            logger.info(f"Executing {method_key} with tolerance={tolerance}, max_iterations={max_iterations}")
                            result_data = algorithm_func(A, b, initial_guess, tolerance, max_iterations)
                            
                        except Exception as e:
                            logger.exception(f"Error with iterative method: {e}")
                            raise ValueError(f"Error en parámetros de método iterativo: {e}")
                    else:
                        # Direct methods (Gauss, LU)
                        logger.info(f"Executing direct method: {method_key}")
                        result_data = algorithm_func(A, b)
                    
                    # Guardar en historial
                    if result_data and 'error' not in result_data:
                        input_data = {
                            'matrix_data': matrix_data,
                            'vector_data': vector_data
                        }
                        if method_key in ['jacobi', 'gauss_seidel']:
                            input_data.update({
                                'initial_guess': initial_guess.tolist() if hasattr(initial_guess, 'tolist') else initial_guess, 
                                'tolerance': tolerance, 
                                'max_iterations': max_iterations
                            })
                        save_to_history(request, 'sistemas_lineales', method_key, input_data, result_data)
                    
                    context['results'] = result_data
                    context['theory'] = theory

                else:
                    context['error_message'] = f"Método '{method_key}' no encontrado en el mapeo."
                    logger.error(f"Method key '{method_key}' not found in METHOD_MAP.")

            except Exception as e:
                logger.exception(f"Error executing algorithm '{method_key}'")
                context['error_message'] = f"Error al ejecutar el algoritmo: {str(e)}"
                result_data = {'error': str(e), 'steps': [f"Error Inesperado: {str(e)}"]}
                context['results'] = result_data

        else:
            logger.warning(f"Form is invalid: {form.errors.as_json()}")
            context['error_message'] = "El formulario contiene errores. Por favor, revise los campos marcados."

        return render(request, self.template_name, context)

# Alias for clarity
sistema_lineal_solver_view = SistemaLinealView.as_view()
