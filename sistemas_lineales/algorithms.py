import numpy as np
import sympy
import matplotlib.pyplot as plt
import os
import uuid
import json
from core.utils import fix_nested_matrices

def format_matrix_sympy(matrix):
    """Formats a NumPy matrix using SymPy for LaTeX rendering."""
    try:
        # Convert float to rational for potentially cleaner representation
        # Using a reasonable max_denominator to avoid overly complex fractions
        # Or simply use sympy.Matrix for direct LaTeX output of floats
        sp_matrix = sympy.Matrix(matrix).evalf(4) # Display with 4 significant digits
        
        # A more direct approach to generate proper LaTeX for matrices
        # Bypass sympy's potentially problematic nesting
        if isinstance(matrix, np.ndarray) and len(matrix.shape) > 1:
            rows, cols = matrix.shape
            
            # Generate the LaTeX manually to avoid nested matrices
            latex_str = "\\begin{bmatrix}\n"
            for i in range(rows):
                row_str = " & ".join([f"{matrix[i, j]:.4f}" for j in range(cols)])
                latex_str += row_str
                if i < rows - 1:
                    latex_str += " \\\\\n"
                else:
                    latex_str += "\n"
            latex_str += "\\end{bmatrix}"
            
            return latex_str
        else:
            # Fall back to SymPy for vectors or other cases
            latex_str = sympy.latex(sp_matrix)
            
            # Fix nested matrix environments regardless of where they occur
            return fix_nested_matrices(latex_str)
    except Exception as e:
        # Fallback to simple string representation if our formatting fails
        print(f"Error formatting matrix: {e}")
        return np.array_str(matrix, precision=4)

def format_lu_element(letter, i, j):
    """Creates a properly escaped LaTeX subscript for matrix elements.
    
    Args:
        letter: The matrix letter (e.g., 'L', 'U', 'A')
        i, j: The row and column indices (1-based)
    
    Returns:
        Properly formatted LaTeX for the matrix element
    """
    return f"{letter}_{{{i},{j}}}"

def format_lu_step(step_type, element, equation, value):
    """Creates a consistent format for LU decomposition steps.
    
    Args:
        step_type: Type of step (e.g., 'Calculando L_21')
        element: The matrix element being calculated (e.g., 'L_{2,1}')
        equation: The equation in LaTeX format
        value: The final calculated value
        
    Returns:
        Formatted step string with consistent LaTeX
    """
    return f"- {step_type}: ${equation} = {value:.4f}$"

def gaussian_elimination(A, b, exact_solution=None):
    """
    Solves Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        exact_solution: Optional exact solution vector for error comparison
    """
    steps = ["## Inicio: Eliminación Gaussiana con Pivoteo Parcial"]
    try:
        n = len(b)
        Ab = np.concatenate((A.astype(float), b.reshape(-1, 1).astype(float)), axis=1)

        steps.append("**Sistema Original (Matriz Aumentada):**")
        steps.append(f"${format_matrix_sympy(Ab)}$")

        # Forward Elimination with Pivoting
        for k in range(n - 1):
            steps.append(f"\n**Paso {k + 1}: Eliminación en columna {k + 1}**")

            # --- Pivoting --- #
            pivot_row = k + np.argmax(np.abs(Ab[k:, k]))
            if pivot_row != k:
                steps.append(f"- Pivoteo: Intercambiando fila {k + 1} con fila {pivot_row + 1} (mayor pivote: {Ab[pivot_row, k]:.4f}) ")
                Ab[[k, pivot_row]] = Ab[[pivot_row, k]] # Swap rows
                steps.append(f"  Matriz después del intercambio:")
                steps.append(f"  ${format_matrix_sympy(Ab)}$")
            else:
                steps.append(f"- Pivote: {Ab[k, k]:.4f} en fila {k+1} (ya es el mayor en valor absoluto)")

            # Check for near-zero pivot after pivoting
            if np.isclose(Ab[k, k], 0):
                error_msg = f"Error: Pivote cercano a cero ({Ab[k, k]:.2e}) encontrado en la columna {k + 1} después del pivoteo. La matriz puede ser singular."
                steps.append(f"**{error_msg}**")
                return {'error': error_msg, 'steps': steps}
            # --- End Pivoting --- #

            # --- Elimination --- #
            steps.append(f"- Eliminando elementos debajo del pivote en columna {k+1}:")
            for i in range(k + 1, n):
                if not np.isclose(Ab[i, k], 0): # Only proceed if element is not already zero
                    factor = Ab[i, k] / Ab[k, k]
                    steps.append(f"  - Fila {i + 1} = Fila {i + 1} - ({factor:.4f}) \times Fila {k + 1}") # Using times symbol
                    Ab[i, k:] = Ab[i, k:] - factor * Ab[k, k:]
                    # It's good practice to explicitly zero out the element due to potential float inaccuracies
                    Ab[i, k] = 0.0
                else:
                    steps.append(f"  - Elemento en Fila {i+1}, Columna {k+1} ya es cero.")

            steps.append(f"  Matriz después de la eliminación en columna {k+1}:")
            steps.append(f"  ${format_matrix_sympy(Ab)}$")
            # --- End Elimination --- #

        # Check for singularity based on last pivot
        if np.isclose(Ab[n - 1, n - 1], 0):
            error_msg = f"Error: Elemento diagonal final ({Ab[n - 1, n - 1]:.2e}) es cercano a cero. El sistema puede no tener solución única (singular)."
            steps.append(f"\n**{error_msg}**")
            return {'error': error_msg, 'steps': steps}

        steps.append("\n**Fase de Eliminación Completada. Matriz Triangular Superior:**")
        steps.append(f"${format_matrix_sympy(Ab)}$")

        # Back Substitution
        x = np.zeros(n)
        steps.append("\n**Fase de Sustitución Regresiva:**")
        for i in range(n - 1, -1, -1):
            # Format step using LaTeX
            sum_ax = np.dot(Ab[i, i + 1:n], x[i + 1:n])
            # Handle potential division by zero which should have been caught earlier
            if np.isclose(Ab[i, i], 0):
                error_msg = f"Error: División por cero al calcular x_{{{i+1}}}."
                steps.append(f"**{error_msg}**")
                return {'error': error_msg, 'steps': steps}

            x[i] = (Ab[i, n] - sum_ax) / Ab[i, i]
            # Use a simpler approach without complex subscripts causing KaTeX parsing errors
            if len(Ab[i, i + 1:n]) > 0:
                # Show the calculation without the problematic sum notation
                step_latex = f" $ x_{{{i + 1}}} = \\frac{{ b'_{{{i+1}}} - \\text{{términos conocidos}} }}{{ U_{{{i+1}{i+1}}} }} = \\frac{{ {Ab[i, n]:.4f} - ({sum_ax:.4f}) }}{{ {Ab[i, i]:.4f} }} = {x[i]:.6f} $ "
            else:
                # No sum needed for the last variable
                step_latex = f" $ x_{{{i + 1}}} = \\frac{{ b'_{{{i+1}}} }}{{ U_{{{i+1}{i+1}}} }} = \\frac{{ {Ab[i, n]:.4f} }}{{ {Ab[i, i]:.4f} }} = {x[i]:.6f} $ "
            
            steps.append(f"- Calculando x_{{{i + 1}}}: {step_latex}")


        steps.append("\n**Solución Final:**")
        steps.append(f"$x = {format_matrix_sympy(x.reshape(-1,1))}$") # Format vector as column matrix
        
        # Calculate error metrics if exact solution is provided
        error_metrics = None
        if exact_solution is not None:
            error_metrics = calculate_error_metrics(exact_solution, x)
            steps.append("\n**Comparación con Solución Exacta:**")
            steps.append(f"Solución Exacta: $x_{{exact}} = {format_matrix_sympy(np.array(exact_solution).reshape(-1,1))}$")
            steps.append(f"Error Absoluto Máximo: ${error_metrics['max_abs_error']:.6e}$")
            steps.append(f"Error Relativo Máximo: ${error_metrics['max_rel_error']:.6e}$")
            steps.append(f"Norma del Error Absoluto: ${error_metrics['abs_error_norm']:.6e}$")

        result = {
            'method': 'Eliminación Gaussiana (con pivoteo parcial)',
            'steps': steps,
            'solution': x.tolist() # Return solution as list
        }
        
        # Add error metrics to result if available
        if error_metrics:
            result['error_metrics'] = error_metrics
            result['exact_solution'] = exact_solution

        return result
    except Exception as e:
        error_msg = f"Error inesperado durante la eliminación gaussiana: {e}"
        steps.append(f"**{error_msg}**")
        return {'error': error_msg, 'steps': steps}

def lu_decomposition(A, b, exact_solution=None):
    """
    Solves Ax = b using LU decomposition (Doolittle method).
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        exact_solution: Optional exact solution vector for error comparison
    """
    steps = ["## Inicio: Descomposición LU (Doolittle)"]
    try:
        n = A.shape[0]
        if A.shape[1] != n:
            return {'error': 'La matriz A debe ser cuadrada.', 'steps': steps}

        L = np.zeros((n, n))
        U = np.zeros((n, n))
        A_float = A.astype(float)
        b_float = b.astype(float)

        steps.append("**Matriz A Original:**")
        steps.append(f"${format_matrix_sympy(A_float)}$")
        steps.append("\n**Calculando matrices L y U (Método Doolittle: L[i,i] = 1):**")

        for k in range(n):
            # Calculate U
            steps.append(f"\n**Calculando fila {k+1} de U:**")
            for j in range(k, n):
                sum_lu = np.dot(L[k, :k], U[:k, j])
                U[k, j] = A_float[k, j] - sum_lu
                
                # Simple direct LaTeX formatting
                if k > 0:
                    # Create simple sum terms
                    sum_parts = []
                    for p in range(k):
                        sum_parts.append(f"L_{{{k+1},{p+1}}} \\cdot U_{{{p+1},{j+1}}}")
                    sum_expr = " + ".join(sum_parts)
                    latex = f"U_{{{k+1},{j+1}}} = A_{{{k+1},{j+1}}} - ({sum_expr}) = {U[k, j]:.4f}"
                else:
                    latex = f"U_{{{k+1},{j+1}}} = A_{{{k+1},{j+1}}} = {U[k, j]:.4f}"
                
                steps.append(f"- Calculando U_{{{k+1},{j+1}}}: ${latex}$")

            # Check for zero pivot in U
            if np.isclose(U[k, k], 0):
                error_msg = f"Error: Pivote U[{k+1},{k+1}] cercano a cero ({U[k, k]:.2e}). La matriz puede ser singular o requerir pivoteo (no implementado en esta versión de LU)."
                steps.append(f"**{error_msg}**")
                steps.append("\n**Matrices L y U Parciales:**")
                steps.append(f"L = ${format_matrix_sympy(L)}$")
                steps.append(f"U = ${format_matrix_sympy(U)}$")
                return {'error': error_msg, 'steps': steps, 'L_matrix': L.tolist(), 'U_matrix': U.tolist()}

            # Calculate L
            steps.append(f"\n**Calculando columna {k+1} de L:**")
            
            # Set diagonal to 1 (Doolittle)
            L[k, k] = 1.0
            steps.append(f"- Calculando L_{{{k+1},{k+1}}}: $L_{{{k+1},{k+1}}} = 1.0$ (Doolittle)")
            
            for i in range(k + 1, n):
                sum_lu = np.dot(L[i, :k], U[:k, k])
                L[i, k] = (A_float[i, k] - sum_lu) / U[k, k]
                
                # Simple direct LaTeX formatting
                if k > 0:
                    # Create simple sum terms
                    sum_parts = []
                    for p in range(k):
                        sum_parts.append(f"L_{{{i+1},{p+1}}} \\cdot U_{{{p+1},{k+1}}}")
                    sum_expr = " + ".join(sum_parts)
                    latex = f"L_{{{i+1},{k+1}}} = \\frac{{A_{{{i+1},{k+1}}} - ({sum_expr})}}{{U_{{{k+1},{k+1}}}}} = {L[i, k]:.4f}"
                else:
                    latex = f"L_{{{i+1},{k+1}}} = \\frac{{A_{{{i+1},{k+1}}}}}{{U_{{{k+1},{k+1}}}}} = {L[i, k]:.4f}"
                
                steps.append(f"- Calculando L_{{{i+1},{k+1}}}: ${latex}$")

        steps.append("\n**Descomposición LU Completada:**")
        steps.append(f"L = ${format_matrix_sympy(L)}$")
        steps.append(f"U = ${format_matrix_sympy(U)}$")

        # --- Solve Ly = b (Forward Substitution) --- #
        y = np.zeros(n)
        steps.append("\n**Resolviendo Ly = b (Sustitución Progresiva):**")
        
        for i in range(n):
            sum_ly = np.dot(L[i, :i], y[:i])
            y[i] = (b_float[i] - sum_ly) / L[i, i]
            
            # Simple direct LaTeX formatting
            if i > 0:
                # Create simple sum terms
                sum_parts = []
                for j in range(i):
                    sum_parts.append(f"L_{{{i+1},{j+1}}} \\cdot y_{{{j+1}}}")
                sum_expr = " + ".join(sum_parts)
                latex = f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}} - ({sum_expr})}}{{L_{{{i+1},{i+1}}}}} = {y[i]:.6f}"
            else:
                latex = f"y_{{{i+1}}} = \\frac{{b_{{{i+1}}}}}{{L_{{{i+1},{i+1}}}}} = {y[i]:.6f}"
            
            steps.append(f"- Calculando y_{{{i+1}}}: ${latex}$")
        
        steps.append(f"\nVector intermedio $y = {format_matrix_sympy(y.reshape(-1,1))}$")

        # --- Solve Ux = y (Backward Substitution) --- #
        x = np.zeros(n)
        steps.append("\n**Resolviendo Ux = y (Sustitución Regresiva):**")
        
        for i in range(n - 1, -1, -1):
            sum_ux = np.dot(U[i, i + 1:], x[i + 1:])
            x[i] = (y[i] - sum_ux) / U[i, i]
            
            # Instead of using a complex sum notation with commas that cause KaTeX parsing errors
            if len(U[i, i + 1:n]) > 0:
                # For brevity and clarity, just show specific calculation without using the problematic sum notation
                step_latex = f" $ x_{{{i + 1}}} = \\frac{{ b'_{{{i+1}}} - \\text{{términos conocidos}} }}{{ U_{{{i+1}{i+1}}} }} = \\frac{{ {y[i]:.4f} - ({sum_ux:.4f}) }}{{ {U[i, i]:.4f} }} = {x[i]:.6f} $ "
            else:
                # No sum needed for the last variable
                step_latex = f" $ x_{{{i + 1}}} = \\frac{{ b'_{{{i+1}}} }}{{ U_{{{i+1}{i+1}}} }} = \\frac{{ {y[i]:.4f} }}{{ {U[i, i]:.4f} }} = {x[i]:.6f} $ "
            
            steps.append(f"- Calculando x_{{{i + 1}}}: {step_latex}")

        steps.append("\n**Solución Final:**")
        steps.append(f"$x = {format_matrix_sympy(x.reshape(-1,1))}$")

        # Calculate error metrics if exact solution is provided
        error_metrics = None
        if exact_solution is not None:
            error_metrics = calculate_error_metrics(exact_solution, x)
            steps.append("\n**Comparación con Solución Exacta:**")
            steps.append(f"Solución Exacta: $x_{{exact}} = {format_matrix_sympy(np.array(exact_solution).reshape(-1,1))}$")
            steps.append(f"Error Absoluto Máximo: ${error_metrics['max_abs_error']:.6e}$")
            steps.append(f"Error Relativo Máximo: ${error_metrics['max_rel_error']:.6e}$")
            steps.append(f"Norma del Error Absoluto: ${error_metrics['abs_error_norm']:.6e}$")
        
        result = {
            'method': 'Descomposición LU (Doolittle)',
            'steps': steps,
            'solution': x.tolist(),
            'L_matrix': L.tolist(),
            'U_matrix': U.tolist()
        }
        
        # Add error metrics to result if available
        if error_metrics:
            result['error_metrics'] = error_metrics
            result['exact_solution'] = exact_solution
        
        return result

    except Exception as e:
        error_msg = f"Error inesperado durante la descomposición LU: {e}"
        steps.append(f"**{error_msg}**")
        return {'error': error_msg, 'steps': steps}

def jacobi_method(A, b, x0, tol, max_iter, exact_solution=None):
    """
    Solves Ax = b using the Jacobi iterative method.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess for solution
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        exact_solution: Optional exact solution vector for error comparison
    """
    steps = ["## Inicio: Método Iterativo de Jacobi"]
    try:
        n = len(b)
        A_float = A.astype(float)
        b_float = b.astype(float)
        x = x0.copy()
        
        # Check if matrix is diagonally dominant (optional improvement)
        is_diag_dom = True
        for i in range(n):
            diagonal = abs(A_float[i, i])
            sum_row = sum(abs(A_float[i, j]) for j in range(n) if j != i)
            if diagonal <= sum_row:
                is_diag_dom = False
                break
                
        if not is_diag_dom:
            steps.append("**Advertencia:** La matriz no es estrictamente diagonalmente dominante. El método de Jacobi podría no converger.")
        
        steps.append("**Matriz A Original:**")
        steps.append(f"${format_matrix_sympy(A_float)}$")
        steps.append("\n**Vector b:**")
        steps.append(f"${format_matrix_sympy(b_float.reshape(-1, 1))}$")
        steps.append("\n**Vector inicial x₀:**")
        steps.append(f"${format_matrix_sympy(x0.reshape(-1, 1))}$")
        
        # Check for zeros on the diagonal
        for i in range(n):
            if np.isclose(A_float[i, i], 0):
                error_msg = f"Error: Elemento diagonal A[{i+1},{i+1}] es cercano a cero. El método de Jacobi no puede continuar."
                steps.append(f"**{error_msg}**")
                return {'error': error_msg, 'steps': steps}
        
        # Initialize iteration history
        iterations = []
        error_history = []
        k = 0
        error = tol + 1  # Initialize error to be greater than tolerance
        
        steps.append("\n**Fórmula de iteración para cada componente:**")
        for i in range(n):
            # Create LaTeX for equation of i-th component
            sum_parts = []
            for j in range(n):
                if j != i:
                    sum_parts.append(f"\\frac{{A_{{{i+1},{j+1}}} \\cdot x^{{(k)}}_{{{j+1}}}}}{{A_{{{i+1},{i+1}}}}}")
            
            sum_expr = " - ".join(sum_parts)
            formula = f"x_{{{i+1}}}^{{(k+1)}} = \\frac{{b_{{{i+1}}}}}{{A_{{{i+1},{i+1}}}}} - ({sum_expr})"
            steps.append(f"- ${formula}$")
        
        steps.append("\n**Proceso Iterativo:**")
        
        # Iteration loop
        while error > tol and k < max_iter:
            x_new = np.zeros_like(x)
            
            # Add iteration header
            steps.append(f"\n**Iteración {k+1}:**")
            steps.append(f"- Vector actual: $x^{{({k})}} = {format_matrix_sympy(x.reshape(-1, 1))}$")
            
            # Calculate new x components
            for i in range(n):
                # Calculate sum for non-diagonal terms
                suma = 0
                sum_parts = []
                for j in range(n):
                    if j != i:
                        term = A_float[i, j] * x[j]
                        suma += term
                        sum_parts.append(f"A_{{{i+1},{j+1}}} \\cdot x_{{{j+1}}}^{{({k})}} = {A_float[i, j]:.4f} \\cdot {x[j]:.4f} = {term:.4f}")
                
                # Calculate new x[i]
                x_new[i] = (b_float[i] - suma) / A_float[i, i]
                
                # Compute step details
                sum_expr = " + ".join(sum_parts)
                calc_latex = f"x_{{{i+1}}}^{{({k+1})}} = \\frac{{b_{{{i+1}}} - ({sum_expr})}}{{A_{{{i+1},{i+1}}}}} = \\frac{{{b_float[i]:.4f} - ({suma:.4f})}}{{{A_float[i, i]:.4f}}} = {x_new[i]:.6f}"
                steps.append(f"- Calculando $x_{{{i+1}}}^{{({k+1})}}$: ${calc_latex}$")
            
            # Compute error (using max norm by default)
            if k > 0:  # Skip first iteration error calculation
                error = np.max(np.abs(x_new - x))
                error_history.append(error)
                steps.append(f"- Error: $\\|x^{{({k+1})}} - x^{{({k})}}\\|_\\infty = {error:.6e}$")
            
            # Save iteration data
            iterations.append(x_new.tolist())
            
            # Update x for next iteration
            x = x_new.copy()
            k += 1
            
            # Check if error is below tolerance
            if error <= tol:
                steps.append(f"\n**Convergencia alcanzada en {k} iteraciones** (error = {error:.6e} ≤ {tol:.6e})")
            elif k >= max_iter:
                steps.append(f"\n**Máximo de iteraciones alcanzado ({max_iter}) sin convergencia** (error final = {error:.6e})")
        
        # Create summary table
        steps.append("\n**Tabla de Iteraciones:**")
        table_rows = []
        header_row = ["k"] + [f"x_{i+1}" for i in range(n)] + ["Error"]
        table_rows.append(" | ".join(header_row))
        table_rows.append(" | ".join(["---"] * (n + 2)))
        
        # Initial row (k=0)
        initial_row = ["0"] + [f"{x0[i]:.6f}" for i in range(n)] + ["---"]
        table_rows.append(" | ".join(initial_row))
        
        # Iteration rows
        for i in range(min(k, max_iter)):
            error_val = "---" if i == 0 else f"{error_history[i-1]:.6e}"
            row = [str(i+1)] + [f"{iterations[i][j]:.6f}" for j in range(n)] + [error_val]
            table_rows.append(" | ".join(row))
        
        steps.append("\n" + "\n".join(table_rows))
        
        # Create data structure for Tabulator - ensure all values are JSON serializable
        iterations_table_data = []
        
        try:
            # Add initial row (k=0)
            initial_data = {}  # Usar un diccionario sin tipo explícito
            initial_data["k"] = 0  # Añadir la clave k que faltaba
            for j in range(n):
                # Convert numpy values to Python native types
                initial_data[f"x_{j+1}"] = float(x0[j])
            initial_data["error"] = None
            iterations_table_data.append(initial_data)
            
            # Add iteration rows
            for i in range(min(k, max_iter)):
                row_data = {}  # Usar un diccionario sin tipo explícito
                row_data["k"] = i+1  # Añadir la clave k que faltaba
                for j in range(n):
                    # Convert numpy values to Python native types
                    row_data[f"x_{j+1}"] = float(iterations[i][j])
                # Use None for first iteration error
                row_data["error"] = None if i == 0 else float(error_history[i-1])
                iterations_table_data.append(row_data)
        except Exception as e:
            # If there's any error in preparing table data, log it but continue
            print(f"Error preparing table data: {e}")
            iterations_table_data = []  # Use empty list as fallback
        
        steps.append("\n**Solución Final:**")
        steps.append(f"$x = {format_matrix_sympy(x.reshape(-1,1))}$")
        
        # Prepare data for plotting error vs iteration
        error_plot_data = {
            'iterations': list(range(1, len(error_history) + 1)),
            'errors': error_history
        }
        
        # Convertir iterations_table_data a JSON para evitar problemas de serialización
        iterations_table_json = "[]"
        try:
            # Asegurarse de que todos los valores sean serializables
            clean_table_data = []
            for row in iterations_table_data:
                clean_row = {}
                for key, value in row.items():
                    if key == "error" and value is None:
                        clean_row[key] = None
                    elif isinstance(value, np.integer):
                        clean_row[key] = int(value)
                    elif isinstance(value, np.floating):
                        clean_row[key] = float(value)
                    else:
                        clean_row[key] = value
                clean_table_data.append(clean_row)
            
            iterations_table_json = json.dumps(clean_table_data)
        except Exception as e:
            print(f"Error serializando datos de tabla: {e}")
        
        # Create the result object
        result = {
            'method': 'Método de Jacobi',
            'steps': steps,
            'solution': x.tolist(),
            'convergence': 'converged' if error <= tol else 'diverged',
            'iterations': iterations,
            'error_history': error_history,
            'error_plot_data': error_plot_data,
            'iterations_table_data': iterations_table_json  # Enviar la tabla como JSON string
        }
        
        # Calculate error metrics if exact solution is provided
        error_metrics = None
        if exact_solution is not None and error <= tol:
            error_metrics = calculate_error_metrics(exact_solution, x)
            steps.append("\n**Comparación con Solución Exacta:**")
            steps.append(f"Solución Exacta: $x_{{exact}} = {format_matrix_sympy(np.array(exact_solution).reshape(-1,1))}$")
            steps.append(f"Error Absoluto Máximo: ${error_metrics['max_abs_error']:.6e}$")
            steps.append(f"Error Relativo Máximo: ${error_metrics['max_rel_error']:.6e}$")
            steps.append(f"Norma del Error Absoluto: ${error_metrics['abs_error_norm']:.6e}$")
        
        # Add error metrics to result if available
        if error_metrics:
            result['error_metrics'] = error_metrics
            result['exact_solution'] = exact_solution
        
        return result
    except Exception as e:
        error_msg = f"Error inesperado durante el método de Jacobi: {e}"
        steps.append(f"**{error_msg}**")
        return {'error': error_msg, 'steps': steps}

def gauss_seidel_method(A, b, x0, tol, max_iter, exact_solution=None):
    """
    Solves Ax = b using the Gauss-Seidel iterative method.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess for solution
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        exact_solution: Optional exact solution vector for error comparison
    """
    steps = ["## Inicio: Método Iterativo de Gauss-Seidel"]
    try:
        n = len(b)
        A_float = A.astype(float)
        b_float = b.astype(float)
        x = x0.copy()
        
        # Check if matrix is diagonally dominant (optional improvement)
        is_diag_dom = True
        for i in range(n):
            diagonal = abs(A_float[i, i])
            sum_row = sum(abs(A_float[i, j]) for j in range(n) if j != i)
            if diagonal <= sum_row:
                is_diag_dom = False
                break
                
        if not is_diag_dom:
            steps.append("**Advertencia:** La matriz no es estrictamente diagonalmente dominante. El método de Gauss-Seidel podría no converger.")
        
        steps.append("**Matriz A Original:**")
        steps.append(f"${format_matrix_sympy(A_float)}$")
        steps.append("\n**Vector b:**")
        steps.append(f"${format_matrix_sympy(b_float.reshape(-1, 1))}$")
        steps.append("\n**Vector inicial x₀:**")
        steps.append(f"${format_matrix_sympy(x0.reshape(-1, 1))}$")
        
        # Check for zeros on the diagonal
        for i in range(n):
            if np.isclose(A_float[i, i], 0):
                error_msg = f"Error: Elemento diagonal A[{i+1},{i+1}] es cercano a cero. El método de Gauss-Seidel no puede continuar."
                steps.append(f"**{error_msg}**")
                return {'error': error_msg, 'steps': steps}
        
        # Initialize iteration history
        iterations = []
        error_history = []
        k = 0
        error = tol + 1  # Initialize error to be greater than tolerance
        
        steps.append("\n**Fórmula de iteración para cada componente:**")
        for i in range(n):
            # Create LaTeX for equation of i-th component 
            # (Gauss-Seidel uses updated values for components already calculated in this iteration)
            sum_parts_prev = []
            sum_parts_current = []
            
            for j in range(n):
                if j < i:  # Use updated values (x^(k+1))
                    sum_parts_current.append(f"\\frac{{A_{{{i+1},{j+1}}} \\cdot x^{{(k+1)}}_{{{j+1}}}}}{{A_{{{i+1},{i+1}}}}}")
                elif j > i:  # Use previous iteration values (x^(k))
                    sum_parts_prev.append(f"\\frac{{A_{{{i+1},{j+1}}} \\cdot x^{{(k)}}_{{{j+1}}}}}{{A_{{{i+1},{i+1}}}}}")
            
            formula_parts = []
            formula_parts.append(f"\\frac{{b_{{{i+1}}}}}{{A_{{{i+1},{i+1}}}}}")
            
            if sum_parts_current:
                formula_parts.append(f"- ({' + '.join(sum_parts_current)})")
                
            if sum_parts_prev:
                formula_parts.append(f"- ({' + '.join(sum_parts_prev)})")
                
            formula = f"x_{{{i+1}}}^{{(k+1)}} = {' '.join(formula_parts)}"
            steps.append(f"- ${formula}$")
        
        steps.append("\n**Proceso Iterativo:**")
        
        # Iteration loop
        while error > tol and k < max_iter:
            x_new = x.copy()  # Important: start with current values, will update in-place
            
            # Add iteration header
            steps.append(f"\n**Iteración {k+1}:**")
            steps.append(f"- Vector al inicio de la iteración: $x^{{({k})}} = {format_matrix_sympy(x.reshape(-1, 1))}$")
            
            # Calculate new x components
            for i in range(n):
                # Calculate sum for terms using already updated values (j < i)
                suma_current = 0
                sum_parts_current = []
                for j in range(i):
                    term = A_float[i, j] * x_new[j]
                    suma_current += term
                    sum_parts_current.append(f"A_{{{i+1},{j+1}}} \\cdot x_{{{j+1}}}^{{({k+1})}} = {A_float[i, j]:.4f} \\cdot {x_new[j]:.4f} = {term:.4f}")
                
                # Calculate sum for terms using previous iteration values (j > i)
                suma_prev = 0
                sum_parts_prev = []
                for j in range(i+1, n):
                    term = A_float[i, j] * x[j]
                    suma_prev += term
                    sum_parts_prev.append(f"A_{{{i+1},{j+1}}} \\cdot x_{{{j+1}}}^{{({k})}} = {A_float[i, j]:.4f} \\cdot {x[j]:.4f} = {term:.4f}")
                
                # Calculate new x[i] (Gauss-Seidel updates in-place)
                previous_value = x_new[i]
                x_new[i] = (b_float[i] - suma_current - suma_prev) / A_float[i, i]
                
                # Build detailed step explanation
                calc_latex = f"x_{{{i+1}}}^{{({k+1})}} = \\frac{{{b_float[i]:.4f}"
                
                if sum_parts_current:
                    calc_latex += f" - ({suma_current:.4f})"
                
                if sum_parts_prev:
                    calc_latex += f" - ({suma_prev:.4f})"
                
                calc_latex += f"}}{{{A_float[i, i]:.4f}}} = {x_new[i]:.6f}"
                
                # Add component explanations
                current_text = ""
                if sum_parts_current:
                    current_text = " - (" + " + ".join(sum_parts_current) + ")"
                    
                prev_text = ""
                if sum_parts_prev:
                    prev_text = " - (" + " + ".join(sum_parts_prev) + ")"
                
                full_latex = f"x_{{{i+1}}}^{{({k+1})}} = \\frac{{b_{{{i+1}}}{current_text}{prev_text}}}{{A_{{{i+1},{i+1}}}}} = {calc_latex}"
                
                steps.append(f"- Calculando $x_{{{i+1}}}^{{({k+1})}}$: ${full_latex}$")
            
            # After updating all components, compute error
            error = np.max(np.abs(x_new - x))
            # Ensure error is a native Python float, not numpy.float
            error_history.append(float(error))
            steps.append(f"- Error: $\\|x^{{({k+1})}} - x^{{({k})}}\\|_\\infty = {error:.6e}$")
            
            # Save iteration data
            iterations.append(x_new.tolist())
            
            # Update x for next iteration
            x = x_new.copy()
            k += 1
            
            # Check if error is below tolerance
            if error <= tol:
                steps.append(f"\n**Convergencia alcanzada en {k} iteraciones** (error = {error:.6e} ≤ {tol:.6e})")
            elif k >= max_iter:
                steps.append(f"\n**Máximo de iteraciones alcanzado ({max_iter}) sin convergencia** (error final = {error:.6e})")
        
        # Create summary table
        steps.append("\n**Tabla de Iteraciones:**")
        table_rows = []
        header_row = ["k"] + [f"x_{i+1}" for i in range(n)] + ["Error"]
        table_rows.append(" | ".join(header_row))
        table_rows.append(" | ".join(["---"] * (n + 2)))
        
        # Initial row (k=0)
        initial_row = ["0"] + [f"{x0[i]:.6f}" for i in range(n)] + ["---"]
        table_rows.append(" | ".join(initial_row))
        
        # Iteration rows
        for i in range(min(k, max_iter)):
            error_val = f"{error_history[i]:.6e}"
            row = [str(i+1)] + [f"{iterations[i][j]:.6f}" for j in range(n)] + [error_val]
            table_rows.append(" | ".join(row))
        
        steps.append("\n" + "\n".join(table_rows))
        
        # Create data structure for Tabulator - ensure all values are JSON serializable
        iterations_table_data = []
        
        try:
            # Add initial row (k=0)
            initial_data = {}  # Usar un diccionario sin tipo explícito
            initial_data["k"] = 0  # Añadir la clave k que faltaba
            for j in range(n):
                # Convert numpy values to Python native types
                initial_data[f"x_{j+1}"] = float(x0[j])
            initial_data["error"] = None
            iterations_table_data.append(initial_data)
            
            # Add iteration rows
            for i in range(min(k, max_iter)):
                row_data = {}  # Usar un diccionario sin tipo explícito
                row_data["k"] = i+1  # Añadir la clave k que faltaba
                for j in range(n):
                    # Convert numpy values to Python native types
                    row_data[f"x_{j+1}"] = float(iterations[i][j])
                # For Gauss-Seidel we have error for each iteration
                row_data["error"] = float(error_history[i])
                iterations_table_data.append(row_data)
        except Exception as e:
            # If there's any error in preparing table data, log it but continue
            print(f"Error preparing table data: {e}")
            iterations_table_data = []  # Use empty list as fallback
        
        steps.append("\n**Solución Final:**")
        steps.append(f"$x = {format_matrix_sympy(x.reshape(-1,1))}$")
        
        # Prepare data for plotting error vs iteration
        error_plot_data = {
            'iterations': list(range(1, len(error_history) + 1)),
            'errors': error_history
        }
        
        # Convertir iterations_table_data a JSON para evitar problemas de serialización
        iterations_table_json = "[]"
        try:
            # Asegurarse de que todos los valores sean serializables
            clean_table_data = []
            for row in iterations_table_data:
                clean_row = {}
                for key, value in row.items():
                    if key == "error" and value is None:
                        clean_row[key] = None
                    elif isinstance(value, np.integer):
                        clean_row[key] = int(value)
                    elif isinstance(value, np.floating):
                        clean_row[key] = float(value)
                    else:
                        clean_row[key] = value
                clean_table_data.append(clean_row)
            
            iterations_table_json = json.dumps(clean_table_data)
        except Exception as e:
            print(f"Error serializando datos de tabla: {e}")
        
        # Create the result object
        result = {
            'method': 'Método de Gauss-Seidel',
            'steps': steps,
            'solution': x.tolist(),
            'convergence': 'converged' if error <= tol else 'diverged',
            'iterations': iterations,
            'error_history': error_history,
            'error_plot_data': error_plot_data,
            'iterations_table_data': iterations_table_json  # Enviar la tabla como JSON string
        }
        
        # Calculate error metrics if exact solution is provided
        error_metrics = None
        if exact_solution is not None and error <= tol:
            error_metrics = calculate_error_metrics(exact_solution, x)
            steps.append("\n**Comparación con Solución Exacta:**")
            steps.append(f"Solución Exacta: $x_{{exact}} = {format_matrix_sympy(np.array(exact_solution).reshape(-1,1))}$")
            steps.append(f"Error Absoluto Máximo: ${error_metrics['max_abs_error']:.6e}$")
            steps.append(f"Error Relativo Máximo: ${error_metrics['max_rel_error']:.6e}$")
            steps.append(f"Norma del Error Absoluto: ${error_metrics['abs_error_norm']:.6e}$")
        
        # Add error metrics to result if available
        if error_metrics:
            result['error_metrics'] = error_metrics
            result['exact_solution'] = exact_solution
        
        return result
    except Exception as e:
        error_msg = f"Error inesperado durante el método de Gauss-Seidel: {e}"
        steps.append(f"**{error_msg}**")
        return {'error': error_msg, 'steps': steps}

def calculate_error_metrics(exact_solution, approx_solution):
    """
    Calculate error metrics between exact and approximate solutions.
    
    Args:
        exact_solution: Numpy array of the exact solution values
        approx_solution: Numpy array of the approximate solution values
        
    Returns:
        Dict with error metrics: absolute error, relative error, etc.
    """
    exact = np.array(exact_solution, dtype=float)
    approx = np.array(approx_solution, dtype=float)
    
    # Calculate error metrics
    absolute_error = np.abs(exact - approx)
    relative_error = np.zeros_like(absolute_error)
    # Avoid division by zero for relative error
    nonzero_mask = ~np.isclose(exact, 0)
    relative_error[nonzero_mask] = absolute_error[nonzero_mask] / np.abs(exact[nonzero_mask])
    
    # Calculate overall error norms
    abs_error_norm = np.linalg.norm(absolute_error)
    rel_error_norm = np.linalg.norm(relative_error)
    
    return {
        'absolute_error': absolute_error.tolist(),
        'relative_error': relative_error.tolist(),
        'abs_error_norm': float(abs_error_norm),
        'rel_error_norm': float(rel_error_norm),
        'max_abs_error': float(np.max(absolute_error)),
        'max_rel_error': float(np.max(relative_error))
    }

def solve_system(A, b, method="gauss", tol=1e-10, max_iter=100, initial_guess=None):
    """
    Función unificada para resolver sistemas lineales con diferentes métodos.
    
    Args:
        A: Matriz de coeficientes
        b: Vector del lado derecho
        method: Método a utilizar ("gauss", "lu", "jacobi", "gauss_seidel")
        tol: Tolerancia para métodos iterativos
        max_iter: Máximo de iteraciones para métodos iterativos
        initial_guess: Vector inicial para métodos iterativos
        
    Returns:
        dict: Diccionario con los resultados del método
    """
    # Asegurar que A y b sean arrays de NumPy
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Seleccionar el método adecuado
    if method == "gauss":
        return gaussian_elimination(A, b)
    elif method == "lu":
        return lu_decomposition(A, b)
    elif method == "jacobi":
        if initial_guess is None:
            initial_guess = np.zeros_like(b)
        return jacobi_method(A, b, initial_guess, tol, max_iter)
    elif method == "gauss_seidel":
        if initial_guess is None:
            initial_guess = np.zeros_like(b)
        return gauss_seidel_method(A, b, initial_guess, tol, max_iter)
    else:
        raise ValueError(f"Método '{method}' no implementado. Opciones válidas: gauss, lu, jacobi, gauss_seidel")

# Example Usage (for testing)
if __name__ == '__main__':
    A_test = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b_test = np.array([8, -11, -3])
    A_lu_test = np.array([[1, 4, -3], [-2, 8, 5], [3, 4, 7]])
    b_lu_test = np.array([1, -2, 6])

    print("--- Gaussian Elimination (Test 1) ---")
    result = gaussian_elimination(A_test, b_test)
    if 'error' in result:
        print(f"Error: {result['error']}")
        for step in result['steps']:
             print(step)
    else:
        # for step in result['steps']:
        #     print(step)
        print(f"Solution: {result['solution']}")

    print("\n--- LU Decomposition (Test 1) ---")
    result_lu = lu_decomposition(A_lu_test, b_lu_test)
    if 'error' in result_lu:
        print(f"Error: {result_lu['error']}")
        # for step in result_lu['steps']:
        #      print(step)
    else:
         print(f"Solution: {result_lu['solution']}")
         print(f"L:\n{np.array(result_lu['L_matrix'])}")
         print(f"U:\n{np.array(result_lu['U_matrix'])}")
         print(f"y: {result_lu['y_vector']}")

    print("\n--- LU Decomposition (Singular Example - should fail) ---")
    A_lu_singular = np.array([[1, 1], [1, 1]])
    b_lu_singular = np.array([2, 3])
    result_lu_s = lu_decomposition(A_lu_singular, b_lu_singular)
    if 'error' in result_lu_s:
        print(f"Error: {result_lu_s['error']}")
    else:
         print(f"Solution: {result_lu_s['solution']}") 

    # Test for Jacobi method
    print("\n--- Jacobi Method Test ---")
    # 3x3 diagonally dominant system
    A_jacobi = np.array([[5, 2, 1], [1, 6, 3], [2, 1, 8]])
    b_jacobi = np.array([8, 10, 11])
    x0_jacobi = np.zeros(3)
    result_jacobi = jacobi_method(A_jacobi, b_jacobi, x0_jacobi, 1e-4, 20)
    if 'error' in result_jacobi:
        print(f"Error: {result_jacobi['error']}")
    else:
        print(f"Solution after {result_jacobi['iterations_count']} iterations: {result_jacobi['solution']}")
        print(f"Final error: {result_jacobi['errors'][-1] if result_jacobi['errors'] else 'N/A'}")

import matplotlib.pyplot as plt
import os
import uuid

def plot_error_convergence(errors, iterations, title="Convergencia del Error", uuid_str=None):
    """Create a plot showing error convergence for iterative methods.
    
    Args:
        errors: List of error values
        iterations: List of iteration numbers
        title: Plot title
        uuid_str: Unique identifier for the filename (if None, one will be generated)
        
    Returns:
        Path to the saved plot image
    """
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, errors, 'bo-', linewidth=1.5)
    plt.yscale('log')  # Logarithmic scale for error
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Número de Iteración')
    plt.ylabel('Error (escala logarítmica)')
    plt.title(title)
    
    # Ensure static/img directory exists
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'img')
    os.makedirs(static_dir, exist_ok=True)
    
    # Generate filename with UUID if not provided
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())
    
    # Save figure
    filename = f'error_convergence_{uuid_str}.png'
    filepath = os.path.join(static_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Return path relative to static directory
    return os.path.join('img', filename) 