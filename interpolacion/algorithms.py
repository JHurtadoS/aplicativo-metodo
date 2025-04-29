import numpy as np
import sympy
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for web servers
import matplotlib.pyplot as plt
import os
import uuid
from pathlib import Path

# Ensure the static/img directory exists
# Adjust this path if your static files are configured differently in settings.py
BASE_DIR = Path(__file__).resolve().parent.parent # Assumes algorithms.py is inside the app folder
STATIC_ROOT_IMG = BASE_DIR / 'static' / 'img'
STATIC_URL_IMG_PREFIX = 'img/' # Relative path used in templates (linked to STATIC_URL)

STATIC_ROOT_IMG.mkdir(parents=True, exist_ok=True)

def generate_plot(x_coords, y_coords, result_func=None, x_dense=None, y_dense=None, title='Interpolación/Ajuste', xlabel='x', ylabel='y'):
    """Generates a plot and saves it, returning the relative URL path."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_coords, y_coords, color='red', label='Puntos Dados', zorder=5) # zorder keeps points on top

    legend_label = f'Resultado ({title})'

    if result_func is not None and x_dense is not None and y_dense is not None: # For interpolation
        ax.plot(x_dense, y_dense, label=legend_label)
    elif result_func is not None: # For linear regression or other functions y=f(x)
        xmin, xmax = np.min(x_coords), np.max(x_coords)
        padding = (xmax - xmin) * 0.1 # Add padding to the plot range
        x_plot = np.linspace(xmin - padding, xmax + padding, 200)
        y_plot = result_func(x_plot)
        ax.plot(x_plot, y_plot, label=legend_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    filename = f"{uuid.uuid4()}.png"
    full_path = STATIC_ROOT_IMG / filename
    try:
        plt.savefig(full_path, dpi=100, bbox_inches="tight")
    except Exception as e:
        print(f"Error saving plot: {e}") # Log error
        plt.close(fig) # Ensure figure is closed even if saving fails
        return None
    plt.close(fig) # Close the plot to free memory

    # Return the relative path for use in templates (e.g., 'img/uuid.png')
    # Assumes STATIC_URL is configured correctly in Django settings
    return os.path.join(STATIC_URL_IMG_PREFIX, filename).replace('\\', '/')

def lagrange_interpolation(points):
    """Calculates Lagrange interpolation."""
    steps = ["## Inicio: Interpolación de Lagrange"]
    try:
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        n = len(points)

        if n < 2:
            return {'error': 'Se necesitan al menos 2 puntos para la interpolación.'}
        if len(set(x_coords)) != n:
             return {'error': 'Los valores de x deben ser únicos para la interpolación de Lagrange.'}

        x = sympy.symbols('x')
        polynomial = sympy.sympify(0)

        steps.append(f"**Número de puntos:** {n}")
        steps.append(f"**Coordenadas x:** `{x_coords}`")
        steps.append(f"**Coordenadas y:** `{y_coords}`")
        steps.append("\n### Calculando Polinomios Base L_i(x)")

        basis_polynomials_latex = []
        for i in range(n):
            L_i = sympy.sympify(1)
            numerator_str = ""
            denominator_str = ""
            for j in range(n):
                if i != j:
                    term_num = (x - x_coords[j])
                    term_den = (x_coords[i] - x_coords[j])
                    if term_den == 0:
                         return {'error': f'División por cero detectada al calcular L_{i}(x). Verifique los puntos.'}
                    L_i *= term_num / term_den
                    numerator_str += f"({sympy.latex(term_num)})"
                    denominator_str += f"({sympy.latex(sympy.sympify(term_den))})"

            L_i_simplified = sympy.simplify(L_i)
            basis_polynomials_latex.append(f"L_{{{i}}}(x) = {sympy.latex(L_i_simplified)}")
            steps.append(f"**L_{{{i}}}(x)** = $\\frac{{{numerator_str}}}{{{denominator_str}}} = {sympy.latex(L_i_simplified)}$")
            polynomial += y_coords[i] * L_i

        polynomial_simplified = sympy.expand(polynomial)
        steps.append("\n### Polinomio Interpolante P(x)")
        steps.append(f"$P(x) = \\sum_{{i=0}}^{{{n-1}}} y_i L_i(x)$")
        steps.append(f"**Polinomio final (simplificado):** $P(x) = {sympy.latex(polynomial_simplified)}$")

        # Generate plot data
        plot_path = None
        try:
            f_lambdified = sympy.lambdify(x, polynomial_simplified, 'numpy')
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            range_pad = (x_max - x_min) * 0.1
            x_dense = np.linspace(x_min - range_pad, x_max + range_pad, 500)
            y_dense = f_lambdified(x_dense)
            plot_path = generate_plot(x_coords, y_coords, x_dense=x_dense, y_dense=y_dense, title='Interpolación de Lagrange')
        except Exception as e:
            steps.append(f"\n*Error al generar la gráfica: {e}*")

        result = {
            'method': 'Interpolación de Lagrange',
            'steps': steps,
            'polynomial_sympy': polynomial_simplified,
            'polynomial_latex': sympy.latex(polynomial_simplified),
            'plot_path': plot_path,
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
        }
        return result
    except Exception as e:
        return {'error': f'Error inesperado en Lagrange: {e}'}

def newton_interpolation(points):
    """Calculates Newton's divided difference interpolation."""
    steps = ["## Inicio: Interpolación de Newton (Diferencias Divididas)"]
    try:
        n = len(points)
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        if n < 2:
            return {'error': 'Se necesitan al menos 2 puntos.'}
        if len(set(x_coords)) != n:
             return {'error': 'Los valores de x deben ser únicos para la interpolación de Newton.'}

        steps.append(f"**Número de puntos:** {n}")
        steps.append(f"**Coordenadas x:** `{x_coords}`")
        steps.append(f"**Coordenadas y:** `{y_coords}`")

        # Initialize divided difference table
        divided_diff = np.zeros((n, n))
        divided_diff[:, 0] = y_coords

        steps.append("\n### Calculando la Tabla de Diferencias Divididas")
        table_steps = []
        # Calculate table
        for j in range(1, n):
            col_steps = [f"**Orden {j}:**"]
            for i in range(n - j):
                try:
                    numerator = divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]
                    denominator = x_coords[i + j] - x_coords[i]
                    if denominator == 0:
                         return {'error': f'División por cero al calcular f[x_{i}, ..., x_{i+j}]. Valores de x duplicados.'}
                    divided_diff[i, j] = numerator / denominator
                    col_steps.append(f"  $f[x_{{{i}}}, \dots, x_{{{i+j}}}] = \\frac{{f[x_{{{i+1}}}, \dots, x_{{{i+j}}}] - f[x_{{{i}}}, \dots, x_{{{i+j-1}}}]}}{{x_{{{i+j}}} - x_{i}}} = \\frac{{{numerator:.4f}}}{{{denominator:.4f}}} = {divided_diff[i, j]:.4f}$ ")
                except IndexError:
                     # Should not happen with correct loop bounds
                     return {'error': 'Error de índice calculando la tabla.'}
            table_steps.extend(col_steps)

        steps.extend(table_steps)

        # Format final table for display (using Markdown table)
        steps.append("\n### Tabla de Diferencias Divididas Final")
        header = ["i", "x_i", "f[x_i]"] + [f"Orden {j+1}" for j in range(n - 1)]
        table_md = " | ".join(header) + "\n" + " | ".join(["---"] * len(header))
        for i in range(n):
            row = [f"{i}", f"{x_coords[i]:.4f}"]
            row.extend([f"{divided_diff[i, j]:.4f}" if j < n - i else "" for j in range(n)])
            table_md += "\n" + " | ".join(row[1:]) # Start from x_i column
        steps.append(table_md)

        # Construct polynomial using SymPy
        x = sympy.symbols('x')
        polynomial = sympy.sympify(divided_diff[0, 0])
        term_product = sympy.sympify(1)
        steps.append("\n### Construyendo el Polinomio de Newton P(x)")
        steps.append(f"$P(x) = f[x_0] + f[x_0, x_1](x-x_0) + f[x_0, x_1, x_2](x-x_0)(x-x_1) + \dots$")
        steps.append(f"**Coeficientes (diagonal superior tabla):** `{[f'{c:.4f}' for c in divided_diff[0, :]]}`")
        steps.append(f"$P_0(x) = {divided_diff[0, 0]:.4f}$ ")

        current_poly_latex = sympy.latex(sympy.N(polynomial, 4))
        for j in range(1, n):
            factor = (x - x_coords[j - 1])
            term_product *= factor
            term = divided_diff[0, j] * term_product
            polynomial += term
            # Show polynomial growth step-by-step
            new_poly_latex = sympy.latex(sympy.N(polynomial, 4))
            steps.append(f"$P_{j}(x) = P_{{{j-1}}}(x) + {divided_diff[0, j]:.4f} \times {sympy.latex(term_product)} = {new_poly_latex}$ ")
            current_poly_latex = new_poly_latex # Update for next step

        polynomial_simplified = sympy.expand(polynomial)
        steps.append(f"\n**Polinomio final (simplificado):** $P(x) = {sympy.latex(polynomial_simplified)}$")

        # Generate plot data
        plot_path = None
        try:
            f_lambdified = sympy.lambdify(x, polynomial_simplified, 'numpy')
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            range_pad = (x_max - x_min) * 0.1
            x_dense = np.linspace(x_min - range_pad, x_max + range_pad, 500)
            y_dense = f_lambdified(x_dense)
            plot_path = generate_plot(x_coords, y_coords, x_dense=x_dense, y_dense=y_dense, title='Interpolación de Newton')
        except Exception as e:
            steps.append(f"\n*Error al generar la gráfica: {e}*")

        result = {
            'method': 'Interpolación de Newton',
            'steps': steps,
            'polynomial_sympy': polynomial_simplified,
            'polynomial_latex': sympy.latex(polynomial_simplified),
            'plot_path': plot_path,
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'divided_difference_table': divided_diff.tolist()
        }
        return result
    except Exception as e:
        return {'error': f'Error inesperado en Newton: {e}'}

def linear_regression(points):
    """Calculates linear regression using least squares."""
    steps = ["## Inicio: Regresión Lineal (Mínimos Cuadrados)"]
    try:
        n = len(points)
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        if n < 2:
            return {'error': 'Se necesitan al menos 2 puntos para la regresión.'}

        steps.append(f"**Número de puntos:** {n}")
        steps.append(f"**Coordenadas x:** `{x_coords}`")
        steps.append(f"**Coordenadas y:** `{y_coords}`")

        # Calculate sums needed for normal equations
        sum_x = np.sum(x_coords)
        sum_y = np.sum(y_coords)
        sum_xy = np.sum(x_coords * y_coords)
        sum_x2 = np.sum(x_coords**2)

        steps.append("\n### Calculando Sumatorias Necesarias")
        steps.append(f"- $ \Sigma x = {sum_x:.4f} $ ")
        steps.append(f"- $ \Sigma y = {sum_y:.4f} $ ")
        steps.append(f"- $ \Sigma (x \cdot y) = {sum_xy:.4f} $ ")
        steps.append(f"- $ \Sigma (x^2) = {sum_x2:.4f} $ ")

        # Calculate slope (a) and intercept (b) for y = ax + b
        denominator = (n * sum_x2 - sum_x**2)
        if np.isclose(denominator, 0):
            error_msg = "División por cero al calcular coeficientes. Ocurre si todos los valores de x son iguales."
            steps.append(f"\n**Error:** {error_msg}")
            return {
                'method': 'Regresión Lineal',
                'error': error_msg,
                'steps': steps
            }

        a = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y - a * sum_x) / n

        steps.append("\n### Calculando Coeficientes (Pendiente 'a' e Intercepto 'b')")
        steps.append(f" $ a = \\frac{{n \Sigma xy - \Sigma x \Sigma y}}{{n \Sigma x^2 - (\Sigma x)^2}} = \\frac{{{n} \times {sum_xy:.4f} - {sum_x:.4f} \times {sum_y:.4f}}}{{{n} \times {sum_x2:.4f} - ({sum_x:.4f})^2}} = {a:.4f} $ ")
        steps.append(f" $ b = \\frac{{\Sigma y - a \Sigma x}}{{n}} = \\frac{{{sum_y:.4f} - {a:.4f} \times {sum_x:.4f}}}{{{n}}} = {b:.4f} $ ")

        # Create SymPy expression for the line
        x = sympy.symbols('x')
        # Use sympify to handle potential precision issues before N()
        line_eq = sympy.sympify(a) * x + sympy.sympify(b)
        line_eq_simplified = sympy.N(line_eq, 5) # Numerical evaluation with 5 significant digits

        steps.append(f"\n### Recta de Regresión $y = ax + b$ ")
        steps.append(f"**Ecuación final:** $y = {sympy.latex(line_eq_simplified)}$")

        # Calculate R-squared (coefficient of determination)
        y_mean = np.mean(y_coords)
        ss_tot = np.sum((y_coords - y_mean)**2)
        if np.isclose(ss_tot, 0): # Handle case where all y are the same
            r_squared = 1.0 # Perfect fit if all y are identical and on the line
        else:
            y_pred = a * x_coords + b
            ss_res = np.sum((y_coords - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot)

        steps.append(f"\n### Coeficiente de Determinación (R²)")
        steps.append(f"$ R^2 = 1 - \\frac{{\Sigma (y_i - \hat{{y}}_i)^2}}{{\Sigma (y_i - \bar{{y}})^2}} = 1 - \\frac{{{ss_res:.4f}}}{{{ss_tot:.4f}}} = {r_squared:.4f} $ ")
        if r_squared >= 0.9:
            interpretation = "Excelente ajuste (R² ≥ 0.9)"
        elif r_squared >= 0.7:
            interpretation = "Ajuste aceptable (0.7 ≤ R² < 0.9)"
        elif r_squared >= 0.5:
             interpretation = "Ajuste débil (0.5 ≤ R² < 0.7)"
        else:
            interpretation = "Ajuste muy débil o inexistente (R² < 0.5)"
        steps.append(f"**Interpretación:** {interpretation}")

        # Generate plot
        plot_path = None
        try:
            # Need to handle the case where lambdify might fail or produce non-numeric output
            f_lambdified = sympy.lambdify(x, line_eq_simplified, modules=['numpy', {'ImmutableMatrix': np.array}])

            # Test lambdified function with a single value first
            test_val = x_coords[0] if len(x_coords) > 0 else 0
            try:
                 _ = f_lambdified(test_val)
            except Exception as lambdify_eval_error:
                 steps.append(f"\n*Advertencia: No se pudo evaluar la función para graficar: {lambdify_eval_error}*")
                 f_lambdified = None # Prevent plotting if evaluation fails

            if f_lambdified:
                 plot_path = generate_plot(x_coords, y_coords, result_func=f_lambdified, title='Regresión Lineal')
            else:
                 plot_path = generate_plot(x_coords, y_coords, title='Regresión Lineal (solo puntos)')

        except Exception as e:
            steps.append(f"\n*Error al generar la gráfica: {e}*")

        result = {
            'method': 'Regresión Lineal',
            'steps': steps,
            'line_sympy': line_eq_simplified,
            'line_latex': sympy.latex(line_eq_simplified),
            'coefficients': {'a': a, 'b': b},
            'r_squared': r_squared,
            'plot_path': plot_path,
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
        }
        return result
    except Exception as e:
        return {'error': f'Error inesperado en Regresión Lineal: {e}'}

# Example usage (for testing)
if __name__ == '__main__':
    # Test points
    test_points_interp = [(1, 2), (3, 5), (4, 8), (0, 1)]
    test_points_regr = [(1, 1.5), (2, 3.8), (3, 6.7), (4, 9.1), (5, 11.2), (6, 13.6), (7, 16)]
    test_points_regr_vertical = [(2, 1), (2, 3), (2, 5)]
    test_points_regr_short = [(1,1)]

    print("--- Lagrange Interpolation ---")
    lagrange_res = lagrange_interpolation(test_points_interp)
    if 'error' in lagrange_res:
        print(f"Error: {lagrange_res['error']}")
    else:
        # for step in lagrange_res['steps']:
        #     print(step)
        print(f"Polynomial LaTeX: ${lagrange_res['polynomial_latex']}$ ")
        print(f"Plot saved to: {lagrange_res['plot_path']}")

    print("\n--- Newton Interpolation ---")
    newton_res = newton_interpolation(test_points_interp)
    if 'error' in newton_res:
        print(f"Error: {newton_res['error']}")
    else:
        # for step in newton_res['steps']:
        #     print(step)
        print(f"Polynomial LaTeX: ${newton_res['polynomial_latex']}$ ")
        print(f"Plot saved to: {newton_res['plot_path']}")

    print("\n--- Linear Regression --- (Good Fit)")
    regr_res = linear_regression(test_points_regr)
    if 'error' in regr_res:
        print(f"Error: {regr_res['error']}")
    else:
        # for step in regr_res['steps']:
        #     print(step)
        print(f"Line LaTeX: ${regr_res['line_latex']}$ ")
        print(f"R-squared: {regr_res['r_squared']:.4f}")
        print(f"Plot saved to: {regr_res['plot_path']}")

    print("\n--- Linear Regression --- (Vertical x)")
    regr_res_vert = linear_regression(test_points_regr_vertical)
    if 'error' in regr_res_vert:
        print(f"Error: {regr_res_vert['error']}")
    else:
        print(f"Line LaTeX: ${regr_res_vert['line_latex']}$ ")
        print(f"Plot saved to: {regr_res_vert['plot_path']}")

    print("\n--- Linear Regression --- (Too Short)")
    regr_res_short = linear_regression(test_points_regr_short)
    if 'error' in regr_res_short:
        print(f"Error: {regr_res_short['error']}")
    else:
        print(f"Line LaTeX: ${regr_res_short['line_latex']}$ ")
        print(f"Plot saved to: {regr_res_short['plot_path']}") 