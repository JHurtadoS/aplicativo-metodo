#  ProyectoMN – Licencia IMT
from typing import List, Tuple, Optional
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import uuid
import os
from dataclasses import dataclass
from .core import Result, InterpResult, plot_points_curve
from sistemas_lineales.algorithms import solve_system
from core.utils import fix_nested_matrices
from pathlib import Path
import re

def builder_splines_system(points: List[Tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye el sistema de ecuaciones para calcular los coeficientes de los splines cúbicos.
    
    Retorna:
        (A, b) donde A es la matriz de coeficientes y b es el vector de términos independientes.
    """
    if len(points) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para splines cúbicos.")
    
    # Ordenar puntos por coordenada x
    points = sorted(points, key=lambda p: p[0])
    
    x_coords_list = [p[0] for p in points]
    y_coords_list = [p[1] for p in points]
    
    # Verificar si hay x duplicados
    if len(x_coords_list) != len(set(x_coords_list)):
        raise ValueError("Los valores de x deben ser únicos.")
    
    n = len(points) - 1  # Número de intervalos
    
    # Para n intervalos, tenemos 4n incógnitas (a, b, c, d para cada intervalo)
    # Necesitamos 4n ecuaciones para resolver el sistema
    
    # Construir matriz del sistema y vector de términos independientes
    # Se usarán condiciones: 
    # 1) S_i(x_i) = y_i y S_i(x_{i+1}) = y_{i+1} (2n ecuaciones)
    # 2) S_i'(x_{i+1}) = S_{i+1}'(x_{i+1}) (n-1 ecuaciones)
    # 3) S_i''(x_{i+1}) = S_{i+1}''(x_{i+1}) (n-1 ecuaciones)
    # 4) S_0''(x_0) = 0 y S_{n-1}''(x_n) = 0 (2 ecuaciones - splines naturales)
    
    # Para cada spline S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
    
    # Número total de ecuaciones = 4n
    A = np.zeros((4*n, 4*n))
    b_vec = np.zeros(4*n)
    
    # Llenar las ecuaciones para cada intervalo
    eq_index = 0
    
    # Condiciones 1: S_i(x_i) = y_i y S_i(x_{i+1}) = y_{i+1}
    for i in range(n):
        h_i = x_coords_list[i+1] - x_coords_list[i]
        
        # S_i(x_i) = a_i = y_i
        A[eq_index, 4*i] = 1  # Coeficiente de a_i
        b_vec[eq_index] = y_coords_list[i]
        eq_index += 1
        
        # S_i(x_{i+1}) = a_i + b_i*h_i + c_i*h_i^2 + d_i*h_i^3 = y_{i+1}
        A[eq_index, 4*i] = 1       # Coeficiente de a_i
        A[eq_index, 4*i+1] = h_i   # Coeficiente de b_i
        A[eq_index, 4*i+2] = h_i**2  # Coeficiente de c_i
        A[eq_index, 4*i+3] = h_i**3  # Coeficiente de d_i
        b_vec[eq_index] = y_coords_list[i+1]
        eq_index += 1
    
    # Condiciones 2: S_i'(x_{i+1}) = S_{i+1}'(x_{i+1}) para i=0,...,n-2
    for i in range(n-1):
        h_i = x_coords_list[i+1] - x_coords_list[i]
        
        # S_i'(x_{i+1}) = b_i + 2*c_i*h_i + 3*d_i*h_i^2
        A[eq_index, 4*i+1] = 1        # Coeficiente de b_i
        A[eq_index, 4*i+2] = 2*h_i    # Coeficiente de c_i
        A[eq_index, 4*i+3] = 3*h_i**2 # Coeficiente de d_i
        
        # S_{i+1}'(x_{i+1}) = b_{i+1}
        A[eq_index, 4*(i+1)+1] = -1   # Coeficiente de b_{i+1}
        
        b_vec[eq_index] = 0  # Igualdad de derivadas
        eq_index += 1
    
    # Condiciones 3: S_i''(x_{i+1}) = S_{i+1}''(x_{i+1}) para i=0,...,n-2
    for i in range(n-1):
        h_i = x_coords_list[i+1] - x_coords_list[i]
        
        # S_i''(x_{i+1}) = 2*c_i + 6*d_i*h_i
        A[eq_index, 4*i+2] = 2        # Coeficiente de c_i
        A[eq_index, 4*i+3] = 6*h_i    # Coeficiente de d_i
        
        # S_{i+1}''(x_{i+1}) = 2*c_{i+1}
        A[eq_index, 4*(i+1)+2] = -2   # Coeficiente de c_{i+1}
        
        b_vec[eq_index] = 0  # Igualdad de segundas derivadas
        eq_index += 1
    
    # Condiciones 4: Splines naturales - segunda derivada cero en los extremos
    # S_0''(x_0) = 2*c_0 = 0
    A[eq_index, 2] = 2  # Coeficiente de c_0
    b_vec[eq_index] = 0
    eq_index += 1
    
    # S_{n-1}''(x_n) = 2*c_{n-1} + 6*d_{n-1}*h_{n-1} = 0
    h_last = x_coords_list[n] - x_coords_list[n-1]
    A[eq_index, 4*(n-1)+2] = 2        # Coeficiente de c_{n-1}
    A[eq_index, 4*(n-1)+3] = 6*h_last # Coeficiente de d_{n-1}
    b_vec[eq_index] = 0
    
    return A, b_vec

def natural_cubic_splines(points: List[Tuple[float, float]], solver: str = "gauss") -> InterpResult:
    """
    Calcula los splines cúbicos naturales que interpolan los puntos dados.
    
    Args:
        points: Lista de tuplas (x, y) que representan los puntos a interpolar.
        solver: Método para resolver el sistema lineal ("gauss" o "lu").
        
    Returns:
        InterpResult: Resultado de la interpolación con splines cúbicos.
    """
    steps = ["<div class='text-2xl font-bold text-blue-800 mt-6 mb-3'>Inicio: Interpolación con Splines Cúbicos Naturales</div>"]
    
    try:
        # Ordenar puntos por coordenada x
        points = sorted(points, key=lambda p: p[0])
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Validar entrada
        if len(points) < 3:
            raise ValueError("Se necesitan al menos 3 puntos para splines cúbicos.")
        
        # Verificar duplicados en x
        if len(x_coords) != len(set(x_coords)):
            raise ValueError("Los valores de x deben ser únicos para la interpolación.")
        
        steps.append(f"<div class='my-2'><span class='font-semibold'>Puntos de entrada (ordenados):</span> {points}</div>")
        steps.append("<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Construyendo el sistema de ecuaciones para splines cúbicos</div>")
        
        # Construir sistema de ecuaciones
        A_matrix, b_vector = builder_splines_system(points)
        
        steps.append(f"<div class='my-2'><span class='font-semibold'>Matriz del sistema para splines (forma resumida):</span></div>")
        steps.append(f"<pre class='bg-gray-50 p-3 rounded border border-gray-200 overflow-auto text-sm'>Forma: {A_matrix.shape}</pre>")
        
        # Resolver el sistema lineal
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Resolviendo el sistema lineal (Método: {solver})</div>")
        result_dict = solve_system(A_matrix, b_vector, solver)
        
        # Extraer los coeficientes de la solución
        if 'error' in result_dict:
            raise ValueError(f"Error al resolver el sistema: {result_dict['error']}")
            
        coefs_all = result_dict['solution'] if 'solution' in result_dict else []
        
        # n es el número de intervalos (número de puntos - 1)
        n = len(points) - 1
        
        # Organizar los coeficientes por spline e intervalo
        splines_coefs_list = []
        for i in range(n):
            spline_i_data = {
                'intervalo': (x_coords[i], x_coords[i+1]),
                'coeficientes': {
                    'a': coefs_all[4*i],
                    'b': coefs_all[4*i+1],
                    'c': coefs_all[4*i+2],
                    'd': coefs_all[4*i+3]
                }
            }
            splines_coefs_list.append(spline_i_data)
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Coeficientes de los Splines Cúbicos</div>")
        for i, spline_data in enumerate(splines_coefs_list):
            steps.append(f"<div class='mt-3 mb-1 font-semibold text-blue-600'>Spline {i+1} <span class='font-normal'>(Intervalo: [{spline_data['intervalo'][0]}, {spline_data['intervalo'][1]}])</span></div>")
            steps.append("<div class='ml-4 grid grid-cols-2 gap-2 my-1'>")
            steps.append(f"<div><span class='font-medium'>a =</span> <code class='bg-gray-100 px-1 rounded'>{spline_data['coeficientes']['a']:.6f}</code></div>")
            steps.append(f"<div><span class='font-medium'>b =</span> <code class='bg-gray-100 px-1 rounded'>{spline_data['coeficientes']['b']:.6f}</code></div>")
            steps.append(f"<div><span class='font-medium'>c =</span> <code class='bg-gray-100 px-1 rounded'>{spline_data['coeficientes']['c']:.6f}</code></div>")
            steps.append(f"<div><span class='font-medium'>d =</span> <code class='bg-gray-100 px-1 rounded'>{spline_data['coeficientes']['d']:.6f}</code></div>")
            steps.append("</div>")
        
        # Construir representación simbólica de los splines
        x_sym = sp.symbols('x')
        splines_symbolic_list = []
        splines_expr_list = []
        
        for i, spline_data in enumerate(splines_coefs_list):
            xi = x_coords[i]
            a_c, b_c, c_c, d_c = (spline_data['coeficientes']['a'], 
                                  spline_data['coeficientes']['b'],
                                  spline_data['coeficientes']['c'],
                                  spline_data['coeficientes']['d'])
            
            # S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
            term_expr = a_c + b_c*(x_sym-xi) + c_c*(x_sym-xi)**2 + d_c*(x_sym-xi)**3
            spline_func_piecewise = sp.Piecewise(
                (term_expr, sp.And(x_sym > xi, x_sym < x_coords[i+1])),
                (0, True)
            )
            splines_symbolic_list.append(spline_func_piecewise)
            splines_expr_list.append(term_expr)
        
        # Función combinada para todos los intervalos
        spline_combined_expr = sum(splines_symbolic_list)
        
        # Convertir a función para evaluación numérica
        spline_lambda_func = sp.lambdify(x_sym, spline_combined_expr, 'numpy')
        
        # Generar representación LaTeX más compatible con KaTeX
        poly_tex_cases = r"\begin{cases}"
        for i, single_expr in enumerate(splines_expr_list):
            interval_coords = splines_coefs_list[i]['intervalo']
            # Format the expression with proper LaTeX spacing and alignment
            expr_latex = sp.latex(single_expr)
            # Replace scientific notation expressions
            expr_latex = expr_latex.replace(r"10^{-", r"10^{-")
            
            # Add the cases entry with proper spacing
            poly_tex_cases += expr_latex + r", & " + sp.latex(interval_coords[0]) + r" < x < " + sp.latex(interval_coords[1])
            if i < len(splines_expr_list) - 1:
                poly_tex_cases += r" \\"
        poly_tex_cases += r"\end{cases}"
        
        # Corregir posibles problemas con matrices anidadas en el LaTeX
        poly_tex_final = fix_nested_matrices(poly_tex_cases)
        
        # Further improvements for KaTeX compatibility
        poly_tex_final = poly_tex_final.replace(r"\dot", r"\cdot")
        poly_tex_final = poly_tex_final.replace(r"\left(", r"(").replace(r"\right)", r")")
        
        # Fix scientific notation
        poly_tex_final = re.sub(r"\\cdot\s*10\^\{-(\d+)\}", r" \\times 10^{-\1}", poly_tex_final)
        poly_tex_final = re.sub(r"\\cdot\s*10\^\{(\d+)\}", r" \\times 10^{\1}", poly_tex_final)
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Función Spline Resultante</div>")
        steps.append(f"<div class='bg-gray-50 p-3 my-2 rounded border border-gray-200'>$$ S(x) = {poly_tex_final} $$</div>")
        
        # Generar la gráfica
        uuid_str_val = str(uuid.uuid4())
        
        # Generar la gráfica usando plot_splines
        plot_path_val = plot_splines(points, spline_lambda_func, x_coords, uuid_str_val)
        
        # Preparar resultados
        result = InterpResult(
            metodo="Interpolación con Splines Cúbicos",
            entrada={"puntos": points, "solver": solver},
            pasos=steps,
            salida={"splines": splines_coefs_list, "polinomio_tex": poly_tex_final},
            grafico_path=plot_path_val,
            coeficientes=[coef_val for spline_item in splines_coefs_list for coef_val in spline_item['coeficientes'].values()],
            polinomio_tex=poly_tex_final
        )
        
        return result
    except Exception as e:
        raise ValueError(f"Error en interpolación con splines: {str(e)}")

def plot_splines(points: List[Tuple[float, float]], spline_func_lambda, x_nodes, plot_uuid: Optional[str] = None) -> str:
    """
    Genera una gráfica con los puntos y la curva de splines.
    
    Args:
        points: Lista de puntos (x,y)
        spline_func: Función lambda que evalúa el spline
        x_points: Lista de puntos x que definen los intervalos
        uuid_str: Identificador único para el nombre del archivo
        
    Returns:
        Ruta al archivo de imagen generado
    """
    if plot_uuid is None:
        plot_uuid = str(uuid.uuid4())
    
    # Asegurar que el directorio de imágenes exista
    BASE_DIR_PATH = Path(__file__).resolve().parent.parent.parent
    STATIC_ROOT_IMG_PATH = BASE_DIR_PATH / 'static' / 'img'
    STATIC_URL_IMG_PREFIX_VAL = 'img/'
    STATIC_ROOT_IMG_PATH.mkdir(parents=True, exist_ok=True)
    
    x_data = np.array([p[0] for p in points])
    y_data = np.array([p[1] for p in points])
    
    fig, ax_plot = plt.subplots(figsize=(8, 6))
    ax_plot.scatter(x_data, y_data, color='red', label='Puntos Dados', zorder=5)
    
    # Generar puntos para la curva de splines
    x_min_val, x_max_val = np.min(x_data), np.max(x_data)
    range_pad = (x_max_val - x_min_val) * 0.05  # Un poco de margen
    
    # Para cada intervalo, evaluamos la función spline correspondiente
    for i in range(len(x_nodes) - 1):
        x_start_interval, x_end_interval = x_nodes[i], x_nodes[i+1]
        # Crear puntos que excluyan los extremos del intervalo
        x_dense_points = np.linspace(x_start_interval, x_end_interval, 102)[1:-1]  # Excluir primer y último punto
        y_dense_points = spline_func_lambda(x_dense_points)
        
        # Plotear el spline para este intervalo
        ax_plot.plot(x_dense_points, y_dense_points, '-', label=f'Spline {i+1}' if i == 0 else "")
    
    ax_plot.set_title('Interpolación con Splines Cúbicos')
    ax_plot.set_xlabel('x')
    ax_plot.set_ylabel('y')
    ax_plot.grid(True, linestyle='--', alpha=0.6)
    ax_plot.legend()
    
    # Guardar imagen
    img_filename = f"{plot_uuid}.png"
    full_img_path = STATIC_ROOT_IMG_PATH / img_filename
    plt.savefig(full_img_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    # Devolver la ruta relativa
    return os.path.join(STATIC_URL_IMG_PREFIX_VAL, img_filename).replace('\\', '/') 