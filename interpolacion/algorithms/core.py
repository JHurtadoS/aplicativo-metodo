#  ProyectoMN – Licencia IMT
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import uuid
import os
from pathlib import Path
from sistemas_lineales.algorithms import solve_system
from core.utils import fix_nested_matrices
from .builder import builder_vandermonde, builder_newton_triangular
import re

# Asegurar que el directorio de imágenes exista
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_ROOT_IMG = BASE_DIR / 'static' / 'img'
STATIC_URL_IMG_PREFIX = 'img/'
STATIC_ROOT_IMG.mkdir(parents=True, exist_ok=True)

@dataclass
class Result:
    """Clase base para resultados de métodos numéricos."""
    metodo: str
    entrada: dict
    pasos: list
    salida: dict
    grafico_path: Optional[str] = None

@dataclass
class InterpResult(Result):
    """Resultado de interpolación numérica."""
    coeficientes: list[float] = field(default_factory=list)
    polinomio_tex: str = ""

def plot_points_curve(points: List[Tuple[float, float]], poly, uuid_str: Optional[str] = None) -> str:
    """Genera una gráfica con los puntos y la curva interpolante."""
    if uuid_str is None:
        uuid_str = str(uuid.uuid4())
    
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_coords, y_coords, color='red', label='Puntos Dados', zorder=5)
    
    # Curva interpolante
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    range_pad = (x_max - x_min) * 0.1
    x_dense = np.linspace(x_min - range_pad, x_max + range_pad, 500)
    
    # Convertir sympy poly a función numpy
    x_sym = sp.symbols('x')
    f_lambdified = sp.lambdify(x_sym, poly, 'numpy')
    y_dense = f_lambdified(x_dense)
    
    ax.plot(x_dense, y_dense, label='Polinomio Interpolante')
    ax.set_title('Interpolación Polinómica')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Guardar imagen
    filename = f"{uuid_str}.png"
    full_path = STATIC_ROOT_IMG / filename
    plt.savefig(full_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    # Devolver la ruta relativa
    return os.path.join(STATIC_URL_IMG_PREFIX, filename).replace('\\', '/')

def lagrange(points: List[Tuple[float, float]], decimales: int = 6, solver: str = "gauss") -> InterpResult:
    """Calcula el polinomio interpolante de Lagrange usando el solver indicado."""
    steps = ["<div class='text-2xl font-bold text-blue-800 mt-6 mb-3'>Inicio: Interpolación de Lagrange (Usando Sistema de Ecuaciones)</div>"]
    try:
        # Validar entrada
        if len(points) < 2:
            raise ValueError("Se necesitan al menos 2 puntos para la interpolación.")
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Verificar duplicados en x
        if len(x_coords) != len(set(x_coords)):
            raise ValueError("Los valores de x deben ser únicos para la interpolación.")
        
        # Construir el sistema lineal usando el builder
        steps.append(f"<div class='my-2'><span class='font-semibold'>Puntos de entrada:</span> {points}</div>")
        steps.append("<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Construyendo el sistema lineal con matriz de Vandermonde</div>")
        A, b = builder_vandermonde(points)
        
        steps.append(f"<div class='my-2'><span class='font-semibold'>Matriz de Vandermonde:</span></div>")
        steps.append(f"<pre class='bg-gray-50 p-3 rounded border border-gray-200 overflow-auto text-sm'>{A}</pre>")
        steps.append(f"<div class='my-2'><span class='font-semibold'>Vector de valores y:</span></div>")
        steps.append(f"<pre class='bg-gray-50 p-3 rounded border border-gray-200 overflow-auto text-sm'>{b}</pre>")
        
        # Resolver el sistema lineal usando el solver de sistemas lineales
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Resolviendo el sistema lineal (Método: {solver})</div>")
        result_dict = solve_system(A, b, solver)
        
        # Extraer los coeficientes de la solución
        if 'error' in result_dict:
            raise ValueError(f"Error al resolver el sistema: {result_dict['error']}")
            
        coefs = result_dict['solution'] if 'solution' in result_dict else []
        steps.append(f"<div class='my-2'><span class='font-semibold'>Coeficientes obtenidos:</span> <code class='bg-gray-100 px-1 rounded'>{coefs}</code></div>")
        
        # Construir el polinomio con SymPy
        x = sp.symbols('x')
        poly = 0
        for i, coef in enumerate(coefs):
            poly += coef * x**i
        
        # Simplificar el polinomio
        poly_simplified = sp.expand(poly)
        poly_tex = sp.latex(poly_simplified)
        # Corregir posibles problemas con matrices anidadas en el LaTeX
        poly_tex = fix_nested_matrices(poly_tex)
        # Mejoras para compatibilidad con KaTeX
        poly_tex = poly_tex.replace(r"\dot", r"\cdot").replace(r"\left(", r"(").replace(r"\right)", r")")
        
        # Fix scientific notation with regex
        # Replace scientific notation with better KaTeX formatting
        # Replace \cdot 10^{-X} with \times 10^{-X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{-(\d+)\}", r" \\times 10^{-\1}", poly_tex)
        # Replace \cdot 10^{X} with \times 10^{X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{(\d+)\}", r" \\times 10^{\1}", poly_tex)
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Polinomio Interpolante</div>")
        steps.append(f"<div class='bg-gray-50 p-3 my-2 rounded border border-gray-200'>$P(x) = {poly_tex}$</div>")
        
        # Generar la gráfica
        uuid_str = str(uuid.uuid4())
        plot_path = plot_points_curve(points, poly_simplified, uuid_str)
        
        # Crear el resultado
        result = InterpResult(
            metodo="Interpolación de Lagrange",
            entrada={"puntos": points, "solver": solver},
            pasos=steps,
            salida={"coeficientes": coefs, "polinomio_tex": poly_tex},
            grafico_path=plot_path,
            coeficientes=coefs,
            polinomio_tex=poly_tex
        )
        
        return result
    except Exception as e:
        raise ValueError(f"Error en interpolación de Lagrange: {str(e)}")

def newton(points: List[Tuple[float, float]], decimales: int = 6, solver: str = "gauss") -> InterpResult:
    """Calcula el polinomio interpolante de Newton (diferencias divididas)."""
    steps = ["<div class='text-2xl font-bold text-blue-800 mt-6 mb-3'>Inicio: Interpolación de Newton (Usando Sistema Triangular)</div>"]
    try:
        # Validar entrada
        if len(points) < 2:
            raise ValueError("Se necesitan al menos 2 puntos para la interpolación.")
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Verificar duplicados en x
        if len(x_coords) != len(set(x_coords)):
            raise ValueError("Los valores de x deben ser únicos para la interpolación.")
        
        # Construir el sistema lineal triangular
        steps.append(f"<div class='my-2'><span class='font-semibold'>Puntos de entrada:</span> {points}</div>")
        steps.append("<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Construyendo el sistema lineal triangular para Newton</div>")
        T, b = builder_newton_triangular(points)
        
        steps.append(f"<div class='my-2'><span class='font-semibold'>Matriz triangular de Newton:</span></div>")
        steps.append(f"<pre class='bg-gray-50 p-3 rounded border border-gray-200 overflow-auto text-sm'>{T}</pre>")
        steps.append(f"<div class='my-2'><span class='font-semibold'>Vector de valores y:</span></div>")
        steps.append(f"<pre class='bg-gray-50 p-3 rounded border border-gray-200 overflow-auto text-sm'>{b}</pre>")
        
        # Resolver el sistema lineal usando el solver de sistemas lineales
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Resolviendo el sistema lineal (Método: {solver})</div>")
        result_dict = solve_system(T, b, solver)
        
        # Extraer los coeficientes de la solución
        if 'error' in result_dict:
            raise ValueError(f"Error al resolver el sistema: {result_dict['error']}")
            
        coefs = result_dict['solution'] if 'solution' in result_dict else []
        steps.append(f"<div class='my-2'><span class='font-semibold'>Coeficientes de diferencias divididas:</span> <code class='bg-gray-100 px-1 rounded'>{coefs}</code></div>")
        
        # Construir el polinomio de Newton con SymPy
        x = sp.symbols('x')
        poly = coefs[0]  # Término constante
        prod = 1
        
        for i in range(1, len(coefs)):
            prod *= (x - x_coords[i-1])
            poly += coefs[i] * prod
        
        # Simplificar el polinomio
        poly_simplified = sp.expand(poly)
        poly_tex = sp.latex(poly_simplified)
        # Corregir posibles problemas con matrices anidadas en el LaTeX
        poly_tex = fix_nested_matrices(poly_tex)
        # Mejoras para compatibilidad con KaTeX
        poly_tex = poly_tex.replace(r"\dot", r"\cdot").replace(r"\left(", r"(").replace(r"\right)", r")")
        
        # Fix scientific notation with regex
        # Replace scientific notation with better KaTeX formatting
        # Replace \cdot 10^{-X} with \times 10^{-X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{-(\d+)\}", r" \\times 10^{-\1}", poly_tex)
        # Replace \cdot 10^{X} with \times 10^{X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{(\d+)\}", r" \\times 10^{\1}", poly_tex)
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Polinomio Interpolante de Newton</div>")
        steps.append(f"<div class='bg-gray-50 p-3 my-2 rounded border border-gray-200'>$P(x) = {poly_tex}$</div>")
        
        # Generar la gráfica
        uuid_str = str(uuid.uuid4())
        plot_path = plot_points_curve(points, poly_simplified, uuid_str)
        
        # Crear el resultado
        result = InterpResult(
            metodo="Interpolación de Newton",
            entrada={"puntos": points, "solver": solver},
            pasos=steps,
            salida={"coeficientes": coefs, "polinomio_tex": poly_tex},
            grafico_path=plot_path,
            coeficientes=coefs,
            polinomio_tex=poly_tex
        )
        
        return result
    except Exception as e:
        raise ValueError(f"Error en interpolación de Newton: {str(e)}")

def linear_regression(points: List[Tuple[float, float]]) -> Result:
    """Calcula la regresión lineal usando mínimos cuadrados."""
    steps = ["<div class='text-2xl font-bold text-blue-800 mt-6 mb-3'>Inicio: Regresión Lineal (Mínimos Cuadrados)</div>"]
    try:
        n = len(points)
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        if n < 2:
            raise ValueError("Se necesitan al menos 2 puntos para la regresión.")

        steps.append(f"<div class='my-2'><span class='font-semibold'>Número de puntos:</span> {n}</div>")
        steps.append(f"<div class='my-2'><span class='font-semibold'>Coordenadas x:</span> <code class='bg-gray-100 px-1 rounded'>{x_coords}</code></div>")
        steps.append(f"<div class='my-2'><span class='font-semibold'>Coordenadas y:</span> <code class='bg-gray-100 px-1 rounded'>{y_coords}</code></div>")

        # Calculate sums needed for normal equations
        sum_x = np.sum(x_coords)
        sum_y = np.sum(y_coords)
        sum_xy = np.sum(x_coords * y_coords)
        sum_x2 = np.sum(x_coords**2)

        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n

        steps.append("<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Calculando coeficientes con ecuaciones normales</div>")
        steps.append(f"<div class='ml-3 my-1'><span class='font-medium text-blue-600'>Pendiente (a):</span> <code class='bg-gray-100 px-1 rounded'>{slope:.6f}</code></div>")
        steps.append(f"<div class='ml-3 my-1'><span class='font-medium text-blue-600'>Intercepto (b):</span> <code class='bg-gray-100 px-1 rounded'>{intercept:.6f}</code></div>")

        # Build the polynomial with SymPy
        x = sp.symbols('x')
        poly = slope * x + intercept
        poly_tex = sp.latex(poly)
        # Corregir posibles problemas con matrices anidadas en el LaTeX
        poly_tex = fix_nested_matrices(poly_tex)
        # Mejoras para compatibilidad con KaTeX
        poly_tex = poly_tex.replace(r"\dot", r"\cdot").replace(r"\left(", r"(").replace(r"\right)", r")")
        
        # Fix scientific notation with regex
        # Replace scientific notation with better KaTeX formatting
        # Replace \cdot 10^{-X} with \times 10^{-X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{-(\d+)\}", r" \\times 10^{-\1}", poly_tex)
        # Replace \cdot 10^{X} with \times 10^{X}
        poly_tex = re.sub(r"\\cdot\s*10\^\{(\d+)\}", r" \\times 10^{\1}", poly_tex)
        
        # Create regression function
        def regression_func(x_vals):
            return slope * x_vals + intercept

        # Calculate R-squared
        y_pred = regression_func(x_coords)
        ss_total = np.sum((y_coords - np.mean(y_coords))**2)
        ss_residual = np.sum((y_coords - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Ecuación de la recta</div>")
        steps.append(f"<div class='bg-gray-50 p-3 my-2 rounded border border-gray-200'>$y = {slope:.6f}x + {intercept:.6f}$</div>")
        
        steps.append(f"<div class='text-xl font-bold text-blue-700 mt-5 mb-2'>Bondad del ajuste</div>")
        steps.append(f"<div class='ml-3 my-1'><span class='font-medium text-blue-600'>R²:</span> <code class='bg-gray-100 px-1 rounded'>{r_squared:.6f}</code></div>")

        # Generate plot
        uuid_str = str(uuid.uuid4())
        plot_path = None
        
        # Creación del gráfico
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_coords, y_coords, color='red', label='Puntos Dados', zorder=5)
            
            # Línea de regresión
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            range_pad = (x_max - x_min) * 0.1
            x_dense = np.linspace(x_min - range_pad, x_max + range_pad, 500)
            y_dense = regression_func(x_dense)
            
            ax.plot(x_dense, y_dense, label=f'y = {slope:.4f}x + {intercept:.4f}')
            ax.set_title('Regresión Lineal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Guardar imagen
            filename = f"{uuid_str}.png"
            full_path = STATIC_ROOT_IMG / filename
            plt.savefig(full_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            
            plot_path = os.path.join(STATIC_URL_IMG_PREFIX, filename).replace('\\', '/')
        except Exception as e:
            steps.append(f"<div class='text-red-600 italic mt-2'>Error al generar la gráfica: {e}</div>")

        result = Result(
            metodo="Regresión Lineal (Mínimos Cuadrados)",
            entrada={"puntos": points},
            pasos=steps,
            salida={
                "pendiente": slope,
                "intercepto": intercept,
                "ecuacion": f"y = {slope:.6f}x + {intercept:.6f}",
                "r_cuadrado": r_squared
            },
            grafico_path=plot_path
        )
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error en regresión lineal: {str(e)}") 