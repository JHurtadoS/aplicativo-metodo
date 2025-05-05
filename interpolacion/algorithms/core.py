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
    steps = ["## Inicio: Interpolación de Lagrange (Usando Sistema de Ecuaciones)"]
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
        steps.append(f"**Puntos de entrada:** {points}")
        steps.append("\n### Construyendo el sistema lineal con matriz de Vandermonde")
        A, b = builder_vandermonde(points)
        
        steps.append(f"**Matriz de Vandermonde:**")
        steps.append(f"```\n{A}\n```")
        steps.append(f"**Vector de valores y:**")
        steps.append(f"```\n{b}\n```")
        
        # Resolver el sistema lineal usando el solver de sistemas lineales
        steps.append(f"\n### Resolviendo el sistema lineal (Método: {solver})")
        result_dict = solve_system(A, b, solver)
        
        # Extraer los coeficientes de la solución
        if 'error' in result_dict:
            raise ValueError(f"Error al resolver el sistema: {result_dict['error']}")
            
        coefs = result_dict['solution'] if 'solution' in result_dict else []
        steps.append(f"**Coeficientes obtenidos:** {coefs}")
        
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
        steps.append(f"\n### Polinomio Interpolante")
        steps.append(f"$P(x) = {poly_tex}$")
        
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
    steps = ["## Inicio: Interpolación de Newton (Usando Sistema Triangular)"]
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
        steps.append(f"**Puntos de entrada:** {points}")
        steps.append("\n### Construyendo el sistema lineal triangular para Newton")
        T, b = builder_newton_triangular(points)
        
        steps.append(f"**Matriz triangular de Newton:**")
        steps.append(f"```\n{T}\n```")
        steps.append(f"**Vector de valores y:**")
        steps.append(f"```\n{b}\n```")
        
        # Resolver el sistema lineal usando el solver de sistemas lineales
        steps.append(f"\n### Resolviendo el sistema lineal (Método: {solver})")
        result_dict = solve_system(T, b, solver)
        
        # Extraer los coeficientes de la solución
        if 'error' in result_dict:
            raise ValueError(f"Error al resolver el sistema: {result_dict['error']}")
            
        coefs = result_dict['solution'] if 'solution' in result_dict else []
        steps.append(f"**Coeficientes de diferencias divididas:** {coefs}")
        
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
        steps.append(f"\n### Polinomio Interpolante de Newton")
        steps.append(f"$P(x) = {poly_tex}$")
        
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
    steps = ["## Inicio: Regresión Lineal (Mínimos Cuadrados)"]
    try:
        n = len(points)
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        if n < 2:
            raise ValueError("Se necesitan al menos 2 puntos para la regresión.")

        steps.append(f"**Número de puntos:** {n}")
        steps.append(f"**Coordenadas x:** `{x_coords}`")
        steps.append(f"**Coordenadas y:** `{y_coords}`")

        # Calculate sums needed for normal equations
        sum_x = np.sum(x_coords)
        sum_y = np.sum(y_coords)
        sum_xy = np.sum(x_coords * y_coords)
        sum_x2 = np.sum(x_coords**2)

        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n

        steps.append("\n### Calculando coeficientes con ecuaciones normales")
        steps.append(f"- **Pendiente (a)**: `{slope:.6f}`")
        steps.append(f"- **Intercepto (b)**: `{intercept:.6f}`")

        # Build the polynomial with SymPy
        x = sp.symbols('x')
        poly = slope * x + intercept
        poly_tex = sp.latex(poly)
        # Corregir posibles problemas con matrices anidadas en el LaTeX
        poly_tex = fix_nested_matrices(poly_tex)
        
        # Create regression function
        def regression_func(x_vals):
            return slope * x_vals + intercept

        # Calculate R-squared
        y_pred = regression_func(x_coords)
        ss_total = np.sum((y_coords - np.mean(y_coords))**2)
        ss_residual = np.sum((y_coords - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        steps.append(f"\n### Ecuación de la recta")
        steps.append(f"$y = {slope:.6f}x + {intercept:.6f}$")
        steps.append(f"\n### Bondad del ajuste")
        steps.append(f"- **R²**: `{r_squared:.6f}`")

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
            steps.append(f"\n*Error al generar la gráfica: {e}*")

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