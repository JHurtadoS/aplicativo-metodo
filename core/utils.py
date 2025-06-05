import os
import uuid
from datetime import datetime
from pathlib import Path
from weasyprint import HTML, CSS
from django.template.loader import render_to_string
from django.conf import settings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configurar matplotlib para español
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Asegurar que existan los directorios necesarios
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / 'static' / 'pdfs'
IMAGES_DIR = BASE_DIR / 'static' / 'images'
PDF_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class NMResult:
    """DataClass unificado para resultados de métodos numéricos"""
    tema: str               # 'derivacion', 'integracion', 'edo'
    metodo: str             # 'central', 'simpson', 'rk4', etc.
    entrada: Dict[str, Any] # datos del formulario
    pasos: List[Dict]       # estructuras intermedias
    valor: float            # resultado final
    error: Optional[float]  # error estimado/exacto
    grafico_path: Optional[str]  # ruta a PNG
    analisis: Optional[Dict[str, Any]] = None  # análisis estadístico (principalmente para EDOs)
    
    def __post_init__(self):
        if not hasattr(self, 'detalles') or self.detalles is None:
            self.detalles = {}
        if self.analisis is None:
            self.analisis = {}

# =============================================================================
# FUNCIONES DE DERIVACIÓN NUMÉRICA
# =============================================================================

def forward_diff(func, x0, h):
    """Diferencia hacia adelante: f'(x0) ≈ (f(x0+h) - f(x0))/h"""
    f_x0 = func(x0)
    f_x0_h = func(x0 + h)
    resultado = (f_x0_h - f_x0) / h
    
    pasos = [
        {'descripcion': 'Evaluación f(x₀)', 'calculo': f'f({x0}) = {f_x0}'},
        {'descripcion': 'Evaluación f(x₀+h)', 'calculo': f'f({x0 + h}) = {f_x0_h}'},
        {'descripcion': 'Aplicar fórmula', 'calculo': f"f'({x0}) ≈ ({f_x0_h} - {f_x0})/{h} = {resultado}"}
    ]
    
    return resultado, pasos

def backward_diff(func, x0, h):
    """Diferencia hacia atrás: f'(x0) ≈ (f(x0) - f(x0-h))/h"""
    f_x0 = func(x0)
    f_x0_h = func(x0 - h)
    resultado = (f_x0 - f_x0_h) / h
    
    pasos = [
        {'descripcion': 'Evaluación f(x₀)', 'calculo': f'f({x0}) = {f_x0}'},
        {'descripcion': 'Evaluación f(x₀-h)', 'calculo': f'f({x0 - h}) = {f_x0_h}'},
        {'descripcion': 'Aplicar fórmula', 'calculo': f"f'({x0}) ≈ ({f_x0} - {f_x0_h})/{h} = {resultado}"}
    ]
    
    return resultado, pasos

def central_diff(func, x0, h):
    """Diferencia central: f'(x0) ≈ (f(x0+h) - f(x0-h))/(2h)"""
    f_x0_plus_h = func(x0 + h)
    f_x0_minus_h = func(x0 - h)
    resultado = (f_x0_plus_h - f_x0_minus_h) / (2 * h)
    
    pasos = [
        {'descripcion': 'Evaluación f(x₀+h)', 'calculo': f'f({x0 + h}) = {f_x0_plus_h}'},
        {'descripcion': 'Evaluación f(x₀-h)', 'calculo': f'f({x0 - h}) = {f_x0_minus_h}'},
        {'descripcion': 'Aplicar fórmula central', 'calculo': f"f'({x0}) ≈ ({f_x0_plus_h} - {f_x0_minus_h})/(2×{h}) = {resultado}"}
    ]
    
    return resultado, pasos

def second_derivative_central(func, x0, h):
    """Segunda derivada central: f''(x0) ≈ (f(x0+h) - 2f(x0) + f(x0-h))/h²"""
    f_x0_plus_h = func(x0 + h)
    f_x0 = func(x0)
    f_x0_minus_h = func(x0 - h)
    resultado = (f_x0_plus_h - 2*f_x0 + f_x0_minus_h) / (h**2)
    
    pasos = [
        {'descripcion': 'Evaluación f(x₀+h)', 'calculo': f'f({x0 + h}) = {f_x0_plus_h}'},
        {'descripcion': 'Evaluación f(x₀)', 'calculo': f'f({x0}) = {f_x0}'},
        {'descripcion': 'Evaluación f(x₀-h)', 'calculo': f'f({x0 - h}) = {f_x0_minus_h}'},
        {'descripcion': 'Aplicar fórmula segunda derivada', 'calculo': f"f''({x0}) ≈ ({f_x0_plus_h} - 2×{f_x0} + {f_x0_minus_h})/{h}² = {resultado}"}
    ]
    
    return resultado, pasos

def five_point_second_derivative(func, x0, h):
    """Segunda derivada con 5 puntos - O(h⁴): 
    f''(x0) ≈ (-f(x0-2h) + 16f(x0-h) - 30f(x0) + 16f(x0+h) - f(x0+2h))/(12h²)"""
    f_x0_minus_2h = func(x0 - 2*h)
    f_x0_minus_h = func(x0 - h)
    f_x0 = func(x0)
    f_x0_plus_h = func(x0 + h)
    f_x0_plus_2h = func(x0 + 2*h)
    
    resultado = (-f_x0_minus_2h + 16*f_x0_minus_h - 30*f_x0 + 16*f_x0_plus_h - f_x0_plus_2h) / (12 * h**2)
    
    pasos = [
        {'descripcion': 'Evaluación f(x₀-2h)', 'calculo': f'f({x0 - 2*h}) = {f_x0_minus_2h}'},
        {'descripcion': 'Evaluación f(x₀-h)', 'calculo': f'f({x0 - h}) = {f_x0_minus_h}'},
        {'descripcion': 'Evaluación f(x₀)', 'calculo': f'f({x0}) = {f_x0}'},
        {'descripcion': 'Evaluación f(x₀+h)', 'calculo': f'f({x0 + h}) = {f_x0_plus_h}'},
        {'descripcion': 'Evaluación f(x₀+2h)', 'calculo': f'f({x0 + 2*h}) = {f_x0_plus_2h}'},
        {'descripcion': 'Aplicar fórmula 5 puntos', 'calculo': f"f''({x0}) ≈ (-{f_x0_minus_2h} + 16×{f_x0_minus_h} - 30×{f_x0} + 16×{f_x0_plus_h} - {f_x0_plus_2h})/(12×{h}²) = {resultado}"}
    ]
    
    return resultado, pasos

def richardson_extrapolation(func, x0, h):
    """Extrapolación de Richardson para primera derivada: R = (4*D(h/2) - D(h))/3"""
    # Calcular derivada con paso h usando diferencia central
    _, pasos_h = central_diff(func, x0, h)
    d_h = (func(x0 + h) - func(x0 - h)) / (2 * h)
    
    # Calcular derivada con paso h/2
    h_half = h / 2
    _, pasos_h_half = central_diff(func, x0, h_half)
    d_h_half = (func(x0 + h_half) - func(x0 - h_half)) / (2 * h_half)
    
    # Aplicar Richardson
    resultado = (4 * d_h_half - d_h) / 3
    
    pasos = [
        {'descripcion': 'Derivada con paso h', 'calculo': f"D(h={h}) = {d_h}"},
        {'descripcion': 'Derivada con paso h/2', 'calculo': f"D(h/2={h_half}) = {d_h_half}"},
        {'descripcion': 'Extrapolación Richardson', 'calculo': f"R = (4×{d_h_half} - {d_h})/3 = {resultado}"}
    ]
    
    return resultado, pasos

def calcular_derivada(function_str, x0, h, method):
    """Función principal para calcular derivadas numéricas"""
    try:
        # Convertir función string a función evaluable
        x = sp.Symbol('x')
        expr = sp.sympify(function_str)
        func = sp.lambdify(x, expr, 'numpy')
        
        # Seleccionar método
        methods = {
            'forward': forward_diff,
            'backward': backward_diff,
            'central': central_diff,
            'second_derivative': second_derivative_central,
            'five_point': five_point_second_derivative,
            'richardson': richardson_extrapolation
        }
        
        if method not in methods:
            raise ValueError(f"Método no reconocido: {method}")
        
        resultado, pasos = methods[method](func, x0, h)
        
        # Calcular error de curvatura para diferencias centrales y análisis de truncamiento vs redondeo
        error = None
        error_info = {}
        
        if method == 'central':
            try:
                f_x0 = func(x0)
                f_x0_plus_h = func(x0 + h)
                f_x0_minus_h = func(x0 - h)
                error_curvatura = abs(f_x0_plus_h - 2*f_x0 + f_x0_minus_h) / (h**2)
                error = error_curvatura
                error_info['curvature_error'] = error_curvatura
                error_info['tipo'] = 'Error de curvatura (Ec)'
            except:
                error = None
        
        # Para métodos de segunda derivada, intentar calcular derivada exacta si es posible
        if method in ['second_derivative', 'five_point']:
            try:
                segunda_derivada_exacta = sp.diff(expr, x, 2)
                valor_exacto_expr = segunda_derivada_exacta.subs(x, x0)
                valor_exacto = float(sp.N(valor_exacto_expr))
                error_absoluto = abs(resultado - valor_exacto)
                error_relativo = error_absoluto / abs(valor_exacto) if valor_exacto != 0 else float('inf')
                error = error_absoluto
                error_info.update({
                    'exact_value': valor_exacto,
                    'absolute_error': error_absoluto,
                    'relative_error': error_relativo,
                    'tipo': 'Error vs solución exacta'
                })
            except:
                error = None
        
        # Para Richardson, mostrar mejora de orden
        if method == 'richardson':
            try:
                # Calcular derivada exacta para comparar
                primera_derivada_exacta = sp.diff(expr, x, 1)
                valor_exacto_expr = primera_derivada_exacta.subs(x, x0)
                valor_exacto = float(sp.N(valor_exacto_expr))
                error_absoluto = abs(resultado - valor_exacto)
                error_relativo = error_absoluto / abs(valor_exacto) if valor_exacto != 0 else float('inf')
                error = error_absoluto
                error_info.update({
                    'exact_value': valor_exacto,
                    'absolute_error': error_absoluto,
                    'relative_error': error_relativo,
                    'tipo': 'Error Richardson vs exacta'
                })
            except:
                error = None
        
        return resultado, pasos, error, func, expr, error_info
        
    except Exception as e:
        raise ValueError(f"Error en el cálculo de la derivada: {str(e)}")

# =============================================================================
# FUNCIONES DE INTEGRACIÓN NUMÉRICA
# =============================================================================

def trapezoidal_rule(func, a, b, n):
    """Regla del trapecio compuesta con pasos detallados"""
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = [func(x) for x in x_vals]
    
    pasos = []
    
    # Paso 1: Explicación del método
    pasos.append({
        'descripcion': 'Fórmula de la Regla del Trapecio Compuesta',
        'calculo': f'La regla del trapecio aproxima la integral dividiendo el intervalo en $n = {n}$ subintervalos y aproximando el área bajo la curva usando trapezoides. La fórmula es: $$\\int_{{a}}^{{b}} f(x)dx \\approx \\frac{{h}}{{2}}[f(x_0) + 2f(x_1) + 2f(x_2) + \\cdots + 2f(x_{{n-1}}) + f(x_n)]$$'
    })
    
    # Paso 2: Cálculo del ancho de paso
    pasos.append({
        'descripcion': 'Cálculo del ancho de paso h',
        'calculo': f'$$h = \\frac{{b - a}}{{n}} = \\frac{{{b} - {a}}}{{{n}}} = {h:.6f}$$'
    })
    
    # Paso 3: Definición de los puntos xi
    pasos.append({
        'descripcion': 'Puntos de evaluación',
        'calculo': f'Los puntos de evaluación se calculan como: $$x_i = a + i \\cdot h = {a} + i \\cdot {h:.6f}$$ para $i = 0, 1, 2, \\ldots, {n}$'
    })
    
    # Paso 4: Mostrar todos los puntos xi
    puntos_str = ', '.join([f'x_{{{i}}} = {x:.4f}' for i, x in enumerate(x_vals)])
    pasos.append({
        'descripcion': 'Valores específicos de los puntos',
        'calculo': f'$${puntos_str}$$'
    })
    
    # Paso 5: Evaluación de la función en cada punto
    pasos.append({
        'descripcion': 'Evaluación de f(x) en cada punto',
        'calculo': 'Calculamos $f(x_i)$ para cada punto:'
    })
    
    for i, (x, f_x) in enumerate(zip(x_vals, f_vals)):
        coef = 1 if i == 0 or i == n else 2
        coef_str = "1" if coef == 1 else "2"
        pasos.append({
            'descripcion': f'f(x_{i}) - Punto {"extremo" if coef == 1 else "interior"}',
            'calculo': f'$$f(x_{{{i}}}) = f({x:.4f}) = {f_x:.6f}$$ Coeficiente: ${coef_str}$'
        })
    
    # Paso 6: Aplicación de la fórmula
    suma_extremos = f_vals[0] + f_vals[-1]
    suma_interiores = sum(f_vals[1:-1])
    
    pasos.append({
        'descripcion': 'Suma de términos extremos',
        'calculo': f'$$f(x_0) + f(x_{n}) = {f_vals[0]:.6f} + {f_vals[-1]:.6f} = {suma_extremos:.6f}$$'
    })
    
    if len(f_vals) > 2:
        pasos.append({
            'descripcion': 'Suma de términos interiores (multiplicados por 2)',
            'calculo': f'$$2 \\sum_{{i=1}}^{{n-1}} f(x_i) = 2({" + ".join([f"{f_x:.6f}" for f_x in f_vals[1:-1]])}) = 2 \\times {suma_interiores:.6f} = {2*suma_interiores:.6f}$$'
        })
    
    # Paso 7: Cálculo final
    suma_total = suma_extremos + 2*suma_interiores
    resultado = h * suma_total / 2
    
    pasos.append({
        'descripcion': 'Suma total dentro de los corchetes',
        'calculo': f'$$f(x_0) + 2\\sum_{{i=1}}^{{n-1}} f(x_i) + f(x_n) = {suma_extremos:.6f} + {2*suma_interiores:.6f} = {suma_total:.6f}$$'
    })
    
    pasos.append({
        'descripcion': 'Resultado final de la integral',
        'calculo': f'$$\\int_{{{a}}}^{{{b}}} f(x)dx \\approx \\frac{{h}}{{2}} \\times {suma_total:.6f} = \\frac{{{h:.6f}}}{{2}} \\times {suma_total:.6f} = {resultado:.6f}$$'
    })
    
    return resultado, pasos, x_vals, f_vals

def simpson_rule(func, a, b, n):
    """Regla de Simpson 1/3 compuesta con pasos detallados"""
    if n % 2 != 0:
        raise ValueError("Simpson requiere número par de subintervalos")
    
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = [func(x) for x in x_vals]
    
    pasos = []
    
    # Paso 1: Explicación del método
    pasos.append({
        'descripcion': 'Fórmula de la Regla de Simpson 1/3 Compuesta',
        'calculo': f'La regla de Simpson usa parábolas para aproximar la función entre cada tres puntos consecutivos. Requiere un número par de subintervalos ($n = {n}$). La fórmula es: $$\\int_{{a}}^{{b}} f(x)dx \\approx \\frac{{h}}{{3}}[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \\cdots + 4f(x_{{n-1}}) + f(x_n)]$$'
    })
    
    # Paso 2: Cálculo del ancho de paso
    pasos.append({
        'descripcion': 'Cálculo del ancho de paso h',
        'calculo': f'$$h = \\frac{{b - a}}{{n}} = \\frac{{{b} - {a}}}{{{n}}} = {h:.6f}$$'
    })
    
    # Paso 3: Definición de los puntos xi
    pasos.append({
        'descripcion': 'Puntos de evaluación',
        'calculo': f'Los puntos de evaluación se calculan como: $$x_i = a + i \\cdot h = {a} + i \\cdot {h:.6f}$$ para $i = 0, 1, 2, \\ldots, {n}$'
    })
    
    # Paso 4: Mostrar todos los puntos xi
    puntos_str = ', '.join([f'x_{{{i}}} = {x:.4f}' for i, x in enumerate(x_vals)])
    pasos.append({
        'descripcion': 'Valores específicos de los puntos',
        'calculo': f'$${puntos_str}$$'
    })
    
    # Paso 5: Explicación de coeficientes
    pasos.append({
        'descripcion': 'Patrón de coeficientes en Simpson',
        'calculo': 'Los coeficientes siguen el patrón: $1, 4, 2, 4, 2, \\ldots, 4, 1$ donde: \\n• Extremos ($x_0, x_n$): coeficiente $1$ \\n• Índices impares ($x_1, x_3, x_5, \\ldots$): coeficiente $4$ \\n• Índices pares internos ($x_2, x_4, x_6, \\ldots$): coeficiente $2$'
    })
    
    # Paso 6: Evaluación de la función en cada punto
    pasos.append({
        'descripcion': 'Evaluación de f(x) en cada punto con sus coeficientes',
        'calculo': 'Calculamos $f(x_i)$ para cada punto y asignamos el coeficiente correspondiente:'
    })
    
    extremos = []
    impares = []
    pares_internos = []
    
    for i, (x, f_x) in enumerate(zip(x_vals, f_vals)):
        if i == 0 or i == n:
            coef = 1
            tipo = "extremo"
            extremos.append(f_x)
        elif i % 2 == 1:
            coef = 4
            tipo = "índice impar"
            impares.append(f_x)
        else:
            coef = 2
            tipo = "índice par interno"
            pares_internos.append(f_x)
            
        pasos.append({
            'descripcion': f'f(x_{i}) - {tipo.title()}',
            'calculo': f'$$f(x_{{{i}}}) = f({x:.4f}) = {f_x:.6f}$$ Coeficiente: ${coef}$'
        })
    
    # Paso 7: Agrupación por coeficientes
    suma_extremos = sum(extremos)
    suma_impares = sum(impares)
    suma_pares_internos = sum(pares_internos)
    
    pasos.append({
        'descripcion': 'Suma de términos extremos (coeficiente 1)',
        'calculo': f'$$f(x_0) + f(x_{n}) = {" + ".join([f"{f:.6f}" for f in extremos])} = {suma_extremos:.6f}$$'
    })
    
    if impares:
        pasos.append({
            'descripcion': 'Suma de términos con índices impares (coeficiente 4)',
            'calculo': f'$$4 \\sum_{{\\text{{impares}}}} f(x_i) = 4({" + ".join([f"{f:.6f}" for f in impares])}) = 4 \\times {suma_impares:.6f} = {4*suma_impares:.6f}$$'
        })
    
    if pares_internos:
        pasos.append({
            'descripcion': 'Suma de términos con índices pares internos (coeficiente 2)',
            'calculo': f'$$2 \\sum_{{\\text{{pares internos}}}} f(x_i) = 2({" + ".join([f"{f:.6f}" for f in pares_internos])}) = 2 \\times {suma_pares_internos:.6f} = {2*suma_pares_internos:.6f}$$'
        })
    
    # Paso 8: Cálculo final
    suma_total = suma_extremos + 4*suma_impares + 2*suma_pares_internos
    resultado = h * suma_total / 3
    
    pasos.append({
        'descripcion': 'Suma total dentro de los corchetes',
        'calculo': f'$$\\text{{Suma total}} = {suma_extremos:.6f} + {4*suma_impares:.6f} + {2*suma_pares_internos:.6f} = {suma_total:.6f}$$'
    })
    
    pasos.append({
        'descripcion': 'Resultado final de la integral',
        'calculo': f'$$\\int_{{{a}}}^{{{b}}} f(x)dx \\approx \\frac{{h}}{{3}} \\times {suma_total:.6f} = \\frac{{{h:.6f}}}{{3}} \\times {suma_total:.6f} = {resultado:.6f}$$'
    })
    
    return resultado, pasos, x_vals, f_vals

def midpoint_rule(func, a, b, n):
    """Regla del punto medio compuesta con pasos detallados"""
    h = (b - a) / n
    x_vals = []
    f_vals = []
    
    pasos = []
    
    # Paso 1: Explicación del método
    pasos.append({
        'descripcion': 'Fórmula de la Regla del Punto Medio Compuesta',
        'calculo': f'La regla del punto medio evalúa la función en el centro de cada subintervalo. Dividimos el intervalo en $n = {n}$ subintervalos y usamos: $$\\int_{{a}}^{{b}} f(x)dx \\approx h \\sum_{{i=0}}^{{n-1}} f\\left(x_{{i+\\frac{{1}}{{2}}}}\\right)$$ donde $x_{{i+\\frac{{1}}{{2}}}}$ es el punto medio del subintervalo $[x_i, x_{{i+1}}]$'
    })
    
    # Paso 2: Cálculo del ancho de paso
    pasos.append({
        'descripcion': 'Cálculo del ancho de paso h',
        'calculo': f'$$h = \\frac{{b - a}}{{n}} = \\frac{{{b} - {a}}}{{{n}}} = {h:.6f}$$'
    })
    
    # Paso 3: Explicación de puntos medios
    pasos.append({
        'descripcion': 'Puntos medios de cada subintervalo',
        'calculo': f'Para cada subintervalo $[x_i, x_{{i+1}}]$, el punto medio se calcula como: $$x_{{i+\\frac{{1}}{{2}}}} = a + \\left(i + \\frac{{1}}{{2}}\\right) \\cdot h = {a} + \\left(i + 0.5\\right) \\times {h:.6f}$$'
    })
    
    # Paso 4: Cálculo de puntos medios y evaluación
    pasos.append({
        'descripcion': 'Cálculo de cada punto medio y evaluación de f(x)',
        'calculo': 'Calculamos los puntos medios y evaluamos la función en cada uno:'
    })
    
    suma = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        f_mid = func(x_mid)
        x_vals.append(x_mid)
        f_vals.append(f_mid)
        suma += f_mid
        
        # Mostrar cálculo del punto medio
        pasos.append({
            'descripcion': f'Subintervalo {i+1}: [{a + i*h:.4f}, {a + (i+1)*h:.4f}]',
            'calculo': f'$$x_{{{i}+\\frac{{1}}{{2}}}} = {a} + ({i} + 0.5) \\times {h:.6f} = {x_mid:.6f}$$'
        })
        
        # Mostrar evaluación de la función
        pasos.append({
            'descripcion': f'Evaluación de f en el punto medio {i+1}',
            'calculo': f'$$f(x_{{{i}+\\frac{{1}}{{2}}}}) = f({x_mid:.6f}) = {f_mid:.6f}$$'
        })
    
    # Paso 5: Suma de todas las evaluaciones
    pasos.append({
        'descripcion': 'Suma de todas las evaluaciones de función',
        'calculo': f'$$\\sum_{{i=0}}^{{{n-1}}} f\\left(x_{{i+\\frac{{1}}{{2}}}}\\right) = {" + ".join([f"{f:.6f}" for f in f_vals])} = {suma:.6f}$$'
    })
    
    # Paso 6: Cálculo final
    resultado = h * suma
    
    pasos.append({
        'descripcion': 'Resultado final de la integral',
        'calculo': f'$$\\int_{{{a}}}^{{{b}}} f(x)dx \\approx h \\times \\sum_{{i=0}}^{{{n-1}}} f\\left(x_{{i+\\frac{{1}}{{2}}}}\\right) = {h:.6f} \\times {suma:.6f} = {resultado:.6f}$$'
    })
    
    return resultado, pasos, np.array(x_vals), f_vals

def calcular_integral(function_str, a, b, n, method):
    """Función principal para calcular integrales numéricas"""
    try:
        # Convertir función string a función evaluable
        x = sp.Symbol('x')
        expr = sp.sympify(function_str)
        func = sp.lambdify(x, expr, 'numpy')
        
        # Seleccionar método
        if method == 'trapezoidal':
            resultado, pasos, x_vals, f_vals = trapezoidal_rule(func, a, b, n)
        elif method == 'simpson':
            resultado, pasos, x_vals, f_vals = simpson_rule(func, a, b, n)
        elif method == 'midpoint':
            resultado, pasos, x_vals, f_vals = midpoint_rule(func, a, b, n)
        else:
            raise ValueError(f"Método no reconocido: {method}")
        
        # Calcular integral exacta si es posible para estimar error
        error = None
        error_info = {}
        h = (b - a) / n
        
        try:
            # Agregar separador para sección de análisis de errores
            pasos.append({
                'descripcion': '--- ANÁLISIS DE ERRORES ---',
                'calculo': 'A continuación analizamos la precisión del método calculando diferentes tipos de errores:'
            })
            
            # Paso 1: Cálculo de la integral exacta
            pasos.append({
                'descripcion': 'Cálculo de la integral exacta',
                'calculo': f'Intentamos calcular la integral exacta usando cálculo simbólico: $$\\int_{{{a}}}^{{{b}}} ({expr}) dx$$'
            })
            
            integral_exacta_expr = sp.integrate(expr, (x, a, b))
            
            pasos.append({
                'descripcion': 'Resultado de la integración simbólica',
                'calculo': f'$$\\int_{{{a}}}^{{{b}}} ({expr}) dx = {sp.latex(integral_exacta_expr)}$$'
            })
            
            # Convertir a string y luego a float para evitar problemas de tipo
            integral_exacta_str = str(integral_exacta_expr.evalf())
            pasos.append({
                'descripcion': 'Evaluación numérica de la integral exacta',
                'calculo': f'$$\\text{{Valor exacto}} = {integral_exacta_expr} = {integral_exacta_str}$$'
            })
            
            if integral_exacta_str.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                integral_exacta = float(integral_exacta_str)
                
                # Paso 2: Cálculo del error absoluto
                error_absoluto = abs(resultado - integral_exacta)
                pasos.append({
                    'descripcion': 'Cálculo del error absoluto',
                    'calculo': f'$$E_{{\\text{{abs}}}} = |\\text{{Aproximación}} - \\text{{Valor exacto}}| = |{resultado:.8f} - {integral_exacta:.8f}| = {error_absoluto:.8f}$$'
                })
                
                # Paso 3: Cálculo del error relativo
                if integral_exacta != 0:
                    error_relativo = error_absoluto / abs(integral_exacta)
                    error_relativo_porcentaje = error_relativo * 100
                    pasos.append({
                        'descripcion': 'Cálculo del error relativo',
                        'calculo': f'$$E_{{\\text{{rel}}}} = \\frac{{E_{{\\text{{abs}}}}}}{{|\\text{{Valor exacto}}|}} = \\frac{{{error_absoluto:.8f}}}{{|{integral_exacta:.8f}|}} = {error_relativo:.8f} = {error_relativo_porcentaje:.6f}\\%$$'
                    })
                else:
                    error_relativo = float('inf')
                    error_relativo_porcentaje = float('inf')
                    pasos.append({
                        'descripcion': 'Error relativo no definido',
                        'calculo': 'El error relativo no está definido porque el valor exacto es cero.'
                    })
                
                error = error_absoluto
                error_info.update({
                    'exact_value': integral_exacta,
                    'absolute_error': error_absoluto,
                    'relative_error': error_relativo_porcentaje,
                    'tipo': 'Error vs integral exacta'
                })
                
                # Paso 4: Análisis del error teórico según el método
                pasos.append({
                    'descripcion': 'Análisis del error teórico',
                    'calculo': f'Ahora calculamos el error teórico predicho por la teoría para el método {method}:'
                })
                
                if method == 'trapezoidal':
                    try:
                        pasos.append({
                            'descripcion': 'Fórmula del error teórico para Trapecio',
                            'calculo': f'Para la regla del trapecio, el error teórico está dado por: $$E_{{\\text{{trapecio}}}} \\leq \\frac{{(b-a)^3}}{{12n^2}} \\max_{{x \\in [a,b]}} |f\'\'(x)|$$'
                        })
                        
                        # Calcular segunda derivada
                        segunda_derivada = sp.diff(expr, x, 2)
                        pasos.append({
                            'descripcion': 'Cálculo de la segunda derivada',
                            'calculo': f'$$f\'\'(x) = \\frac{{d^2}}{{dx^2}}({expr}) = {sp.latex(segunda_derivada)}$$'
                        })
                        
                        # Estimar máximo de |f''(x)| en el intervalo
                        x_test = np.linspace(a, b, 100)
                        f_segunda_vals = [abs(float(sp.N(segunda_derivada.subs(x, xi)))) for xi in x_test]
                        max_f_segunda = max(f_segunda_vals)
                        
                        pasos.append({
                            'descripcion': 'Estimación del máximo de |f\'\'(x)|',
                            'calculo': f'Evaluando $|f\'\'(x)|$ en {len(x_test)} puntos del intervalo $[{a}, {b}]$: $$\\max_{{x \\in [{a},{b}]}} |f\'\'(x)| \\approx {max_f_segunda:.6f}$$'
                        })
                        
                        error_teorico_trap = abs((b - a)**3 / (12 * n**2) * max_f_segunda)
                        pasos.append({
                            'descripcion': 'Cálculo del error teórico del trapecio',
                            'calculo': f'$$E_{{\\text{{trapecio}}}} \\leq \\frac{{({b}-{a})^3}}{{12 \\times {n}^2}} \\times {max_f_segunda:.6f} = \\frac{{{(b-a)**3:.6f}}}{{12 \\times {n**2}}} \\times {max_f_segunda:.6f} = {error_teorico_trap:.8f}$$'
                        })
                        
                        error_info['theoretical_error'] = error_teorico_trap
                        error_info['order'] = 'O(h²)'
                        
                        pasos.append({
                            'descripcion': 'Comparación de errores',
                            'calculo': f'$$\\begin{{align}} \\text{{Error real:}} &\\quad {error_absoluto:.8f} \\\\ \\text{{Error teórico estimado:}} &\\quad {error_teorico_trap:.8f} \\\\ \\text{{Orden de convergencia:}} &\\quad O(h^2) \\text{{ donde }} h = {h:.6f} \\end{{align}}$$'
                        })
                        
                    except Exception as e:
                        pasos.append({
                            'descripcion': 'Error en cálculo teórico',
                            'calculo': f'No se pudo calcular el error teórico: {str(e)}'
                        })
                
                elif method == 'simpson':
                    try:
                        pasos.append({
                            'descripcion': 'Fórmula del error teórico para Simpson',
                            'calculo': f'Para la regla de Simpson, el error teórico está dado por: $$E_{{\\text{{Simpson}}}} \\leq \\frac{{(b-a)^5}}{{180n^4}} \\max_{{x \\in [a,b]}} |f^{{(4)}}(x)|$$'
                        })
                        
                        # Calcular cuarta derivada
                        cuarta_derivada = sp.diff(expr, x, 4)
                        pasos.append({
                            'descripcion': 'Cálculo de la cuarta derivada',
                            'calculo': f'$$f^{{(4)}}(x) = \\frac{{d^4}}{{dx^4}}({expr}) = {sp.latex(cuarta_derivada)}$$'
                        })
                        
                        x_test = np.linspace(a, b, 100)
                        f_cuarta_vals = [abs(float(sp.N(cuarta_derivada.subs(x, xi)))) for xi in x_test]
                        max_f_cuarta = max(f_cuarta_vals)
                        
                        pasos.append({
                            'descripcion': 'Estimación del máximo de |f⁽⁴⁾(x)|',
                            'calculo': f'Evaluando $|f^{{(4)}}(x)|$ en {len(x_test)} puntos del intervalo $[{a}, {b}]$: $$\\max_{{x \\in [{a},{b}]}} |f^{{(4)}}(x)| \\approx {max_f_cuarta:.6f}$$'
                        })
                        
                        error_teorico_simpson = abs((b - a)**5 / (180 * n**4) * max_f_cuarta)
                        pasos.append({
                            'descripcion': 'Cálculo del error teórico de Simpson',
                            'calculo': f'$$E_{{\\text{{Simpson}}}} \\leq \\frac{{({b}-{a})^5}}{{180 \\times {n}^4}} \\times {max_f_cuarta:.6f} = \\frac{{{(b-a)**5:.6f}}}{{180 \\times {n**4}}} \\times {max_f_cuarta:.6f} = {error_teorico_simpson:.8f}$$'
                        })
                        
                        error_info['theoretical_error'] = error_teorico_simpson
                        error_info['order'] = 'O(h⁴)'
                        
                        pasos.append({
                            'descripcion': 'Comparación de errores',
                            'calculo': f'$$\\begin{{align}} \\text{{Error real:}} &\\quad {error_absoluto:.8f} \\\\ \\text{{Error teórico estimado:}} &\\quad {error_teorico_simpson:.8f} \\\\ \\text{{Orden de convergencia:}} &\\quad O(h^4) \\text{{ donde }} h = {h:.6f} \\end{{align}}$$'
                        })
                        
                    except Exception as e:
                        pasos.append({
                            'descripcion': 'Error en cálculo teórico',
                            'calculo': f'No se pudo calcular el error teórico: {str(e)}'
                        })
                
                elif method == 'midpoint':
                    try:
                        pasos.append({
                            'descripcion': 'Fórmula del error teórico para Punto Medio',
                            'calculo': f'Para la regla del punto medio, el error teórico está dado por: $$E_{{\\text{{punto medio}}}} \\leq \\frac{{(b-a)^3}}{{24n^2}} \\max_{{x \\in [a,b]}} |f\'\'(x)|$$'
                        })
                        
                        # Calcular segunda derivada (igual que trapecio pero diferente constante)
                        segunda_derivada = sp.diff(expr, x, 2)
                        pasos.append({
                            'descripcion': 'Cálculo de la segunda derivada',
                            'calculo': f'$$f\'\'(x) = \\frac{{d^2}}{{dx^2}}({expr}) = {sp.latex(segunda_derivada)}$$'
                        })
                        
                        x_test = np.linspace(a, b, 100)
                        f_segunda_vals = [abs(float(sp.N(segunda_derivada.subs(x, xi)))) for xi in x_test]
                        max_f_segunda = max(f_segunda_vals)
                        
                        pasos.append({
                            'descripcion': 'Estimación del máximo de |f\'\'(x)|',
                            'calculo': f'Evaluando $|f\'\'(x)|$ en {len(x_test)} puntos del intervalo $[{a}, {b}]$: $$\\max_{{x \\in [{a},{b}]}} |f\'\'(x)| \\approx {max_f_segunda:.6f}$$'
                        })
                        
                        error_teorico_mid = abs((b - a)**3 / (24 * n**2) * max_f_segunda)
                        pasos.append({
                            'descripcion': 'Cálculo del error teórico del punto medio',
                            'calculo': f'$$E_{{\\text{{punto medio}}}} \\leq \\frac{{({b}-{a})^3}}{{24 \\times {n}^2}} \\times {max_f_segunda:.6f} = \\frac{{{(b-a)**3:.6f}}}{{24 \\times {n**2}}} \\times {max_f_segunda:.6f} = {error_teorico_mid:.8f}$$'
                        })
                        
                        error_info['theoretical_error'] = error_teorico_mid
                        error_info['order'] = 'O(h²)'
                        
                        pasos.append({
                            'descripcion': 'Comparación de errores',
                            'calculo': f'$$\\begin{{align}} \\text{{Error real:}} &\\quad {error_absoluto:.8f} \\\\ \\text{{Error teórico estimado:}} &\\quad {error_teorico_mid:.8f} \\\\ \\text{{Orden de convergencia:}} &\\quad O(h^2) \\text{{ donde }} h = {h:.6f} \\end{{align}}$$'
                        })
                        
                    except Exception as e:
                        pasos.append({
                            'descripcion': 'Error en cálculo teórico',
                            'calculo': f'No se pudo calcular el error teórico: {str(e)}'
                        })
                
        except Exception as e:
            # Si no se puede calcular la integral exacta, agregar nota explicativa
            pasos.append({
                'descripcion': 'Integral exacta no disponible',
                'calculo': f'No se pudo calcular la integral exacta simbólicamente para esta función. Esto puede deberse a que la función no tiene una antiderivada en forma cerrada o la expresión es demasiado compleja para el cálculo simbólico. **Resultado numérico:** ${resultado:.8f}$'
            })
            error = None
        
        return resultado, pasos, error, func, expr, x_vals, f_vals, error_info
        
    except Exception as e:
        raise ValueError(f"Error en el cálculo de la integral: {str(e)}")

# =============================================================================
# FUNCIONES DE EDOs
# =============================================================================

def euler_method(func, t0, y0, h, n_steps):
    """Método de Euler para EDOs"""
    t_vals = [t0]
    y_vals = [y0]
    pasos = []
    
    for i in range(n_steps):
        t_n = t_vals[-1]
        y_n = y_vals[-1]
        
        # Calcular siguiente punto
        f_val = func(t_n, y_n)
        y_next = y_n + h * f_val
        t_next = t_n + h
        
        # Calcular cambio en y (siempre el cambio real)
        delta_y = y_next - y_n
        
        pasos.append({
            'step': i + 1,
            'tn': t_next,  # Tiempo al que llegamos después del paso
            'yn': y_next,  # Valor al que llegamos después del paso
            'f_tn_yn': f_val,  # f(t_n, y_n) usado para calcular el paso
            'yn_plus_1': y_next,  # Redundante, pero mantenido por compatibilidad
            'delta_y': delta_y,  # Cambio real en y
            # Valores del punto de partida del paso
            't_inicial': t_n,
            'y_inicial': y_n
        })
        
        t_vals.append(t_next)
        y_vals.append(y_next)
    
    return t_vals, y_vals, pasos

def rk4_method(func, t0, y0, h, n_steps):
    """Método Runge-Kutta de 4to orden para EDOs"""
    t_vals = [t0]
    y_vals = [y0]
    pasos = []
    
    for i in range(n_steps):
        t_n = t_vals[-1]
        y_n = y_vals[-1]
        
        # Calcular k1, k2, k3, k4
        k1 = func(t_n, y_n)
        k2 = func(t_n + h/2, y_n + h*k1/2)
        k3 = func(t_n + h/2, y_n + h*k2/2)
        k4 = func(t_n + h, y_n + h*k3)
        
        # Calcular siguiente punto
        y_next = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t_n + h
        
        # Calcular cambio en y (siempre el cambio real)
        delta_y = y_next - y_n
        
        pasos.append({
            'step': i + 1,
            'tn': t_next,  # Tiempo al que llegamos después del paso
            'yn': y_next,  # Valor al que llegamos después del paso
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'k4': k4,
            'yn_plus_1': y_next,  # Redundante, pero mantenido por compatibilidad
            'delta_y': delta_y,  # Cambio real en y
            # Valores del punto de partida del paso
            't_inicial': t_n,
            'y_inicial': y_n
        })
        
        t_vals.append(t_next)
        y_vals.append(y_next)
    
    return t_vals, y_vals, pasos

def calcular_analisis_edo(pasos, entrada, valor_final, tiempo_final):
    """
    Calcula análisis estadístico detallado de la solución EDO
    """
    import numpy as np
    
    # Extraer valores y tiempos (incluir punto inicial + todos los pasos)
    tiempos = [entrada.t0] + [p['tn'] for p in pasos]
    valores = [entrada.y0] + [p['yn'] for p in pasos]
    
    # Estadísticas básicas
    min_valor = min(valores)
    max_valor = max(valores)
    min_idx = valores.index(min_valor)
    max_idx = valores.index(max_valor)
    min_tiempo = tiempos[min_idx]
    max_tiempo = tiempos[max_idx]
    
    promedio = np.mean(valores)
    desviacion = np.std(valores, ddof=1) if len(valores) > 1 else 0
    
    # Cambios y tendencias
    cambio_absoluto = valor_final - entrada.y0
    cambio_relativo = (cambio_absoluto / entrada.y0 * 100) if entrada.y0 != 0 else 0
    incremento_promedio = cambio_absoluto / entrada.n_steps
    
    # Tendencia general
    if abs(cambio_absoluto) > 0.01:
        tendencia = "Creciente" if cambio_absoluto > 0 else "Decreciente"
    else:
        tendencia = "Estable"
    
    # Análisis de precisión
    rango_temporal = tiempo_final - entrada.t0
    precision_estimada = f"10^{-len(str(entrada.h).split('.')[-1])}" if '.' in str(entrada.h) else "10^-1"
    decimales_precision = len(str(entrada.h).split('.')[-1]) if '.' in str(entrada.h) else 1
    
    return {
        'min_valor': min_valor,
        'max_valor': max_valor,
        'min_tiempo': min_tiempo,
        'max_tiempo': max_tiempo,
        'promedio': promedio,
        'desviacion': desviacion,
        'cambio_absoluto': cambio_absoluto,
        'cambio_relativo': cambio_relativo,
        'incremento_promedio': incremento_promedio,
        'tendencia': tendencia,
        'rango_temporal': rango_temporal,
        'precision_estimada': precision_estimada,
        'decimales_precision': decimales_precision,
        # Nuevos campos para evitar cálculos en template
        'valor_inicial': entrada.y0,
        'valor_final': valor_final,
        'cambio_absoluto_formula': f"{valor_final:.6f} - {entrada.y0} = {cambio_absoluto:.6f}",
        'incremento_formula_num': f"{cambio_absoluto:.6f}",
        'incremento_formula_den': entrada.n_steps,
        'total_puntos': len(valores)
    }

def calcular_edo(function_str, t0, y0, h, n_steps, method):
    """
    Calcula la solución de una EDO usando método de Euler o Runge-Kutta
    
    Returns:
        tuple: (t_vals, y_vals, pasos, func, expr, analisis)
    """
    try:
        # Parsear y preparar función
        expr_original = function_str.strip()
        expr_for_calc = expr_original.replace('y', 'y_var').replace('t', 't_var')
        
        # Crear expresión legible para mostrar
        expr_display = expr_original.replace('**', '^')
        
        # Definir función lambda
        func = lambda t_var, y_var: eval(expr_for_calc)
        
        # Crear objeto de entrada para análisis
        from types import SimpleNamespace
        entrada = SimpleNamespace()
        entrada.t0 = t0
        entrada.y0 = y0
        entrada.h = h
        entrada.n_steps = n_steps
        
        # Calcular solución según método
        if method == 'euler':
            t_vals, y_vals, pasos = euler_method(func, t0, y0, h, n_steps)
        elif method == 'rk4':
            t_vals, y_vals, pasos = rk4_method(func, t0, y0, h, n_steps)
        else:
            raise ValueError(f"Método no reconocido: {method}")
        
        # Calcular análisis estadístico
        analisis = calcular_analisis_edo(pasos, entrada, y_vals[-1], t_vals[-1])
        
        return t_vals, y_vals, pasos, func, expr_display, analisis
        
    except Exception as e:
        raise ValueError(f"Error en el cálculo de la EDO: {str(e)}")

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def generar_grafico_derivada(func, expr, x0, h, method, resultado):
    """Genera gráfico para derivación numérica"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Rango para graficar la función
        x_range = np.linspace(x0 - 3*h, x0 + 3*h, 200)
        y_range = [func(x) for x in x_range]
        
        # Graficar función
        ax.plot(x_range, y_range, 'b-', linewidth=2, label=f'f(x) = {expr}')
        
        # Marcar puntos relevantes según el método
        if method == 'forward':
            ax.plot([x0, x0+h], [func(x0), func(x0+h)], 'ro-', markersize=8, linewidth=2, label='Secante hacia adelante')
            ax.plot(x0, func(x0), 'ro', markersize=10)
            ax.plot(x0+h, func(x0+h), 'ro', markersize=10)
        elif method == 'backward':
            ax.plot([x0-h, x0], [func(x0-h), func(x0)], 'go-', markersize=8, linewidth=2, label='Secante hacia atrás')
            ax.plot(x0-h, func(x0-h), 'go', markersize=10)
            ax.plot(x0, func(x0), 'go', markersize=10)
        else:  # central
            ax.plot([x0-h, x0+h], [func(x0-h), func(x0+h)], 'mo-', markersize=8, linewidth=2, label='Secante central')
            ax.plot(x0-h, func(x0-h), 'mo', markersize=10)
            ax.plot(x0+h, func(x0+h), 'mo', markersize=10)
            ax.plot(x0, func(x0), 'ko', markersize=8, label=f'x₀ = {x0}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Derivación Numérica - Método: {method.title()}\nf\'({x0}) ≈ {resultado:.6f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Guardar
        filename = f"derivada_{method}_{uuid.uuid4().hex[:8]}.png"
        filepath = IMAGES_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"images/{filename}"
        
    except Exception as e:
        plt.close()
        print(f"Error generando gráfico de derivada: {e}")
        return None

def generar_grafico_integral(func, expr, a, b, x_vals, f_vals, method, resultado):
    """Genera gráfico para integración numérica"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Función continua
        x_cont = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 500)
        y_cont = [func(x) for x in x_cont]
        ax.plot(x_cont, y_cont, 'b-', linewidth=2, label=f'f(x) = {expr}')
        
        # Área bajo la curva
        x_fill = np.linspace(a, b, 200)
        y_fill = [func(x) for x in x_fill]
        ax.fill_between(x_fill, y_fill, alpha=0.3, color='lightblue', label='Área exacta')
        
        if method == 'trapezoidal':
            # Dibujar trapecios
            for i in range(len(x_vals)-1):
                x_trap = [x_vals[i], x_vals[i+1], x_vals[i+1], x_vals[i]]
                y_trap = [0, 0, f_vals[i+1], f_vals[i]]
                ax.fill(x_trap, y_trap, alpha=0.5, color='red', edgecolor='black', linewidth=1)
            
            ax.plot(x_vals, f_vals, 'ro-', markersize=6, linewidth=2, label='Aproximación trapezoidal')
            
        else:  # simpson
            # Para Simpson, dibujar parábolas (simplificado)
            ax.plot(x_vals, f_vals, 'go-', markersize=6, linewidth=2, label='Puntos de Simpson')
            
            # Rellenar área aproximada
            for i in range(0, len(x_vals)-1, 2):
                if i+2 < len(x_vals):
                    x_seg = np.linspace(x_vals[i], x_vals[i+2], 50)
                    # Interpolación parabólica simple para visualizar
                    y_seg = [func(x) for x in x_seg]
                    ax.fill_between(x_seg, y_seg, alpha=0.3, color='green')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Integración Numérica - {method.title()}\n∫f(x)dx ≈ {resultado:.6f} en [{a}, {b}]')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Guardar
        filename = f"integral_{method}_{uuid.uuid4().hex[:8]}.png"
        filepath = IMAGES_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"images/{filename}"
        
    except Exception as e:
        plt.close()
        print(f"Error generando gráfico de integral: {e}")
        return None

def generar_grafico_edo(t_vals, y_vals, expr, method, t0, y0):
    """Genera gráfico para EDOs"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Solución numérica
        ax.plot(t_vals, y_vals, 'bo-', markersize=6, linewidth=2, label=f'Solución {method.upper()}')
        ax.plot(t0, y0, 'ro', markersize=10, label=f'Condición inicial ({t0}, {y0})')
        
        ax.set_xlabel('t')
        ax.set_ylabel('y(t)')
        ax.set_title(f'EDO: dy/dt = {expr}\nMétodo: {method.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Guardar
        filename = f"edo_{method}_{uuid.uuid4().hex[:8]}.png"
        filepath = IMAGES_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"images/{filename}"
        
    except Exception as e:
        plt.close()
        print(f"Error generando gráfico de EDO: {e}")
        return None

def generate_operation_pdf(result_data, operation_type):
    """
    Genera un PDF a partir de los datos de una operación
    
    Args:
        result_data (dict): Datos de resultado de la operación
        operation_type (str): Tipo de operación ('sistemas_lineales', 'interpolacion', etc.)
    
    Returns:
        str: Ruta relativa al PDF generado
    """
    # Crear un identificador único para el archivo
    file_uuid = uuid.uuid4().hex
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{operation_type}_{timestamp}_{file_uuid}.pdf"
    
    # Preparar el contexto para la plantilla
    # Convertir result_data a un diccionario si es necesario
    result_dict = {}
    if hasattr(result_data, '__dict__'):
        # Si result_data es un objeto con atributos, convertirlo a dict
        result_dict = {key: value for key, value in result_data.__dict__.items()}
    elif isinstance(result_data, dict):
        # Si ya es un dict, usarlo directamente
        result_dict = result_data
    
    context = {
        'result': result_dict,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'operation_type': operation_type
    }
    
    # Renderizar HTML según el tipo de operación
    template_name = f"pdf/{operation_type}_pdf.html"
    html_string = render_to_string(template_name, context)
    
    # Crear PDF con WeasyPrint
    pdf_path = PDF_DIR / filename
    
    # Establecer CSS para el PDF (incluye Tailwind básico para estilos)
    css = CSS(string='''
        body { font-family: sans-serif; margin: 2cm; }
        h1 { color: #4f46e5; font-size: 24px; margin-bottom: 20px; }
        h2 { color: #374151; font-size: 18px; margin-top: 15px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
        th { background-color: #f9fafb; }
        img { max-width: 100%; height: auto; }
        .container { max-width: 800px; margin: 0 auto; }
        .footer { margin-top: 30px; font-size: 12px; color: #6b7280; text-align: center; }
    ''')
    
    HTML(string=html_string).write_pdf(pdf_path, stylesheets=[css])
    
    # Devolver la ruta relativa para el acceso desde la web
    rel_path = f"pdfs/{filename}"
    return rel_path

def save_to_history(request, operation_type, method, input_data, result_data):
    """
    Guarda una operación en el historial de la sesión
    
    Args:
        request: Objeto request de Django
        operation_type (str): Tipo de operación ('sistemas_lineales', 'interpolacion', etc.)
        method (str): Método específico utilizado 
        input_data (dict): Datos de entrada
        result_data (dict): Resultado de la operación
    """
    # Generar PDF
    pdf_path = generate_operation_pdf(result_data, operation_type)
    
    # Inicializar historial si no existe
    if 'historial' not in request.session:
        request.session['historial'] = []
    
    # Obtener la ruta del gráfico, de forma segura para diferentes tipos de objetos (dict o objeto personalizado)
    grafico_path = None
    if hasattr(result_data, 'grafico_path'):
        grafico_path = result_data.grafico_path
    elif isinstance(result_data, dict) and 'grafico_path' in result_data:
        grafico_path = result_data['grafico_path']
    
    # Crear entrada de historial
    historial_item = {
        'tipo': operation_type,
        'metodo': method,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'entrada': input_data,
        'resultado_resumen': _get_result_summary(result_data, operation_type),
        'grafico_path': grafico_path,
        'pdf_path': pdf_path
    }
    
    # Añadir al historial
    request.session['historial'].append(historial_item)
    request.session.modified = True
    
    return historial_item

def _get_result_summary(result_data, operation_type):
    """
    Genera un resumen del resultado para mostrar en el historial
    """
    if operation_type == 'sistemas_lineales':
        if 'solution' in result_data:
            return {'solucion': result_data['solution'][:3]}  # Primeros 3 valores
        return {'info': 'Sistema resuelto'}
        
    elif operation_type == 'interpolacion':
        if hasattr(result_data, 'polinomio_tex'):
            return {'polinomio': result_data.polinomio_tex[:50] + '...' if len(result_data.polinomio_tex) > 50 else result_data.polinomio_tex}
        return {'info': 'Interpolación calculada'}
        
    return {'info': 'Operación completada'}

# Add this helper function if the file contains LaTeX rendering code for matrices
def fix_nested_matrices(latex_str):
    """
    Fixes the issue with nested matrices in LaTeX representation.
    
    Args:
        latex_str: LaTeX string that might contain nested matrices
        
    Returns:
        Corrected LaTeX string with single matrix
    """
    if not latex_str:
        return latex_str
        
    # Remove nested bmatrix environments wherever they appear
    if "\\begin{bmatrix}\\begin{bmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{bmatrix}\\begin{bmatrix}", "\\begin{bmatrix}")
        latex_str = latex_str.replace("\\end{bmatrix}\\end{bmatrix}", "\\end{bmatrix}")
    
    # Remove nested pmatrix environments wherever they appear
    if "\\begin{pmatrix}\\begin{pmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{pmatrix}\\begin{pmatrix}", "\\begin{pmatrix}")
        latex_str = latex_str.replace("\\end{pmatrix}\\end{pmatrix}", "\\end{pmatrix}")
    
    # Also handle potential cases with mixed environments
    if "\\begin{bmatrix}\\begin{pmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{bmatrix}\\begin{pmatrix}", "\\begin{bmatrix}")
        latex_str = latex_str.replace("\\end{pmatrix}\\end{bmatrix}", "\\end{bmatrix}")
    
    if "\\begin{pmatrix}\\begin{bmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{pmatrix}\\begin{bmatrix}", "\\begin{pmatrix}")
        latex_str = latex_str.replace("\\end{bmatrix}\\end{pmatrix}", "\\end{pmatrix}")
    
    return latex_str 