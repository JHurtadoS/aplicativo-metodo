#  ProyectoMN – Licencia IMT
# Este archivo es solo un wrapper para importar las funciones
# refactorizadas desde el nuevo paquete algorithms/

from .algorithms.core import (
    lagrange,
    newton,
    linear_regression,
    plot_points_curve,
    Result,
    InterpResult
)

from .algorithms.builder import (
    builder_vandermonde,
    builder_newton_triangular
)

# Re-exportar para mantener compatibilidad con código existente
__all__ = [
    'lagrange',
    'newton',
    'linear_regression',
    'plot_points_curve',
    'Result',
    'InterpResult',
    'builder_vandermonde',
    'builder_newton_triangular'
]

# Para compatibilidad con el código antiguo, usamos alias a las funciones refactorizadas
def lagrange_interpolation(points):
    """Wrapper de compatibilidad para la versión antigua"""
    return lagrange(points)

def newton_interpolation(points):
    """Wrapper de compatibilidad para la versión antigua"""
    return newton(points)

# Comentamos el resto del código antiguo. Esto no se ejecutará.
"""
Código antiguo no utilizado:

import numpy as np
import sympy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
from pathlib import Path
...
"""

# Dejar la función linear_regression por compatibilidad
def linear_regression(points):
    """Wrapper de compatibilidad para la versión antigua"""
    # Llamar al linear_regression de algorithms.core (evitando recursión)
    from .algorithms.core import linear_regression as lr_core
    return lr_core(points) 