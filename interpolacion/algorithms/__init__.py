#  ProyectoMN – Licencia IMT

from .core import lagrange, newton, linear_regression
from .builder import builder_vandermonde, builder_newton_triangular
from .splines import builder_splines_system, natural_cubic_splines

__all__ = [
    'lagrange', 
    'newton', 
    'linear_regression', 
    'builder_vandermonde', 
    'builder_newton_triangular',
    'builder_splines_system',
    'natural_cubic_splines'
] 