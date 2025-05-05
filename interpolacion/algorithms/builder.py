#  ProyectoMN – Licencia IMT
from typing import List, Tuple, Sequence
import numpy as np

def builder_vandermonde(points: List[Tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (A, b) para el sistema lineal que produce el polinomio interpolante de Lagrange."""
    if len(points) < 2:
        raise ValueError("Se necesitan al menos 2 puntos para la interpolación.")
    
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    
    # Verificar si hay x duplicados
    if len(x_values) != len(set(x_values)):
        raise ValueError("Los valores de x deben ser únicos.")
    
    n = len(points)
    A = np.zeros((n, n))
    b = np.array(y_values)
    
    # Construir matriz de Vandermonde
    for i in range(n):
        for j in range(n):
            A[i, j] = x_values[i] ** j
    
    return A, b

def builder_newton_triangular(points: List[Tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (T, b) donde T es triangular inferior para el polinomio de Newton."""
    if len(points) < 2:
        raise ValueError("Se necesitan al menos 2 puntos para la interpolación.")
    
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    
    # Verificar si hay x duplicados
    if len(x_values) != len(set(x_values)):
        raise ValueError("Los valores de x deben ser únicos.")
    
    n = len(points)
    T = np.zeros((n, n))
    b = np.array(y_values)
    
    # Construir matriz triangular inferior
    for i in range(n):
        for j in range(i + 1):
            if j == 0:
                T[i, j] = 1
            else:
                # Producto (x - x_0)(x - x_1)...(x - x_{j-1}) evaluado en x_i
                prod = 1
                for k in range(j):
                    prod *= (x_values[i] - x_values[k])
                T[i, j] = prod
    
    return T, b

# Auto-test mínimo
if __name__ == "__main__":
    # Convertir a float para evitar problemas de tipado
    pts = [(1.0, 2.0), (2.0, 5.0), (3.0, 10.0)]
    Av, bv = builder_vandermonde(pts)
    At, bn = builder_newton_triangular(pts)
    assert Av.shape == (3,3) and bn.shape == (3,)
    assert np.allclose(At, np.tril(At)), "La matriz de Newton debe ser triangular inferior"
    assert not np.allclose(Av, np.tril(Av)), "La matriz de Vandermonde no debe ser triangular inferior"
    print("✅ builders ok") 