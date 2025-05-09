# Guía de Conceptos Numéricos y su Implementación en la App

Esta guía resume **qué hace cada método numérico**, **por qué lo necesitamos** y **cómo se integra en la arquitectura Django + Preact definida**. Sirve como manual interno para cualquier desarrollador que extienda o mantenga la aplicación.

---

## 1 · Sistemas de Ecuaciones Lineales

| Método | Fundamento matemático | Datos que requiere | Pasos clave en la app | Artefactos que genera |
|--------|----------------------|--------------------|-----------------------|-----------------------|
| **Eliminación Gaussiana** | Operaciones elementales por filas → matriz triangular superior (*forward*) y sustitución regresiva. | Matriz **A** (n×n) y vector **b**. | • `algorithms.gauss(A, b)` devuelve todas las matrices aumentadas etapa a etapa.<br>• Vista formatea cada matriz en tabla HTML; pivoteos mostrados. | • Lista `pasos` con matrices.<br>• Solución **x**. |
| **LU – Doolittle/Crout** | Factorizar `A = L·U`, luego resolver `Ly=b` y `Ux=y`. | A y b. | • `algorithms.lu(A)` produce L y U.<br>• Pasos: cálculo de cada fila/columna.<br>• Mostrar L, U y sustituciones. | • L y U en SymPy para latex.<br>• Solución final. |
| **Jacobi** | Iterativo: `x^{k+1} = D^{-1}(b - (L+U)x^k)` | A, b, vector inicial, tol, iter max | • Función retorna tabla de iteraciones + error por paso.<br>• Preact muestra tabla y grafica error vs k. | • PNG error‑convergencia. |
| **Gauss‑Seidel** | Igual que Jacobi pero usa valores actualizados in‑place. | Idem Jacobi | Misma interfaz; convergencia suele ser más rápida. | |


### Integración práctica
* **Entrada**: textarea → parser crea `numpy.ndarray` (matriz) y `np.array` (vector).
* **Precisión**: al presentar matrices se usa `sympy.Matrix(matrix).rational_approximation(max_den=10)` si el usuario activa modo "exacto".
* **Gráficas** (iterativos): función `plot_error(errors, uuid)` guarda PNG en `static/img/`.

---

## 2 · Interpolación y Ajuste de Curvas

| Método | Idea central | Datos | Implementación | Salida / Visual |
|--------|--------------|-------|----------------|-----------|
| **Lagrange** | Combina polinomios base `L_i(x)` para pasar por cada punto. | Lista de (x,y) únicos. | • Construir cada `L_i` simbólicamente con SymPy.<br>• Evaluar/expandir para polinomio final. | • String LaTeX del polinomio.<br>• PNG con curva + puntos. |
| **Newton (dif. divididas)** | Polinomio incremental con coeficientes de la tabla de diferencias. | Lista ordenada de (x,y). | • Función genera tabla (`pasos`).<br>• Coefs `a_k`→ polinomio. | • Tabla HTML.<br>• Misma gráfica que Lagrange (verificación). |
| **Splines Cúbicos** | Construir un polinomio cúbico por intervalo, con continuidad C2 en los nodos. | Lista de (x,y) únicos, mín. 3 puntos. | • Construir sistema 4n×4n para continuidad.<br>• Resolver para coefs a,b,c,d de cada segmento.<br>• Generar representación piecewise. | • Tabla de coeficientes por intervalo.<br>• Fórmula LaTeX en casos.<br>• PNG con curvas por tramo. |
| **Regresión lineal** | Minimiza Σ (y − (ax+b))². | Puntos (x,y). | Calcular sumas, resolver `a`, `b`. | Recta en forma `y=ax+b`, R² opcional, PNG. |

*El campo "grado polinomio" para regresión polinómica se deja `TODO` Corte 3.*

---

## 3 · Diferenciación e Integración (para Corte 3)

| Método | Fórmula base | API planificada |
|--------|--------------|-----------------|
| Diferencias centradas | f'(x₀) ≈ (f(x₀+h)−f(x₀−h))/(2h) | `derivar(f_expr, x0, h, orden=1)` retorna valor y pasos. |
| Trapecio / Simpson | Ver tabla | `integrar(f_expr, a, b, n, metodo)` devuelve valor, tabla y gráfica (áreas). |

---

## 4 · EDO – Euler / RK4 (para Corte 3)

| Método | Ecuación | Algoritmo | Presentación |
|--------|----------|-----------|--------------|
| Euler | y' = f(t,y) | y_{n+1}=y_n+h f(t_n,y_n) | Tabla iteraciones, PNG y(t). |
| RK4 | idem | k₁–k₄, y_{n+1}=y_n+… | Tabla con k's, PNG. |

---

## 5 · Control de precisión y redondeo

* **Parámetro UI "decimales"** (select 2‑10) → formatea números con `round(x, dec)` antes de mostrar.  
* **Análisis de sensibilidad**: al cambiar decimales, recalcular resultado y mostrar "Δ resultado".  
  Ej.: sistemas lineales → mostrar norma(||x_full − x_red||).

---

## 6 · Exportación JSON

```json
{
  "tipo": "Sistema lineal – Gauss",
  "entrada": { "A": [[...]], "b":[...] },
  "pasos": [ { "matriz": [[...]] }, ... ],
  "salida": { "x":[...], "iter": null },
  "grafico": "/static/img/abc123.png",
  "timestamp": "2025‑05‑05T12:34:56"
}
```

---

## 7 · Flujo de integración global

1. **Preact** envía formulario → Django view.  
2. **View** valida, llama algoritmo ➜ `Result`.  
3. Si hay gráfica, view llama `plot_*`, guarda PNG, setea `Result.grafico_path`.  
4. View guarda `Result` en sesión y devuelve HTML (con component `ResultView` montado).  
5. Botón **Exportar** lee `Result` y llama `/export/json/<uuid>` ➜ descarga archivo.  
6. Botón **PDF (C3)** usa mismo HTML y WeasyPrint.

---

## 8 · Extensibilidad

* Añadir método nuevo = función en `algorithms.py` + entrada en dropdown Preact + template parcial.  
* `Result` dataclass garantiza que cualquier método nuevo pueda serializarse y mostrarse sin tocar flujo central. 