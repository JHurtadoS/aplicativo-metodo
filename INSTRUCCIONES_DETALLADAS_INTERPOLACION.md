
# INSTRUCCIONES_DETALLADAS_INTERPOLACION.md
> **Objetivo:** guiar a un LLM/ferramenta de scaffolding (p. ej. Cursor) para generar *por completo* el módulo **`interpolacion`** del proyecto “Métodos Numéricos”.

---

## 0 · Convenciones básicas que el LLM **debe respetar**

| Aspecto | Valor obligatorio |
|---------|-------------------|
| **Stack** | Django 4 · Python 3.10 · NumPy + SymPy · Matplotlib (Agg) |
| **Frontend** | Preact 10 + Tailwind CDN (sin bundler) |
| **Formato salida** | Español puro; comentarios y doc‑strings igualmente en español |
| **Reuse** | Toda resolución de sistemas lineales **DEBE** llamar a `sistemas_lineales.algorithms.solve_system()` (no se acepta `numpy.linalg.solve`). |
| **Persistencia** | Ningún modelo nuevo; historial vía `request.session` y `localStorage`. |
| **Licencia** | Encabezado `#  ProyectoMN – Licencia IMT` en cada archivo creado. |

---

## 1 · Archivos y rutas a crear / modificar

| Ruta relativa | Tipo | Descripción exacta |
|---------------|------|--------------------|
| `interpolacion/__init__.py` | code | Dejar vacío o importar `.algorithms` |
| `interpolacion/algorithms/__init__.py` | code | Reexportar `lagrange`, `newton`, `builder_vandermonde`, `builder_newton_triangular`. |
| `interpolacion/algorithms/builder.py` | code | Contendrá las dos funciones *puras* de construcción de matrices (ver §2). |
| `interpolacion/algorithms/core.py` | code | Implementa `lagrange()` y `newton()` y empaqueta el `InterpResult`. |
| `interpolacion/forms.py` | code | Django Form con campos `puntos` (Textarea) y `metodo` (Choice). |
| `interpolacion/views.py` | code | Vista `form(request)` y helper `render_result(request, result)`. |
| `interpolacion/urls.py` | code | Path `''` → `form` |
| `templates/interpolacion/form.html` | template | Formulario con Tailwind y montaje Preact. |
| `templates/interpolacion/result.html` | template | Muestra pasos, polinomio (MathJax) y gráfica PNG. |
| `static/js/interpolacion_app.js` | JS module | Componente `InterpolacionApp` (recibe `window.INTERP_CTX` JSON). |
| **NO tocar** | — | `sistemas_lineales`; sólo importar su solver. |

---

## 2 · Builders de matrices – especificación de firma y tests internos

```python
# builder.py
from typing import List, Tuple
import numpy as np

def builder_vandermonde(points: List[Tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (A, b) para el sistema lineal que produce el polinomio interpolante de Lagrange."""
    ...

def builder_newton_triangular(points: List[Tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (T, b) donde T es triangular inferior para el polinomio de Newton."""
    ...
```

*Requisitos funcionales*  
1. Lanza `ValueError` si `len(points) < 2` o existen `x` duplicados.  
2. Devuelve `A.shape == (n,n)` y `b.shape == (n,)`.  
3. `np.allclose(A, np.tril(A))` **solo** debe ser `True` para el builder de Newton.

*Auto‑test mínimo dentro del archivo* (para que Cursor lo genere):

```python
if __name__ == "__main__":
    pts = [(1,2),(2,5),(3,10)]
    Av,bv = builder_vandermonde(pts)
    At,bn = builder_newton_triangular(pts)
    assert Av.shape == (3,3) and bn.shape == (3,)
    print("✅  builders ok")
```

---

## 3 · Funciones de alto nivel

```python
# core.py
from dataclasses import dataclass
from ....result import Result   #  import relativo a la clase base

@dataclass
class InterpResult(Result):
    coeficientes: list[float]
    polinomio_tex: str

def lagrange(points, decimales: int = 6, solver: str = "gauss") -> InterpResult:
    """Calcula el polinomio interpolante de Lagrange usando el solver indicado."""

def newton(points, decimales: int = 6, solver: str = "gauss") -> InterpResult:
    """Calcula el polinomio interpolante de Newton (diferencias divididas)."""
```

**Obligaciones internas**  
- `solver` se pasa directo a `solve_system()`.  
- Después de resolver, construir `sympy.Poly` y generar `tex`.  
- Llamar `plot_points_curve(points, poly, uuid)` que ya existe en `core.utils`.  
- Rellenar todas las claves de `InterpResult`.

---

## 4 · Vista Django paso a paso

1. **GET** → renderiza `form.html`.  
2. **POST** →  
   - Parsear `request.POST["puntos"]` → `[(x,y), …]`.  
   - Llamar la función adecuada (`lagrange` o `newton`).  
   - Guardar resultado en `request.session["historial"]` (append).  
   - Renderizar `result.html` con contexto:

```python
{
  "result": result,
  "poly_tex": result.polinomio_tex,
  "pasos": result.pasos,
  "grafico": result.grafico_path
}
```

3. Botones: **“Exportar JSON”** y **“Regresar”**.

---

## 5 · Template `result.html` – estructura obligatoria

```html
{% extends "base.html" %}
{% block content %}
  <h1 class="text-2xl font-bold">{{ result.metodo }}</h1>

  <!-- Polinomio -->
  <div id="poly" class="my-4">
    $$ {{ poly_tex }} $$
  </div>

  <!-- Gráfica -->
  {% if grafico %}
    <img src="{{ grafico }}" alt="Curva interpolante" class="max-w-md mx-auto">
  {% endif %}

  <!-- Pasos -->
  <pre class="font-mono bg-slate-800 text-white p-4 rounded">
{{ pasos|safe }}
  </pre>

  <button id="btn-json" ...>Exportar</button>
{% endblock %}
<script src="https://polyfill.io/mathjax.js"></script>
```

---

## 6 · Checklist de verificación **(actualizar `SETUP_CHECKLIST.md`)**

- [ ] `python interpolacion/algorithms/builder.py` imprime “✅ builders ok”.  
- [ ] En `/interpolacion` Lagrange y Newton retornan el **mismo** valor cuando se evalúa en cada \(x_i\).  
- [ ] El solver indicado se refleja en JSON (`"solver":"gauss"` ó `"lu"`).  
- [ ] Historial muestra tipo “Lagrange” y enlaza a descarga.  
- [ ] Código duplicado inexistente (`grep -R "builder_vander" sistemas_lineales` debe devolver 0 líneas).

---

## 7 · Refactor & reutilización

Si el LLM detecta funciones idénticas en `lagrange`/`newton`:

> **Acción:** Factorizar en `interpolacion/algorithms/_common.py` y que ambas las importen.

Mantener ***Single Responsibility***:  
- Builders → matrices,  
- Core → lógica del método,  
- Vistas → interfaz/serialización,  
- Templates → presentación.

---

## 8 · Tiempo estimado de generación

| Tarea | Tiempo (LLM) |
|-------|--------------|
| Crear archivos & builders | 20 s |
| Core + dataclass `InterpResult` | 15 s |
| Form & Vista | 30 s |
| Templates | 25 s |
| Checklist update | 5 s |
| **Total** | **< 2 min** |

*(Fin de archivo)*
