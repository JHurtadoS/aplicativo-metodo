
# Especificación Técnica – Aplicación Web de Métodos Numéricos (Corte 2)

> **Versión**: 0.9 – 2025‑03‑05  
> **Autores**: Equipo MN · Licencia IMT

---

## 1 · Stack y convenciones globales
| Capa | Tecnología | Versión / Fuente |
|------|------------|------------------|
| Backend web | **Django** | 4.x |
| Lógica numérica | **Python** 3.10 + NumPy · SymPy | NumPy ≥ 1.26, SymPy ≥ 1.12 |
| Gráficas | Matplotlib (backend **Agg**) | 3.8 |
| Frontend | **Preact 10** (CDN unpkg) – **Tailwind 3** (CDN) | sin empaquetador |
| Exportación | WeasyPrint (HTML→PDF) | 61+ |
| Estática | Imágenes de gráficas PNG guardadas en `static/img/{uuid}.png` |
| Licencia | Instituto Mexicano del Transporte – “LIC‑IMT” |
| Idioma | Español únicamente |

---

## 2 · Mapa de rutas

| URL | Vista Django | Componente Preact raíz | Descripción |
|-----|--------------|------------------------|-------------|
| `/` | `core.views.index` | `MenuApp` | Portada con enlaces a módulos + historial |
| `/sistemas` | `sistemas_lineales.views.form` | `SistemaApp` | Formulario + resultados métodos lineales |
| `/interpolacion` | `interpolacion.views.form` | `InterpolacionApp` | Formulario + resultados interpolación/ajuste |
| `/historial` | `core.views.historial` | `HistorialApp` | Lista de operaciones almacenadas (sesión) |

---

## 3 · Estructura de proyecto

```
ProyectoMN/
├── manage.py
├── ProyectoMN/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── sistemas_lineales/
│   ├── algorithms.py
│   ├── forms.py
│   ├── views.py
│   └── templates/...
├── interpolacion/
│   ├── algorithms.py
│   ├── forms.py
│   ├── views.py
│   └── templates/...
├── core/        # índice + historial + páginas auxiliares
│   └── ...
├── static/
│   ├── img/    # PNGs generados
│   └── css/
└── templates/
    └── base.html
```

---

## 4 · Algoritmos implementados (Corte 2)

| Módulo | Métodos |
|--------|---------|
| **Sistemas lineales** | Gauss con pivoteo parcial · LU (Doolittle/Crout) · Jacobi · Gauss‑Seidel |
| **Interpolación/Ajuste** | Lagrange · Newton (div. divididas) · Regresión lineal |

Cada algoritmo expone una función con firma:

```python
Result solve_xxx(Entrada)  # devuelve dataclass Result
```

`Result` posee:
```python
class Result:
    metodo: str
    entrada: dict
    pasos: list  # objetos serializables
    salida: dict  # solución numérica / polinomio
    grafico_path: str | None  # png en static/img
```

---

## 5 · Interfaz y formularios

* **Matrices / vectores**  
  Se ingresa como *textarea* multi‑línea: cada fila por línea, números separados por espacio.  
  Ejemplo 3×3:  
  ```
  2 1 -1 | 8
  -3 -1 2 | -11
  -2 1 2 | -3
  ```
  Una barra `|` separa coeficientes de término independiente.

* **Puntos (x, y)**  
  Texto multilinea `x,y` por línea:  
  ```
  1,2
  2,5
  3,10
  ```

* Campos extra según método (tolerancia, iter máx) se muestran con *conditional rendering* en Preact.

---

## 6 · Historial & almacenamiento

* **Servidor** → se guarda cada `Result` en `request.session["historial"]` (lista).
* **Cliente** → botón “Guardar localmente” serializa el mismo objeto `Result` en `localStorage` para persistir tras cerrar sesión (clave `mn_historial`).

---

## 7 · Exportación

* **JSON**: descarga inmediata (`application/json`) con nombre `{slug}_{fecha}.json`.
* **PDF**: renderizado on‑demand con WeasyPrint de la página de resultados exacta (Tailwind inline).

---

## 8 · Cronograma

| Entrega | Ventana | Objetivo mínimo |
|---------|---------|-----------------|
| **Corte 2** | 4‑9 mayo 2025 | Back‑end + UI de `/sistemas` y `/interpolacion`, exportación JSON, historial sesión. |
| **Corte 3** | Fin semestre | Añadir `/integracion` y `/edo`, exportación PDF estable, UI estilizada Tailwind. |

---

## 9 · Notas de implementación rápidas

* Gráfica se guarda con:  
  ```python
  path = f"static/img/{uuid4()}.png"
  plt.savefig(path, dpi=120, bbox_inches="tight")
  ```
* Sin pruebas unitarias; se documentarán casos manuales en el informe.
* Preact se inserta vía
  ```html
  <script src="https://unpkg.com/preact@10/dist/preact.module.js" type="module"></script>
  ```
  y se monta en un `<div id="app"></div>` por cada página; no se usa enrutador SPA.
* Tailwind CDN:  
  ```html
  <script src="https://cdn.tailwindcss.com"></script>
  ```
* Licencia IMT solo incluida en el README y encabezado de `LICENSE`.

---

*(Fin del documento)*
