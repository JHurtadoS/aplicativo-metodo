# CHANGELOG

## v0.9.1 – 2025-05-06
### Nuevas incorporaciones

- Se crearon plantillas Preact/HTM para los formularios de **Sistemas de Ecuaciones Lineales** e **Interpolación y Ajuste de Curvas** (`templates/sistemas_lineales/form.html` y `templates/interpolacion/form.html`), con contenedores flex y paneles izquierdo/derecho.
- Se actualizaron las vistas en `sistemas_lineales/views.py` e `interpolacion/views.py` para renderizar estos nuevos templates en lugar de la portada genérica.
- Se agregó listado de **Tecnologías usadas** (Django 4.x · Python 3.10 · NumPy ≥1.26 · SymPy ≥1.12 · Matplotlib 3.8 · Preact 10 · HTM 3 · Tailwind 3 · WeasyPrint 61+) en la página de inicio (`templates/core/index.html`) para visibilidad rápida.
- Se incluyeron las siguientes librerías vía CDN en `templates/sistemas_lineales/form.html`:
  - **Chart.js** para gráficas de error.
  - **KaTeX** (y módulo de auto-render) para renderizar expresiones LaTeX en el cliente.
  - **Tabulator** para presentar tablas interactivas de pasos.
- Se añadieron placeholders en el panel de resultados (canvas `#errorChart`, div `#formulaOutput`, div `#stepsTable`) para integrar estas herramientas.

### Referencias
- Documentación general y requerimientos: `Conceptos_Implementacion_MN.md`, `Contexto_Objetivos_MN.md`, `ESPECIFICACION_Corte2.md`, `SETUP_CHECKLIST.md`. 