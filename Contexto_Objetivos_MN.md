
# Contexto Académico y Objetivos – Proyecto «App de Métodos Numéricos»

> **Asignatura**: Métodos Numéricos (Semestre 2025‑1)  
> **Entrega parcial 1 (Corte 2)**: 4 – 9 mayo 2025  
> **Entrega final (Corte 3)**: fin de semestre 2025  
> **Equipo**: … (añadir nombres)  

---

## 1 · Propósito general

Desarrollar una **aplicación web interactiva** que permita a estudiantes:

* Resolver problemas clásicos de métodos numéricos con **procedimiento paso a paso**.  
* Visualizar gráficamente los resultados cuando sea pertinente.  
* Exportar los procedimientos en formatos reutilizables (JSON → PDF en versión final).  

La herramienta sirve de apoyo didáctico, refuerza la comprensión teórica y proporciona evidencia de aprendizaje conforme a la rúbrica oficial del curso.

---

## 2 · Temática del curso y cobertura por entregas

| Tema del programa | Sub‑tópicos relevantes | Entrega en la que se libera |
|-------------------|------------------------|----------------------------|
| **1. Introducción** | tipos de error, motivación | *(solo contenido teórico en documentación)* |
| **2. Ecuaciones no lineales** | bisección, falsa posición, Newton‑Raphson, secante | **Corte 3** (opcional) |
| **3. Sistemas de ecuaciones lineales** | Gauss, LU, Jacobi, Gauss‑Seidel, análisis de convergencia | **Corte 2** |
| **4. Interpolación y ajuste de curvas** | Lagrange, Newton, Splines (futuro), regresión lineal | **Corte 2** |
| **5. Diferenciación & Integración numérica** | diferencias finitas, trapecio, Simpson, cuadratura de Gauss | **Corte 3** |
| **6. Métodos para EDO** | Euler, Euler mejorado, Runge‑Kutta (RK4), orden superior | **Corte 3** |

---

## 3 · Objetivos medibles por corte

### Corte 2
- Backend funcional para **Tema 3** y **Tema 4**.  
- Interfaz mínima: entrada de datos → resultado con pasos.  
- Historial por sesión y exportación a **JSON**.  
- Código modular, comentado y con README + licencia IMT.

### Corte 3
- Añadir **Tema 5** y **Tema 6** con la misma profundidad.  
- Exportación a **PDF** vía WeasyPrint.  
- Interfaz estilizada con Tailwind, experiencia de usuario pulida.  
- Informe técnico completo (plantilla oficial) + video opcional.

---

## 4 · Criterios de evaluación clave (rúbricas)

| Corte | Criterio | Puntuación máxima | Clave para «Alto» |
|-------|----------|-------------------|-------------------|
| 2 | Métodos sistemas lineales | 1.0 | Todos los métodos funcionan y validan entradas |
| 2 | Interpolación/ajuste | 1.0 | Proceso paso a paso y gráfica correcta |
| 2 | Explicación métodos | 1.0 | Texto + ayudas visuales claras |
| 2 | Modularidad del código | 1.0 | Estructura por apps, funciones limpias |
| 2 | Documentación inicial | 1.0 | Cronograma y descripción de fases clara |
| 3 | Derivación/Integración | 1.0 | Todos los métodos + visualización |
| 3 | Métodos EDO | 1.0 | RK4 u orden superior funcional |
| 3 | Interfaz final | 1.0 | Intuitiva, estética con Tailwind |
| 3 | Historial/Exportación | 1.0 | Persiste sesión y exporta |
| 3 | Informe final | 1.0 | Completo, estructurado, con capturas |

---

## 5 · Principios de diseño y alcance

1. **Claridad pedagógica** – La app expone cada paso del algoritmo.  
2. **Simplicidad de despliegue** – Preact + Tailwind vía CDN; sin build JS ni BD persistente.  
3. **Separación de responsabilidades** – Algoritmos puros (Python), vistas Django, componentes UI Preact.  
4. **Escalabilidad mínima** – Código listo para extender módulos en Corte 3 sin refactor masivo.  
5. **Prioridad de tiempo** – Implementaciones manuales solo de métodos requeridos; sin tests unitarios formales.

---

## 6 · Entregables

| Entregable | Formato | Responsable |
|------------|---------|-------------|
| Código fuente | Repo GitHub público | Equipo dev |
| Informe técnico | PDF (WeasyPrint generado) | Redactor |
| README + LIC‑IMT | Markdown | Doc lead |
| Video explicativo (opc.) | ≤ 10 min | Diseño/Comms |
| Aplicación ejecutable | Django runserver o contenedor | DevOps (si aplica) |

---

*(Actualizar este documento conforme se afiancen nuevas decisiones o cambien fechas.)*
