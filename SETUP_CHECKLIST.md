
# SETUP_ENV & CHECKLIST – Proyecto Métodos Numéricos

> **Última actualización**: 2025-04-25

---

## 1 · Instalación rápida

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install --upgrade pip
pip install django~=4.2 numpy sympy matplotlib weasyprint

# 3. Clonar el repo
git clone https://github.com/<usuario>/metodos-numericos-app.git
cd metodos-numericos-app

# 4. Variables de entorno mínimas
export DJANGO_SETTINGS_MODULE=ProyectoMN.settings

# 5. Migraciones (no se usan modelos, pero por formalidad)
python manage.py migrate

# 6. Ejecutar servidor local
python manage.py runserver
# Abrir http://127.0.0.1:8000/ en el navegador
```

> **Nota:** WeasyPrint requiere libffi + cairo + pango en el sistema.  
> En Ubuntu/Debian: `sudo apt install libffi-dev libcairo2 pango1.0-tools`.

---

## 2 · Estructura mínima después de clonar

```text
ProyectoMN/
├── manage.py
├── ProyectoMN/settings.py
├── sistemas_lineales/
├── interpolacion/
├── core/
└── static/img/
```

---

## 3 · Checklist de cumplimiento (Corte 2)

| Ítem | Verificación | ¿Listo? |
|------|--------------|---------|
| **Métodos de sistemas lineales** (Gauss, LU, Jacobi, Gauss‑Seidel) funcionan con validación | Ejecutar casos de prueba en `/sistemas` y obtener solución correcta | ☐ |
| **Interpolación/Ajuste** (Lagrange, Newton, Regresión lineal) muestran proceso + gráfica | Ingresar ≥3 puntos en `/interpolacion` y ver PNG + polinomio | ☐ |
| **Explicación paso a paso** visible en ambas secciones | Revisar que tablas/matrices se desplieguen | ☐ |
| **Historial de sesión** almacena operaciones | Resolver 2 problemas y abrir `/historial` | ☐ |
| **Exportación JSON** descarga archivo válido | Pulsar “Exportar” y abrir `.json` | ☐ |
| **README con licencia IMT** presente en raíz del repo | Ver archivo `README.md` | ☐ |
| **Código modular** (`algorithms.py`, `views.py`, etc.) | Estructura de carpetas coincide con SPEC | ☐ |
| **Cronograma/documentación inicial** en informe | Abrir `informe.pdf` sección cronograma | ☐ |

---

## 4 · Checklist extra (Corte 3)

| Ítem | Verificación | ¿Listo? |
|------|--------------|---------|
| Diferenciación & Integración numérica implementadas | `/integracion` retorna resultados y gráfica | ☐ |
| Métodos EDO (Euler, Heun, RK4) implementados | `/edo` muestra tabla k1‑k4 + curva | ☐ |
| Exportación PDF (WeasyPrint) funcional | Botón “Exportar PDF” genera documento | ☐ |
| Interfaz Tailwind pulida (sin estilos por defecto) | Revisar consistencia visual | ☐ |
| Informe técnico completo | Plantilla llena con capturas | ☐ |

---

### Cómo usar el checklist

1. Marque cada casilla ☐ → ☑ cuando la característica esté verificada.  
2. Adjunte checklist completado en el informe final como evidencia de auto‑evaluación.

---

*(Fin del archivo)*
