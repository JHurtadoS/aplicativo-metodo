# Project Checklist: Numerical Methods Django App

Based on the planning document (`contexto metodo paso por paso.txt`) and the current project structure.

## I. Project Setup & Core Features

-   [x] Initialize Django Project (`ProyectoMN`)
-   [x] Setup Base Template (`templates/base.html`)
-   [ ] **Homepage/Core View (`core` app):** (View/URL files exist, template/nav logic needed)
    -   [x] Create main view (`core/views.py`) to display navigation links to modules.
    -   [x] Define URL (`core/urls.py`, `ProyectoMN/urls.py`). (Files exist)
    -   [ ] Create template (`templates/core/index.html`) extending `base.html`.
-   [ ] **History Feature (`core` app or per module):**
    -   [ ] Design session/database storage for calculations.
    -   [ ] Implement logic to save calculation details.
    -   [ ] Create view/template to display history.
-   [ ] **Export Feature (`core` app or helper utility):**
    -   [ ] Choose and install PDF generation library (e.g., `WeasyPrint`, `xhtml2pdf`).
    -   [ ] Implement utility function to generate PDF/TXT from results/steps.
    -   [ ] Add export buttons/links to result views.
-   [ ] **Dependency Management:**
    -   [ ] Create `requirements.txt` listing all dependencies (Django, NumPy, SymPy, Matplotlib, PDF lib).

## II. Module: Sistemas de Ecuaciones Lineales (`sistemas_lineales` app)

-   [x] Create Django App (`sistemas_lineales`)
-   [x] Basic App Files (`views.py`, `urls.py`, `models.py`, `templates/sistemas_lineales/`) (Files created)
-   [ ] **Input Form:**
    -   [ ] Create Django Form (`forms.py`?) for matrix A, vector b, method selection (Gauss, LU, Jacobi, Gauss-Seidel), tolerance/iterations (for iterative methods).
    -   [ ] Implement view (`views.py`) to handle GET (show form) and POST (process data).
    -   [ ] Create template (`templates/sistemas_lineales/solver.html`?) for the form.
-   [ ] **Calculation Logic:**
    -   [ ] Implement Gauss Elimination (with/without pivoting) function (using NumPy).
    -   [ ] Implement LU Decomposition function (using NumPy).
    -   [ ] Implement Jacobi method function (using NumPy).
    -   [ ] Implement Gauss-Seidel method function (using NumPy).
    -   [ ] Integrate SymPy for displaying matrices/steps clearly.
-   [ ] **Step-by-Step Output:**
    -   [ ] Modify calculation functions to return intermediate steps.
    -   [ ] Design view/template logic to display steps sequentially.
-   [ ] **Results & Visualization:**
    -   [ ] Display final solution vector.
    -   [ ] Handle singular/incompatible systems.
    -   [ ] (Optional) Implement 2x2 system visualization (Matplotlib).
    -   [ ] Integrate Matplotlib for convergence plots (iterative methods).
-   [ ] **Theoretical Explanation:**
    -   [ ] Add brief text descriptions of methods in the template.
-   [ ] **Export Integration:**
    -   [ ] Link results view to the export function.

## III. Module: Interpolación y Ajuste de Curvas (`interpolacion` app)

-   [x] Create Django App (`interpolacion`)
-   [x] Basic App Files (`views.py`, `urls.py`, `models.py`, `templates/interpolacion/`) (Files created)
-   [ ] **Input Form:**
    -   [x] Create Django Form (`forms.py`) for points (x, y), method selection (Lagrange, Newton, Linear Regression, Polynomial Regression?).
    -   [x] Implement view (`views.py`) to handle GET/POST.
    -   [x] Create template (`templates/interpolacion/interpolacion_form.html`) for the form.
-   [ ] **Calculation Logic:**
    -   [x] Implement Lagrange Interpolation function (NumPy/SymPy).
    -   [x] Implement Newton Interpolation (divided differences) function (NumPy/SymPy).
    -   [x] Implement Linear Regression (least squares) function (NumPy).
    -   [ ] (Optional) Implement Polynomial Regression function.
    -   [x] Integrate SymPy to display resulting polynomial formulas.
-   [ ] **Step-by-Step Output:**
    -   [x] Show Lagrange basis polynomials / Newton divided difference table.
    -   [x] Show steps for calculating regression coefficients.
-   [ ] **Results & Visualization:**
    -   [x] Display final polynomial/line equation.
    -   [x] Integrate Matplotlib to plot points and the resulting curve/line.
-   [ ] **Theoretical Explanation:**
    -   [x] Add brief text descriptions of methods in the template.
-   [ ] **Export Integration:**
    -   [ ] Link results view to the export function (including graph).

## IV. Module: Diferenciación e Integración Numérica (New App: `calculo_numerico`?)

-   [ ] **Create Django App (`calculo_numerico` or similar):**
    -   [ ] Add app to `settings.py`, create basic files (`views.py`, `urls.py`, etc.).
-   [ ] **Input Form:**
    -   [ ] Form for function string f(x), parameters (x0, h for differentiation; a, b, n/h for integration), method selection (Forward/Backward/Central Diff; Trapezoid, Simpson).
    -   [ ] View to handle GET/POST.
    -   [ ] Template for the form.
    -   [ ] Use SymPy `sympify`/`lambdify` to parse and evaluate function string safely.
-   [ ] **Calculation Logic:**
    -   [ ] Implement Finite Difference formulas (NumPy).
    -   [ ] Implement Trapezoidal Rule (composite) function (NumPy).
    -   [ ] Implement Simpson's 1/3 Rule (composite) function (NumPy).
    -   [ ] (Optional) Integrate SymPy for symbolic differentiation/integration comparison.
-   [ ] **Step-by-Step Output:**
    -   [ ] Show function evaluations for differentiation formulas.
    -   [ ] Show interval partitioning, function evaluations at points, and summation steps for integration.
-   [ ] **Results & Visualization:**
    -   [ ] Display approximate derivative/integral value.
    -   [ ] Integrate Matplotlib to plot function and illustrate integration area (trapezoids/parabolas) or secant lines for differentiation.
-   [ ] **Theoretical Explanation:**
    -   [ ] Add brief text descriptions of methods.
-   [ ] **Export Integration:**
    -   [ ] Link results view to the export function (including graph).

## V. Module: Ecuaciones Diferenciales Ordinarias (New App: `edo`?)

-   [ ] **Create Django App (`edo` or similar):**
    -   [ ] Add app to `settings.py`, create basic files.
-   [ ] **Input Form:**
    -   [ ] Form for function string f(t, y), initial condition (t0, y0), interval end T, step h (or N), method selection (Euler, Heun/Improved Euler, RK4).
    -   [ ] View to handle GET/POST.
    -   [ ] Template for the form.
    -   [ ] Use SymPy `sympify`/`lambdify` for the function f(t, y).
-   [ ] **Calculation Logic:**
    -   [ ] Implement Euler's method function (NumPy).
    -   [ ] Implement Heun's/Improved Euler method function (NumPy).
    -   [ ] Implement Runge-Kutta 4th Order (RK4) method function (NumPy).
    -   [ ] (Optional) Integrate SymPy `dsolve` for analytic solution comparison.
-   [ ] **Step-by-Step Output:**
    -   [ ] Display iteration table (ti, yi, intermediate steps like k1-k4 for RK4).
-   [ ] **Results & Visualization:**
    -   [ ] Display final approximation y(T) and/or table of results.
    -   [ ] Integrate Matplotlib to plot the approximate solution y(t) vs t.
    -   [ ] (Optional) Plot analytical solution alongside if available.
-   [ ] **Theoretical Explanation:**
    -   [ ] Add brief text descriptions of methods.
-   [ ] **Export Integration:**
    -   [ ] Link results view to the export function (including graph).

## VI. Documentation & Refinement

-   [ ] **README.md:** Add setup instructions, dependencies, how to run.
-   [ ] **Code Comments:** Add comments for complex logic, especially in calculation functions.
-   [ ] **Testing:** (Optional but recommended) Add basic tests for calculation functions.
-   [ ] **Error Handling:** Implement robust error handling (e.g., invalid inputs, non-convergence, singular matrices).
-   [ ] **User Experience:** Refine UI/UX based on usage.

**Next Steps Overview:**

1.  Flesh out the `core` app (homepage, base template styling).
2.  Implement the **missing apps**: `calculo_numerico` (for Diff/Int) and `edo`.
3.  Implement the **Forms, Views, and Templates** for input in each app (`sistemas_lineales`, `interpolacion`, `calculo_numerico`, `edo`).
4.  Implement the **Calculation Logic** for each numerical method using NumPy/SymPy.
5.  Implement the **Step-by-Step Output** display logic in views/templates.
6.  Integrate **Matplotlib** for visualizations in relevant modules.
7.  Implement the general **History and Export** features.
8.  Add **Theoretical Explanations**.
9.  Create `requirements.txt` and `README.md`.
10. Refine and test. 