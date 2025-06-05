# 📋 CHECKLIST: Implementación de Métodos Numéricos



## 🧮 **FASE 2: Derivación Numérica**

### 2.1 Modelos y Formularios
- [x] Crear `forms.py` para derivación:
  - [x] Campo función: `CharField` con validación Sympy
  - [x] Campo punto x₀: `FloatField`
  - [x] Campo paso h: `FloatField` (positivo, no cero)
  - [x] Campo método: `ChoiceField` (adelante, atrás, central)
- [x] Implementar validadores personalizados:
  - [x] Validar sintaxis de función con `sympify`
  - [x] Validar que h > 0
  - [x] Validar que x₀ sea numérico

### 2.2 Lógica de Cálculo (utils)
- [x] Crear `numerical_methods.py` en utils:
  - [x] `forward_diff(f, x0, h)` - Diferencia hacia adelante
  - [x] `backward_diff(f, x0, h)` - Diferencia hacia atrás  
  - [x] `central_diff(f, x0, h)` - Diferencia central
- [x] Cada función debe retornar:
  - [x] Resultado numérico
  - [x] Pasos intermedios (evaluaciones f(x₀±h))
  - [x] Detalles del cálculo

### 2.3 Vistas y Templates
- [x] Vista formulario de entrada (`FormView` o función)
- [x] Vista procesamiento y resultados
- [x] Template formulario con:
  - [x] Campos de entrada
  - [x] Selector de método
  - [x] Validación HTML5
- [x] Template resultados con:
  - [x] Tabla paso a paso
  - [x] Resultado final
  - [x] Espacio para gráfico

### 2.4 Visualización
- [x] Función generar gráfico derivación:
  - [x] Plotear f(x) alrededor de x₀
  - [x] Marcar puntos utilizados (x₀, x₀±h)
  - [x] Dibujar línea secante
  - [x] Guardar como PNG en static/images/
- [x] Integrar gráfico en template de resultados

### 2.5 Exportación
- [ ] Template especial para PDF (sin navegación)
- [ ] Función exportar PDF con WeasyPrint
- [x] Función exportar TXT con datos tabulados
- [x] Botones descarga en template resultados

### 2.6 Historial de Sesión
- [x] Guardar operación en `request.session['history']`
- [x] Estructura: tema, entrada, método, resultado
- [x] Vista para mostrar historial

---

## ∫ **FASE 3: Integración Numérica**

### 3.1 Modelos y Formularios
- [x] Crear formulario integración:
  - [x] Campo función: `CharField`
  - [x] Límite inferior a: `FloatField`
  - [x] Límite superior b: `FloatField`
  - [x] Número subintervalos n: `IntegerField`
  - [x] Método: `ChoiceField` (Trapecio, Simpson)
- [x] Validadores específicos:
  - [x] a < b
  - [x] n > 0 y entero
  - [x] Para Simpson: n debe ser par

### 3.2 Lógica de Cálculo
- [x] Implementar funciones utils:
  - [x] `trapezoidal(f, a, b, n)` - Regla del trapecio compuesta
  - [x] `simpson(f, a, b, n)` - Regla de Simpson 1/3
- [x] Cada función retorna:
  - [x] Array de puntos xi
  - [x] Array de evaluaciones f(xi)
  - [x] Coeficientes aplicados
  - [x] Resultado final
  - [x] Pasos intermedios detallados

### 3.3 Vistas y Templates
- [x] Vista formulario integración
- [x] Vista procesamiento con validación n par para Simpson
- [x] Template entrada con validación dinámica
- [x] Template resultados con tabla de xi, f(xi), coeficientes

### 3.4 Visualización
- [x] Función gráfico trapecio:
  - [x] Plotear f(x) en [a,b]
  - [x] Dibujar y rellenar trapezoides
  - [x] Marcar puntos de partición
- [x] Función gráfico Simpson:
  - [x] Plotear f(x) y parábolas aproximantes
  - [x] Rellenar áreas bajo parábolas

### 3.5 Exportación e Historial
- [ ] PDF con tabla de valores y resultado
- [ ] TXT con datos tabulados
- [x] Guardar en historial de sesión

---

## 📊 **FASE 4: Ecuaciones Diferenciales Ordinarias**

### 4.1 Modelos y Formularios
- [x] Formulario EDO:
  - [x] Función f(t,y): `CharField`
  - [x] Condición inicial t₀: `FloatField`
  - [x] Condición inicial y₀: `FloatField`
  - [x] Paso h: `FloatField`
  - [x] Número de pasos N: `IntegerField`
  - [x] Método: `ChoiceField` (Euler, RK4)

### 4.2 Lógica de Cálculo
- [x] Implementar métodos utils:
  - [x] `euler_method(f, t0, y0, h, N)` - Método de Euler
  - [x] `rk4_method(f, t0, y0, h, N)` - Runge-Kutta 4
- [x] Cada función retorna:
  - [x] Arrays (tn, yn)
  - [x] Pasos intermedios (ki para RK4)
  - [x] Estructura detallada de cada iteración

### 4.3 Vistas y Templates
- [x] Vista formulario EDO
- [x] Vista procesamiento con validación f(t,y)
- [x] Template entrada con ejemplos de sintaxis
- [x] Template resultados con tabla tn, yn, ki

### 4.4 Visualización
- [x] Función gráfico EDO:
  - [x] Plotear solución numérica (tn, yn)
  - [x] Mostrar puntos calculados
  - [ ] Si existe solución exacta, compararla
  - [x] Etiquetar ejes apropiadamente

### 4.5 Exportación e Historial
- [ ] PDF con ecuación, condiciones, pasos
- [ ] TXT con tabla (tn, yn)
- [x] Guardar en historial

---

## 🎯 **FASE 5: Características Avanzadas**

### 5.1 Estimación de Error
- [x] Implementar control de error para derivadas:
  - [x] Error de curvatura Ec = (f(x+h)-2f(x)+f(x-h))/h²
- [x] Error compuesto para integración:
  - [x] Comparación con solución analítica usando SymPy
  - [x] Error relativo εr = |ynum - yexact|/|yexact|
- [ ] Comparación con solución analítica para EDOs (si existe)

### 5.2 Paso Adaptativo (Bonus)
- [ ] Integración adaptativa:
  - [ ] Calcular con h y h/2
  - [ ] Comparar resultados
  - [ ] Subdividir si necesario
- [ ] EDO con control de paso:
  - [ ] Implementar par embedded RK4/RK5
  - [ ] Ajustar paso según error local

### 5.3 DataClass Unificado
- [x] Crear `NMResult` dataclass:
  - [x] tema, metodo, entrada, pasos, valor, error, grafico
- [x] Usar en todas las vistas
- [x] Simplificar exportación

---

## 🎨 **FASE 6: UI/UX y Plantillas**

### 6.1 Templates Base
- [x] Template base con Bootstrap 5 (Tailwind CSS usado)
- [x] Navegación entre módulos
- [x] Responsive design

### 6.2 Mini-Wizard por Tema
- [x] Paso 1: Ingreso datos + validación HTML5
- [x] Paso 2: Selección método + tooltips informativos
- [x] Paso 3: Resultados + gráfica + descargas

### 6.3 Historial de Sesión
- [x] Vista historial completo
- [ ] Filtros por tema/método
- [x] Opción limpiar historial
- [ ] Export historial completo

---

## 🔧 **FASE 7:  Optimización**

### 7.1 Cards Habilitadas
- [x] Habilitar card "Diferenciación e Integración"
- [x] Habilitar card "EDO: Euler y RK4"
- [x] URLs configuradas correctamente
- [x] Navegación funcional

### 7.2 Manejo de Errores
- [x] Capturar excepciones Sympy
- [x] Mensajes de error claros
- [x] Fallbacks para cálculos fallidos
- [ ] Logging de errores


## ✅ **VERIFICACIÓN FINAL**

### Funcionalidad Core
- [x] ✅ Derivación numérica completa (3 métodos)
- [x] ✅ Integración numérica completa (2 métodos)
- [x] ✅ EDO completas (2 métodos)
- [x] ✅ Validación de todos los inputs
- [x] ✅ Visualización gráfica
- [ ] ✅ Exportación PDF/TXT
- [x] ✅ Historial de sesión

### UI/UX
- [x] ✅ Interfaz intuitiva y responsive
- [x] ✅ Mensajes de error claros
- [x] ✅ Navegación fluida
- [x] ✅ Visualización de resultados clara

### Robustez
- [x] ✅ Manejo de errores completo
- [x] ✅ Validación exhaustiva
- [x] ✅ Performance aceptable
- [ ] ✅ Documentación completa

## 🚀 **ESTADO ACTUAL**

### ✅ COMPLETADO:
- **Fase 2**: Derivación numérica completa con 3 métodos (adelante, atrás, central)
- **Fase 3**: Integración numérica con trapecio y Simpson
- **Fase 4**: EDOs con Euler y Runge-Kutta 4
- **Fase 5**: DataClass unificado y estimación de errores básica
- **Fase 6**: UI moderna con wizard steps y templates responsivos
- **Fase 7**: Cards habilitadas y navegación funcional

### 🔄 EN PROGRESO:
- Exportación PDF usando WeasyPrint
- Templates específicos para PDF
- Dark mode opcional

### 📋 PENDIENTE:
- Integración adaptativa (bonus)
- Control de paso adaptativo para EDOs (bonus)
- Logging completo de errores
- Filtros avanzados en historial

**Funcionalidades principales implementadas al 85%** ✅