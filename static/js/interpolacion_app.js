// interpolacion_app.js
// Componente Preact para la aplicación de interpolación

// Esperar a que los objetos globales de Preact estén disponibles
(function() {
  // Verificar que Preact y sus hooks estén disponibles globalmente
  if (!window.preact || !window.preactHooks) {
    console.error("Preact o sus hooks no están disponibles. Revise las dependencias.");
    return;
  }
  
  // Usar Preact desde el objeto global con destructuring seguro
  const { h, render, Component } = window.preact;
  const { useState, useEffect } = window.preactHooks;

  // Componente de formulario mejorado
  const PointInput = ({ initialPoints, onChange }) => {
    const [points, setPoints] = useState(initialPoints || "1,2\n2,5\n3,10");
    const [preview, setPreview] = useState([]);
    
    useEffect(() => {
      // Parsear puntos para vista previa
      try {
        const parsed = points.split('\n')
          .map(line => line.trim())
          .filter(line => line)
          .map(line => {
            const [x, y] = line.split(',').map(n => parseFloat(n.trim()));
            return { x, y, valid: !isNaN(x) && !isNaN(y) };
          });
        setPreview(parsed);
        onChange(points);
      } catch (e) {
        console.error("Error parsing points:", e);
      }
    }, [points]);
    
    return h('div', { className: 'mb-6' }, [
      h('label', { className: 'block text-gray-700 text-sm font-bold mb-2' }, 'Puntos (x,y)'),
      h('div', { className: 'flex space-x-4' }, [
        // Textarea para entrada
        h('div', { className: 'w-1/2' }, [
          h('textarea', { 
            id: 'id_points',
            name: 'points',
            value: points,
            className: 'shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline', 
            rows: 6,
            placeholder: 'Ingrese un punto por línea, ej:\n1,2\n3,5\n4,8',
            onChange: (e) => setPoints(e.target.value)
          }),
          h('p', { className: 'text-gray-600 text-xs italic mt-1' }, 
            'Ingrese cada punto (x,y) en una línea separada, con valores separados por coma.')
        ]),
        
        // Vista previa de puntos
        h('div', { className: 'w-1/2 border rounded p-3 bg-gray-50' }, [
          h('h4', { className: 'font-semibold text-sm mb-2' }, 'Vista previa:'),
          preview.length > 0 ? 
            h('table', { className: 'w-full text-sm' }, [
              h('thead', {}, 
                h('tr', {}, [
                  h('th', { className: 'text-left font-medium' }, 'x'),
                  h('th', { className: 'text-left font-medium' }, 'y'),
                  h('th', { className: 'text-left font-medium' }, 'Estado')
                ])
              ),
              h('tbody', {}, preview.map((p, idx) => 
                h('tr', { key: idx, className: p.valid ? 'text-green-700' : 'text-red-500' }, [
                  h('td', {}, p.x || '?'),
                  h('td', {}, p.y || '?'),
                  h('td', {}, p.valid ? '✓' : 'Error')
                ])
              ))
            ]) :
            h('p', { className: 'text-gray-500 italic' }, 'Sin puntos')
        ])
      ])
    ]);
  };

  // Componente principal
  const InterpolacionApp = ({ formAction, csrfToken, initialMethod, initialSolver, initialPoints }) => {
    const [method, setMethod] = useState(initialMethod || 'newton');
    const [solver, setSolver] = useState(initialSolver || 'gauss');
    const [points, setPoints] = useState(initialPoints || '');
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    const methods = [
      { value: 'lagrange', label: 'Interpolación de Lagrange' },
      { value: 'newton', label: 'Interpolación de Newton (Diferencias Divididas)' },
      { value: 'linear_regression', label: 'Regresión Lineal (Mínimos Cuadrados)' }
    ];
    
    const solvers = [
      { value: 'gauss', label: 'Eliminación Gaussiana (con pivoteo parcial)' },
      { value: 'lu', label: 'Descomposición LU (Doolittle)' }
    ];

    // Función para manejar el envío del formulario
    const handleSubmit = (e) => {
      setIsSubmitting(true);
      // No hacemos preventDefault, permitimos que el formulario se procese normalmente
    };
    
    return h('div', { className: 'bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4' }, [
      h('form', { 
        action: formAction, 
        method: 'post',
        onSubmit: handleSubmit
      }, [
        // CSRF Token
        h('input', { 
          type: 'hidden', 
          name: 'csrfmiddlewaretoken', 
          value: csrfToken 
        }),
        
        // Puntos
        h(PointInput, { 
          initialPoints, 
          onChange: setPoints 
        }),
        
        // Método
        h('div', { className: 'mb-6' }, [
          h('label', { className: 'block text-gray-700 text-sm font-bold mb-2' }, 'Método a utilizar'),
          h('div', { className: 'space-y-2' }, 
            methods.map(option => 
              h('div', { className: 'flex items-center', key: option.value }, [
                h('input', { 
                  type: 'radio',
                  id: `method_${option.value}`,
                  name: 'method',
                  value: option.value,
                  checked: method === option.value,
                  onChange: () => setMethod(option.value),
                  className: 'mr-2'
                }),
                h('label', { 
                  htmlFor: `method_${option.value}`,
                  className: 'text-sm text-gray-700 cursor-pointer'
                }, option.label)
              ])
            )
          )
        ]),
        
        // Solver
        h('div', { className: 'mb-6' }, [
          h('label', { className: 'block text-gray-700 text-sm font-bold mb-2' }, 'Solver para sistema lineal'),
          h('select', { 
            id: 'id_solver',
            name: 'solver',
            value: solver,
            onChange: (e) => setSolver(e.target.value),
            className: 'shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline'
          }, solvers.map(option => 
            h('option', { 
              value: option.value, 
              key: option.value 
            }, option.label)
          )),
          h('p', { className: 'text-gray-600 text-xs italic mt-1' }, 
            'Método para resolver el sistema lineal subyacente.')
        ]),
        
        // Botón submit
        h('div', { className: 'flex items-center justify-between' }, [
          h('button', { 
            type: 'submit',
            className: `bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline flex items-center ${isSubmitting ? 'opacity-75 cursor-not-allowed' : ''}`,
            disabled: isSubmitting
          }, [
            isSubmitting && h('svg', { 
              className: 'animate-spin -ml-1 mr-3 h-5 w-5 text-white', 
              xmlns: 'http://www.w3.org/2000/svg', 
              fill: 'none', 
              viewBox: '0 0 24 24' 
            }, [
              h('circle', { 
                className: 'opacity-25', 
                cx: '12', 
                cy: '12', 
                r: '10', 
                stroke: 'currentColor', 
                'stroke-width': '4' 
              }),
              h('path', { 
                className: 'opacity-75', 
                fill: 'currentColor', 
                d: 'M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z' 
              })
            ]),
            isSubmitting ? 'Calculando...' : 'Calcular'
          ])
        ])
      ])
    ]);
  };

  // Función para montar el componente
  function mountInterpolacionApp() {
    try {
      const container = document.getElementById('interpolacion-app');
      if (!container) {
        console.error("No se encontró el contenedor de la aplicación");
        return;
      }
      
      // Buscar el token CSRF en la página
      const csrfTokenElement = document.querySelector('input[name="csrfmiddlewaretoken"]');
      if (!csrfTokenElement) {
        console.error("No se pudo encontrar el token CSRF");
        return;
      }
      
      const csrfToken = csrfTokenElement.value;
      
      // Leer cualquier dato inicial del contexto
      const initialData = window.INTERP_CTX || {};
      
      // Limpiar el contenedor antes de renderizar
      container.innerHTML = '';
      
      // Renderizar el componente Preact
      render(
        h(InterpolacionApp, { 
          formAction: container.dataset.action || '',
          csrfToken,
          initialMethod: initialData.method,
          initialSolver: initialData.solver,
          initialPoints: initialData.points
        }), 
        container
      );
      
      console.log("Aplicación de interpolación montada correctamente");
    } catch (error) {
      console.error("Error al montar la aplicación:", error);
    }
  }

  // Inicializar cuando el DOM esté listo
  document.addEventListener('DOMContentLoaded', mountInterpolacionApp);
})(); 