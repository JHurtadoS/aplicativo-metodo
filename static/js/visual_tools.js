// visual_tools.js: reusable Preact components for visual outputs
// Requires Preact+HTM loaded in HTML

export function ResultsPanel({ canvasId, formulaId, tableId, errorMetricsId, comparisonId }) {
  const html = window.html;
  return html`
    <div class="right-panel">
      <canvas id=${canvasId} style="width:100%; height:200px;"></canvas>
      <div id=${formulaId} class="mt-4"></div>
      <div id=${errorMetricsId} class="mt-4"></div>
      <div id=${comparisonId} class="mt-4"></div>
      <div id=${tableId} class="mt-4"></div>
    </div>
  `;
}

export function initChart(canvasId, data, options) {
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, { data, options, type: options.type || 'line' });
}

export function renderFormula(elementId, latex) {
  const container = document.getElementById(elementId);
  if (container) {
    // Limpiar el contenedor antes de añadir nuevo contenido
    container.innerHTML = '';
    // Crear un div para la fórmula
    const formulaDiv = document.createElement('div');
    container.appendChild(formulaDiv);
    // Renderizar la fórmula directamente
    try {
      katex.render(latex, formulaDiv, {
        displayMode: true,
        throwOnError: false
      });
    } catch (e) {
      console.error('Error al renderizar LaTeX:', e);
      formulaDiv.textContent = `Error en fórmula: ${latex}`;
    }
  }
}

export function initTable(tableId, tableData, columns) {
  return new Tabulator(`#${tableId}`, { data: tableData, layout: 'fitColumns', columns });
}

export function renderErrorMetrics(elementId, errorMetrics) {
  const container = document.getElementById(elementId);
  if (!container || !errorMetrics) return;
  
  // Limpiar el contenedor antes de añadir nuevo contenido
  container.innerHTML = '';
  
  // Crear un div para los errores
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-metrics-panel p-3 border rounded';
  
  // Título
  const title = document.createElement('h4');
  title.textContent = 'Comparación con Solución Exacta';
  title.className = 'text-lg font-bold mb-2';
  errorDiv.appendChild(title);
  
  // Crear tabla para mostrar los errores
  const table = document.createElement('table');
  table.className = 'table-auto w-full text-sm';
  
  // Cabecera de la tabla
  table.innerHTML = `
    <thead>
      <tr class="bg-gray-100">
        <th class="px-4 py-2 text-left">Métrica</th>
        <th class="px-4 py-2 text-right">Valor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="border px-4 py-2">Error Absoluto Máximo</td>
        <td class="border px-4 py-2 text-right">${errorMetrics.max_abs_error.toExponential(6)}</td>
      </tr>
      <tr>
        <td class="border px-4 py-2">Error Relativo Máximo</td>
        <td class="border px-4 py-2 text-right">${errorMetrics.max_rel_error.toExponential(6)}</td>
      </tr>
      <tr>
        <td class="border px-4 py-2">Norma del Error Absoluto</td>
        <td class="border px-4 py-2 text-right">${errorMetrics.abs_error_norm.toExponential(6)}</td>
      </tr>
      <tr>
        <td class="border px-4 py-2">Norma del Error Relativo</td>
        <td class="border px-4 py-2 text-right">${errorMetrics.rel_error_norm.toExponential(6)}</td>
      </tr>
    </tbody>
  `;
  
  errorDiv.appendChild(table);
  container.appendChild(errorDiv);
}

export function renderSolutionComparison(elementId, approximateSolution, exactSolution) {
  const container = document.getElementById(elementId);
  if (!container || !approximateSolution || !exactSolution) return;
  
  // Limpiar el contenedor antes de añadir nuevo contenido
  container.innerHTML = '';
  
  // Crear un div para la comparación
  const comparisonDiv = document.createElement('div');
  comparisonDiv.className = 'solution-comparison-panel p-3 border rounded';
  
  // Título
  const title = document.createElement('h4');
  title.textContent = 'Comparación Visual de Soluciones';
  title.className = 'text-lg font-bold mb-2';
  comparisonDiv.appendChild(title);
  
  // Crear tabla para mostrar ambas soluciones
  const table = document.createElement('table');
  table.className = 'table-auto w-full text-sm mb-4';
  
  // Cabecera de la tabla
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr class="bg-gray-100">
      <th class="px-4 py-2 text-left">Variable</th>
      <th class="px-4 py-2 text-right">Solución Aproximada</th>
      <th class="px-4 py-2 text-right">Solución Exacta</th>
      <th class="px-4 py-2 text-right">Diferencia</th>
    </tr>
  `;
  table.appendChild(thead);
  
  // Cuerpo de la tabla
  const tbody = document.createElement('tbody');
  const n = approximateSolution.length;
  
  for (let i = 0; i < n; i++) {
    const approxValue = approximateSolution[i];
    const exactValue = exactSolution[i];
    const diff = Math.abs(approxValue - exactValue);
    
    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="border px-4 py-2">x<sub>${i+1}</sub></td>
      <td class="border px-4 py-2 text-right">${approxValue.toFixed(6)}</td>
      <td class="border px-4 py-2 text-right">${exactValue.toFixed(6)}</td>
      <td class="border px-4 py-2 text-right">${diff.toExponential(4)}</td>
    `;
    tbody.appendChild(row);
  }
  
  table.appendChild(tbody);
  comparisonDiv.appendChild(table);
  
  // Crear gráfico de barras para comparación visual
  const chartContainer = document.createElement('div');
  chartContainer.className = 'mt-4';
  chartContainer.style.height = '300px';
  
  // Canvas para el gráfico
  const canvas = document.createElement('canvas');
  canvas.id = `comparison-chart-${Math.random().toString(36).substr(2, 9)}`;
  chartContainer.appendChild(canvas);
  comparisonDiv.appendChild(chartContainer);
  
  container.appendChild(comparisonDiv);
  
  // Crear datos para el gráfico
  const labels = Array.from({length: n}, (_, i) => `x${i+1}`);
  const approximateData = approximateSolution.map(val => parseFloat(val));
  const exactData = exactSolution.map(val => parseFloat(val));
  
  // Crear gráfico con Chart.js
  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Solución Aproximada',
          data: approximateData,
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        },
        {
          label: 'Solución Exacta',
          data: exactData,
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Comparación de Soluciones',
          font: {
            size: 16
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const dataIndex = context.dataIndex;
              const datasetIndex = context.datasetIndex;
              const value = context.raw;
              
              if (datasetIndex === 0) {
                const exactValue = exactData[dataIndex];
                const diff = Math.abs(value - exactValue);
                return [`${context.dataset.label}: ${value.toFixed(6)}`, `Diferencia: ${diff.toExponential(4)}`];
              }
              return `${context.dataset.label}: ${value.toFixed(6)}`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: 'Valor'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Variables'
          }
        }
      }
    }
  });
} 