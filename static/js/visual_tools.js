// visual_tools.js: reusable Preact components for visual outputs
// Requires Preact+HTM loaded in HTML

export function ResultsPanel({ canvasId, formulaId, tableId }) {
  const html = window.html;
  return html`
    <div class="right-panel">
      <canvas id=${canvasId} style="width:100%; height:200px;"></canvas>
      <div id=${formulaId} class="mt-4"></div>
      <div id=${tableId} class="mt-4"></div>
    </div>
  `;
}

// Example functions to initialize tools (to be called after rendering):
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