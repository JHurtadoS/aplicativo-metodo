import os
import uuid
from datetime import datetime
from pathlib import Path
from weasyprint import HTML, CSS
from django.template.loader import render_to_string
from django.conf import settings

# Asegurar que existan los directorios necesarios
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / 'static' / 'pdfs'
PDF_DIR.mkdir(parents=True, exist_ok=True)

def generate_operation_pdf(result_data, operation_type):
    """
    Genera un PDF a partir de los datos de una operación
    
    Args:
        result_data (dict): Datos de resultado de la operación
        operation_type (str): Tipo de operación ('sistemas_lineales', 'interpolacion', etc.)
    
    Returns:
        str: Ruta relativa al PDF generado
    """
    # Crear un identificador único para el archivo
    file_uuid = uuid.uuid4().hex
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{operation_type}_{timestamp}_{file_uuid}.pdf"
    
    # Preparar el contexto para la plantilla
    context = {
        'result': result_data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'operation_type': operation_type
    }
    
    # Renderizar HTML según el tipo de operación
    template_name = f"pdf/{operation_type}_pdf.html"
    html_string = render_to_string(template_name, context)
    
    # Crear PDF con WeasyPrint
    pdf_path = PDF_DIR / filename
    
    # Establecer CSS para el PDF (incluye Tailwind básico para estilos)
    css = CSS(string='''
        body { font-family: sans-serif; margin: 2cm; }
        h1 { color: #4f46e5; font-size: 24px; margin-bottom: 20px; }
        h2 { color: #374151; font-size: 18px; margin-top: 15px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
        th { background-color: #f9fafb; }
        img { max-width: 100%; height: auto; }
        .container { max-width: 800px; margin: 0 auto; }
        .footer { margin-top: 30px; font-size: 12px; color: #6b7280; text-align: center; }
    ''')
    
    HTML(string=html_string).write_pdf(pdf_path, stylesheets=[css])
    
    # Devolver la ruta relativa para el acceso desde la web
    rel_path = f"pdfs/{filename}"
    return rel_path

def save_to_history(request, operation_type, method, input_data, result_data):
    """
    Guarda una operación en el historial de la sesión
    
    Args:
        request: Objeto request de Django
        operation_type (str): Tipo de operación ('sistemas_lineales', 'interpolacion', etc.)
        method (str): Método específico utilizado 
        input_data (dict): Datos de entrada
        result_data (dict): Resultado de la operación
    """
    # Generar PDF
    pdf_path = generate_operation_pdf(result_data, operation_type)
    
    # Inicializar historial si no existe
    if 'historial' not in request.session:
        request.session['historial'] = []
    
    # Crear entrada de historial
    historial_item = {
        'tipo': operation_type,
        'metodo': method,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'entrada': input_data,
        'resultado_resumen': _get_result_summary(result_data, operation_type),
        'grafico_path': result_data.get('grafico_path', None),
        'pdf_path': pdf_path
    }
    
    # Añadir al historial
    request.session['historial'].append(historial_item)
    request.session.modified = True
    
    return historial_item

def _get_result_summary(result_data, operation_type):
    """
    Genera un resumen del resultado para mostrar en el historial
    """
    if operation_type == 'sistemas_lineales':
        if 'solution' in result_data:
            return {'solucion': result_data['solution'][:3]}  # Primeros 3 valores
        return {'info': 'Sistema resuelto'}
        
    elif operation_type == 'interpolacion':
        if hasattr(result_data, 'polinomio_tex'):
            return {'polinomio': result_data.polinomio_tex[:50] + '...' if len(result_data.polinomio_tex) > 50 else result_data.polinomio_tex}
        return {'info': 'Interpolación calculada'}
        
    return {'info': 'Operación completada'}

# Add this helper function if the file contains LaTeX rendering code for matrices
def fix_nested_matrices(latex_str):
    """
    Fixes the issue with nested matrices in LaTeX representation.
    
    Args:
        latex_str: LaTeX string that might contain nested matrices
        
    Returns:
        Corrected LaTeX string with single matrix
    """
    if not latex_str:
        return latex_str
        
    # Remove nested bmatrix environments wherever they appear
    if "\\begin{bmatrix}\\begin{bmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{bmatrix}\\begin{bmatrix}", "\\begin{bmatrix}")
        latex_str = latex_str.replace("\\end{bmatrix}\\end{bmatrix}", "\\end{bmatrix}")
    
    # Remove nested pmatrix environments wherever they appear
    if "\\begin{pmatrix}\\begin{pmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{pmatrix}\\begin{pmatrix}", "\\begin{pmatrix}")
        latex_str = latex_str.replace("\\end{pmatrix}\\end{pmatrix}", "\\end{pmatrix}")
    
    # Also handle potential cases with mixed environments
    if "\\begin{bmatrix}\\begin{pmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{bmatrix}\\begin{pmatrix}", "\\begin{bmatrix}")
        latex_str = latex_str.replace("\\end{pmatrix}\\end{bmatrix}", "\\end{bmatrix}")
    
    if "\\begin{pmatrix}\\begin{bmatrix}" in latex_str:
        latex_str = latex_str.replace("\\begin{pmatrix}\\begin{bmatrix}", "\\begin{pmatrix}")
        latex_str = latex_str.replace("\\end{bmatrix}\\end{pmatrix}", "\\end{pmatrix}")
    
    return latex_str 