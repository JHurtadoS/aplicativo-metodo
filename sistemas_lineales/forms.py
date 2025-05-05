from django import forms
import numpy as np
import json # Import json for parsing

METHOD_CHOICES = [
    ('gauss', 'Eliminación Gaussiana (con pivoteo parcial)'),
    ('lu', 'Descomposición LU (Doolittle)'),
    ('jacobi', 'Método de Jacobi'),
    ('gauss_seidel', 'Método de Gauss-Seidel'),
]

class SistemaLinealForm(forms.Form):
    n_size = forms.IntegerField(
        label='Tamaño de la Matriz (n x n)',
        min_value=2,
        max_value=10, # Keep it reasonable for manual input
        initial=3,
        widget=forms.NumberInput(attrs={'id': 'n_size_input', 'class': 'border p-1 rounded w-20'})
    )
    # Hidden field to store the matrix data from the JS component
    matrix_data = forms.CharField(widget=forms.HiddenInput(), required=False)
    # Hidden field to store the vector data from the JS component
    vector_data = forms.CharField(widget=forms.HiddenInput(), required=False)

    # Keep the method selection and iterative parameters
    method = forms.ChoiceField(
        label='Método a utilizar',
        choices=METHOD_CHOICES,
        widget=forms.RadioSelect
    )
    initial_guess = forms.CharField(
        label='Vector Inicial (x₀) - Solo para Jacobi/Gauss-Seidel',
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Ej: 1,0,1'}),
        help_text='Valores separados por coma. Si se deja vacío, se usará un vector de ceros.'
    )
    tolerance = forms.FloatField(
        label='Tolerancia (Error) - Solo para Jacobi/Gauss-Seidel',
        required=False,
        initial=1e-5,
        min_value=1e-15,
        widget=forms.NumberInput(attrs={'step': '1e-6'})
    )
    max_iterations = forms.IntegerField(
        label='Máximo de Iteraciones - Solo para Jacobi/Gauss-Seidel',
        required=False,
        initial=100,
        min_value=1,
        max_value=1000
    )
    
    exact_solution = forms.CharField(
        label='Solución Exacta (Opcional)',
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Ej: 1,0,-2'}),
        help_text='Valores separados por coma. Si se proporciona, se utilizará para calcular el error.'
    )

    # Remove clean_matrix_input as parsing happens differently now
    # We will add a general clean method to construct A and b

    def clean(self):
        cleaned_data = super().clean()
        method = cleaned_data.get('method')
        n_size = cleaned_data.get('n_size')
        matrix_json = cleaned_data.get('matrix_data')
        vector_json = cleaned_data.get('vector_data')

        # Explicitly check if data is missing on POST
        if not matrix_json or not vector_json:
             # Raise a non-field error if the core data is missing
             self.add_error(None, "No se recibieron datos válidos de la matriz/vector desde la cuadrícula. Asegúrese de que JavaScript esté habilitado y funcione correctamente.")
        elif matrix_json and vector_json:
             try:
                 # Check type before loading JSON
                 if not isinstance(matrix_json, str) or not isinstance(vector_json, str):
                    raise ValueError("Los datos de la matriz/vector deben ser texto.")

                 matrix_list = json.loads(matrix_json)
                 vector_list = json.loads(vector_json)

                 if not isinstance(matrix_list, list) or not isinstance(vector_list, list):
                     raise ValueError("Formato de datos JSON inválido (se esperaba una lista).")

                 # Validate dimensions from JSON against n_size if n_size is available
                 if n_size:
                     if len(matrix_list) != n_size or not all(isinstance(row, list) and len(row) == n_size for row in matrix_list):
                          raise forms.ValidationError(f"Las dimensiones de la matriz recibida no coinciden con el tamaño especificado ({n_size}x{n_size}).")
                     if len(vector_list) != n_size:
                          raise forms.ValidationError(f"La longitud del vector b ({len(vector_list)}) no coincide con el tamaño especificado ({n_size}).")
                 else:
                     # If n_size wasn't valid, we can't strictly check dimensions here, but basic structure check is done above
                     pass

                 # Convert to NumPy arrays
                 A = np.array(matrix_list, dtype=float)
                 b = np.array(vector_list, dtype=float)
                 cleaned_data['matrix_A'] = A
                 cleaned_data['vector_b'] = b

             except (json.JSONDecodeError, ValueError, TypeError) as e:
                 self.add_error(None, f"Error al procesar los datos de la matriz/vector JSON: {e}")
             except Exception as e:
                  self.add_error(None, f"Error inesperado al procesar la matriz: {e}")

        # Validate initial guess size against n_size (only if n_size is valid)
        initial_guess_str = cleaned_data.get('initial_guess', '').strip()
        if initial_guess_str:
            if n_size:
                try:
                    guess_list = [float(x.strip()) for x in initial_guess_str.split(',')]
                    guess_vector = np.array(guess_list, dtype=float)
                    if len(guess_vector) != n_size:
                         self.add_error('initial_guess', f"El vector inicial debe tener {n_size} elementos.")
                    else:
                        cleaned_data['initial_guess_vector'] = guess_vector
                except ValueError:
                     self.add_error('initial_guess', "Formato inválido. Use números separados por comas.")
            else:
                 # Can't validate length if n_size is invalid
                 self.add_error('initial_guess', "No se puede validar el vector inicial sin un tamaño de matriz válido.")
        elif n_size: # If guess is empty string, default to zeros only if n_size is valid
            cleaned_data['initial_guess_vector'] = np.zeros(n_size, dtype=float)
        else:
            cleaned_data['initial_guess_vector'] = None

        # Validate exact solution size against n_size (only if n_size is valid)
        exact_solution_str = cleaned_data.get('exact_solution', '').strip()
        if exact_solution_str:
            if n_size:
                try:
                    solution_list = [float(x.strip()) for x in exact_solution_str.split(',')]
                    solution_vector = np.array(solution_list, dtype=float)
                    if len(solution_vector) != n_size:
                         self.add_error('exact_solution', f"El vector de la solución exacta debe tener {n_size} elementos.")
                    else:
                        cleaned_data['exact_solution_vector'] = solution_vector
                except ValueError:
                     self.add_error('exact_solution', "Formato inválido. Use números separados por comas.")
            else:
                 # Can't validate length if n_size is invalid
                 self.add_error('exact_solution', "No se puede validar la solución exacta sin un tamaño de matriz válido.")
        else:
            cleaned_data['exact_solution_vector'] = None

        # Validate required fields for iterative methods
        if method in ['jacobi', 'gauss_seidel']:
            if cleaned_data.get('tolerance') is None:
                self.add_error('tolerance', 'Este campo es requerido para métodos iterativos.')
            if cleaned_data.get('max_iterations') is None:
                 self.add_error('max_iterations', 'Este campo es requerido para métodos iterativos.')

        return cleaned_data 