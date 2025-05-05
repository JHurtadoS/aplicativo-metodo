from django import forms

METHOD_CHOICES = [
    ('lagrange', 'Interpolación de Lagrange'),
    ('newton', 'Interpolación de Newton (Diferencias Divididas)'),
    ('linear_regression', 'Regresión Lineal (Mínimos Cuadrados)'),
    # ('polynomial_regression', 'Regresión Polinomial (Mínimos Cuadrados)'), # Optional
]

SOLVER_CHOICES = [
    ('gauss', 'Eliminación Gaussiana (con pivoteo parcial)'),
    ('lu', 'Descomposición LU (Doolittle)'),
    # Los métodos iterativos no tienen sentido para sistemas de interpolación
]

class InterpolacionForm(forms.Form):
    points = forms.CharField(
        label='Puntos (x,y)',
        widget=forms.Textarea(attrs={'rows': 5, 'placeholder': 'Ingrese un punto por línea, ej:\n1,2\n3,5\n4,8'}),
        help_text='Ingrese cada punto (x,y) en una línea separada, con valores separados por coma.'
    )
    method = forms.ChoiceField(
        label='Método a utilizar',
        choices=METHOD_CHOICES,
        widget=forms.RadioSelect
    )
    solver = forms.ChoiceField(
        label='Solver para sistema lineal',
        choices=SOLVER_CHOICES,
        initial='gauss',
        help_text='Método para resolver el sistema lineal subyacente.'
    )
    # Optional: Add degree field for polynomial regression if implemented
    # polynomial_degree = forms.IntegerField(
    #     label='Grado del polinomio (Regresión)',
    #     min_value=1,
    #     required=False,
    #     widget=forms.NumberInput(attrs={'step': 1})
    # )

    def clean_points(self):
        data = self.cleaned_data['points']
        points_list = []
        lines = data.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                x_str, y_str = line.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                points_list.append((x, y))
            except ValueError:
                raise forms.ValidationError(f"Error en la línea {i+1}: Formato inválido '{line}'. Use 'x,y'.")

        if len(points_list) < 2:
            raise forms.ValidationError("Se necesitan al menos 2 puntos.")

        # Check for duplicate x values for interpolation methods
        method = self.cleaned_data.get('method')
        if method in ['lagrange', 'newton']:
            x_values = [p[0] for p in points_list]
            if len(x_values) != len(set(x_values)):
                raise forms.ValidationError("Los métodos de interpolación requieren valores de 'x' únicos.")

        return points_list

    # Optional validation for polynomial degree
    # def clean(self):
    #     cleaned_data = super().clean()
    #     method = cleaned_data.get("method")
    #     degree = cleaned_data.get("polynomial_degree")
    #     points = cleaned_data.get("points")
    #
    #     if method == 'polynomial_regression':
    #         if degree is None:
    #             raise forms.ValidationError("Debe especificar el grado para la regresión polinomial.")
    #         if points and degree >= len(points):
    #              raise forms.ValidationError("El grado del polinomio debe ser menor que el número de puntos.")
    #
    #     return cleaned_data 