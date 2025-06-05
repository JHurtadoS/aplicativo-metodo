from django import forms
import sympy as sp

# Opciones para métodos de derivación numérica
DERIVATION_METHOD_CHOICES = [
    ('forward', 'Diferencia hacia adelante - O(h)'),
    ('backward', 'Diferencia hacia atrás - O(h)'),
    ('central', 'Diferencia central - O(h²)'),
    ('second_derivative', 'Segunda derivada central - O(h²)'),
    ('five_point', 'Segunda derivada 5 puntos - O(h⁴)'),
    ('richardson', 'Extrapolación Richardson - O(h⁴)'),
]

# Opciones para métodos de integración numérica
INTEGRATION_METHOD_CHOICES = [
    ('trapezoidal', 'Regla del Trapecio'),
    ('simpson', 'Regla de Simpson 1/3'),
    ('midpoint', 'Regla del Punto Medio'),
]

# Opciones para métodos de EDO
EDO_METHOD_CHOICES = [
    ('euler', 'Método de Euler'),
    ('rk4', 'Runge-Kutta 4to orden'),
]

class DerivacionForm(forms.Form):
    """Formulario para derivación numérica"""
    function = forms.CharField(
        label='Función f(x)',
        widget=forms.TextInput(attrs={
            'placeholder': 'Ej: x**2 + 3*x + 1, sin(x), exp(x)',
            'class': 'w-full'
        }),
        help_text='Ingrese la función usando sintaxis de Python/SymPy. Use ** para potencias, sin(), cos(), exp(), log(), etc.'
    )
    
    x0 = forms.FloatField(
        label='Punto x₀',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full'}),
        help_text='Punto donde calcular la derivada aproximada.'
    )
    
    h = forms.FloatField(
        label='Paso h',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full', 'min': '0.0001'}),
        help_text='Tamaño del paso para las diferencias finitas (debe ser positivo y pequeño).'
    )
    
    method = forms.ChoiceField(
        label='Método de derivación',
        choices=DERIVATION_METHOD_CHOICES,
        widget=forms.RadioSelect,
        initial='central',
        help_text='Diferencia central suele ser más precisa.'
    )

    def clean_function(self):
        """Validar que la función sea válida con SymPy"""
        function_str = self.cleaned_data['function']
        try:
            # Intentar convertir a expresión simbólica
            x = sp.Symbol('x')
            expr = sp.sympify(function_str)
            
            # Verificar que la expresión contenga la variable x
            if not expr.has(x):
                # Si no tiene x, verificar si es una constante válida
                if not expr.is_number:
                    raise forms.ValidationError("La función debe contener la variable 'x' o ser una constante válida.")
            
            return function_str
        except Exception as e:
            raise forms.ValidationError(f"Función inválida: {str(e)}. Use sintaxis de SymPy.")

    def clean_h(self):
        """Validar que h sea positivo y no cero"""
        h = self.cleaned_data['h']
        if h <= 0:
            raise forms.ValidationError("El paso h debe ser positivo y mayor que cero.")
        if h > 10:
            raise forms.ValidationError("El paso h parece muy grande. Use un valor menor a 10.")
        return h

class IntegracionForm(forms.Form):
    """Formulario para integración numérica"""
    function = forms.CharField(
        label='Función f(x)',
        widget=forms.TextInput(attrs={
            'placeholder': 'Ej: x**2 + 3*x + 1, sin(x), exp(x)',
            'class': 'w-full'
        }),
        help_text='Función a integrar usando sintaxis de Python/SymPy.'
    )
    
    a = forms.FloatField(
        label='Límite inferior (a)',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full'}),
        help_text='Límite inferior de integración.'
    )
    
    b = forms.FloatField(
        label='Límite superior (b)',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full'}),
        help_text='Límite superior de integración.'
    )
    
    n = forms.IntegerField(
        label='Número de subintervalos (n)',
        widget=forms.NumberInput(attrs={'step': '1', 'class': 'w-full', 'min': '2'}),
        help_text='Número de subintervalos para la aproximación.'
    )
    
    method = forms.ChoiceField(
        label='Método de integración',
        choices=INTEGRATION_METHOD_CHOICES,
        widget=forms.RadioSelect,
        initial='trapezoidal'
    )

    def clean_function(self):
        """Validar función igual que en derivación"""
        function_str = self.cleaned_data['function']
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(function_str)
            if not expr.has(x) and not expr.is_number:
                raise forms.ValidationError("La función debe contener la variable 'x' o ser una constante válida.")
            return function_str
        except Exception as e:
            raise forms.ValidationError(f"Función inválida: {str(e)}. Use sintaxis de SymPy.")

    def clean_n(self):
        """Validar número de subintervalos"""
        n = self.cleaned_data['n']
        if n < 2:
            raise forms.ValidationError("Se necesitan al menos 2 subintervalos.")
        if n > 10000:
            raise forms.ValidationError("Número excesivo de subintervalos. Use un valor menor a 10000.")
        return n

    def clean(self):
        """Validaciones cruzadas"""
        cleaned_data = super().clean()
        a = cleaned_data.get('a')
        b = cleaned_data.get('b')
        n = cleaned_data.get('n')
        method = cleaned_data.get('method')

        # Verificar que a < b
        if a is not None and b is not None:
            if a >= b:
                raise forms.ValidationError("El límite superior (b) debe ser mayor que el inferior (a).")

        # Para Simpson, n debe ser par
        if method == 'simpson' and n is not None:
            if n % 2 != 0:
                raise forms.ValidationError("La regla de Simpson requiere un número par de subintervalos.")

        return cleaned_data

class EDOForm(forms.Form):
    """Formulario para Ecuaciones Diferenciales Ordinarias"""
    function = forms.CharField(
        label='Función f(t,y)',
        widget=forms.TextInput(attrs={
            'placeholder': 'Ej: y - t**2 + 1, t*y + 1, -2*y + t',
            'class': 'w-full'
        }),
        help_text='Función f(t,y) de la EDO dy/dt = f(t,y). Use "t" para tiempo e "y" para la variable dependiente.'
    )
    
    t0 = forms.FloatField(
        label='Tiempo inicial t₀',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full'}),
        help_text='Valor inicial del tiempo.'
    )
    
    y0 = forms.FloatField(
        label='Condición inicial y₀',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full'}),
        help_text='Valor inicial de y: y(t₀) = y₀.'
    )
    
    h = forms.FloatField(
        label='Paso h',
        widget=forms.NumberInput(attrs={'step': 'any', 'class': 'w-full', 'min': '0.0001'}),
        help_text='Tamaño del paso temporal.'
    )
    
    n_steps = forms.IntegerField(
        label='Número de pasos',
        widget=forms.NumberInput(attrs={'step': '1', 'class': 'w-full', 'min': '1'}),
        help_text='Número de pasos a calcular.'
    )
    
    method = forms.ChoiceField(
        label='Método numérico',
        choices=EDO_METHOD_CHOICES,
        widget=forms.RadioSelect,
        initial='euler'
    )

    def clean_function(self):
        """Validar función f(t,y)"""
        function_str = self.cleaned_data['function']
        try:
            t, y = sp.symbols('t y')
            expr = sp.sympify(function_str)
            # Verificar que la expresión sea válida (puede contener t, y o ser constante)
            return function_str
        except Exception as e:
            raise forms.ValidationError(f"Función inválida: {str(e)}. Use sintaxis de SymPy con variables 't' e 'y'.")

    def clean_h(self):
        """Validar paso temporal"""
        h = self.cleaned_data['h']
        if h <= 0:
            raise forms.ValidationError("El paso h debe ser positivo.")
        if h > 1:
            raise forms.ValidationError("El paso h parece muy grande para EDOs. Use un valor menor a 1.")
        return h

    def clean_n_steps(self):
        """Validar número de pasos"""
        n_steps = self.cleaned_data['n_steps']
        if n_steps < 1:
            raise forms.ValidationError("Se necesita al menos 1 paso.")
        if n_steps > 10000:
            raise forms.ValidationError("Número excesivo de pasos. Use un valor menor a 10000.")
        return n_steps 