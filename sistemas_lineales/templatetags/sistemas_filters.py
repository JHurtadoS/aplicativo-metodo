from django import template

register = template.Library()

@register.filter
def get_range(value):
    """
    Crea un rango de 0 a value-1.
    Útil para iterar en un bucle for en las plantillas.
    Ejemplo: {% for i in 5|get_range %}...{% endfor %}
    """
    return range(value)

@register.filter
def get_item(lst, index):
    """
    Obtiene un elemento de una lista por su índice.
    Útil para acceder a elementos de listas en plantillas.
    Ejemplo: {{ my_list|get_item:0 }}
    """
    try:
        return lst[index]
    except (IndexError, KeyError, TypeError):
        return None

@register.filter
def add(value, arg):
    """
    Suma un número a otro.
    Ejemplo: {{ 5|add:"2" }} → 7
    """
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        return value 