from django import template
import re 
register = template.Library()
@register.filter(name='replace')
def replace(text, mode):
    if mode == 'punctuation':
        text = text.replace('.', '')
        text = text.replace(',', '')
    return text