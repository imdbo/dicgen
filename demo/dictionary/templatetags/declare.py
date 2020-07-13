from django import template
register = template.Library()

@register.simple_tag
def declare(val=None):
  return val