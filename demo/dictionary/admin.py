from django.contrib import admin
from .models import Lemma, Definition, Collocation
# Register your models here.

admin.site.register(Lemma)
admin.site.register(Definition)
admin.site.register(Collocation)