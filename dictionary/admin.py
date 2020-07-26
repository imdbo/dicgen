from django.contrib import admin
from .models import Lemma, Definition, Collocation, PoS_tag, Pos_frequency, Context_token, Example
# Register your models here.

admin.site.register(Lemma)
admin.site.register(Definition)
admin.site.register(Collocation)
admin.site.register(Example)
admin.site.register(Pos_frequency)
admin.site.register(Context_token)
admin.site.register(PoS_tag)