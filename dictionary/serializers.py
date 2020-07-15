from .models import Lemma
from rest_framework import serializers

class LemmaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lemma
        fields = ('lemma', 'definition', 'vector_lemmas')
        depth = 2

    