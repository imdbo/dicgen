from django.db import models

# Create your models here.

class Top_common_lemmas(models.Model):
    lemma = models.CharField(max_length=200)
    def __str__(self):
        return self.lemma
class Definition(models.Model):
    definition = models.CharField(max_length=5000)
    def __str__(self):
        return self.definition