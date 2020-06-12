from django.db import models

# Create your models here.

'''
    lemma
    gender
    number
    pronunciation
    definition
    collocations / most frequents constructions
'''

class Collocation(models.Model):
    collocation = models.CharField(max_length=200)
    def __str__(self):
        return self.collocation

class Definition(models.Model):
    definition = models.CharField(max_length=5000)
    singular = models.CharField(max_length=20)
    plural = models.CharField(max_length=20)
    singular = models.CharField(max_length=20)
    collocations = models.ManyToManyField(Collocation)
    def __str__(self):
        return self.definition
        
class Lemma (models.Model):
    lemma = models.CharField(max_length=200)
    definition = models.ManyToManyField(Definition)
    vector_lemmas = models.ManyToManyField("self", blank=True)
    def __str__(self):
        return self.lemma