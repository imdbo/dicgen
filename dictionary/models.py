from django.db import models

# Create your models here.
class Collocation(models.Model):
    collocation = models.CharField
    def __str__(self):
        return self.collocation
        
class PoS_tag(models.Model):
    pos = models.CharField(max_length=20, unique=True)
    def __str__(self):
        return self.pos

class Pos_frequency(models.Model):
    pos = models.ForeignKey(PoS_tag, on_delete=models.CASCADE)
    absolute_frequency = models.IntegerField()
    def __str__(self):
        return self.pos.pos

class Example(models.Model):
    example = models.CharField(max_length=1000)
    def __str__(self):
        return self.example

class Definition(models.Model):
    definition = models.CharField(max_length=5000)
    singular = models.CharField(max_length=20)
    plural = models.CharField(max_length=20)
    local_pos_tag = models.CharField(max_length=20)
    #lemma_disambiguation = models.CharField(max_length=200, blank=True)
    def __str__(self):
        return self.definition

class Context_token(models.Model):
    token = models.CharField(max_length=200)
    similarity = models.FloatField()
    def __str__(self):
        return self.token

class Lemma (models.Model):
    lemma = models.CharField(max_length=200)
    definition = models.ManyToManyField(Definition)
    positive_lemma = models.ManyToManyField(Context_token, blank=True, related_name="positive_lemma")
    negative_lemma = models.ManyToManyField(Context_token, blank=True, related_name="negative_lemma")
    global_pos_tag = models.ManyToManyField(Pos_frequency)
    frequency_w2v = models.IntegerField(null=True)
    disambiguations = models.ManyToManyField(Definition, blank=True, related_name="disambiguation_definition")
    examples = models.ManyToManyField(Example)
    def __str__(self):
        return self.lemma