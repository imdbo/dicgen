from django.shortcuts import render, redirect
from .models import Lemma, Definition
import json
from django.views.generic import ListView
# Create your views here.
from django.http import HttpResponse
from .serializers import LemmaSerializer
from rest_framework import viewsets

LANGUAGE = 'en' #placeholder for language

def read_text_template(path):
     with open(path, 'r') as j:
        return json.load(j)

def index(request):
    return redirect("/home/")   
    
def home(request):
    texts = read_text_template('dictionary/texts/home.json')
    print(texts)
    return render(request, 'dictionary/home.html', texts[LANGUAGE])


def search_lemma(request):
    if request.method == 'GET': # If the form is submitted
        print(request)
        query_lemma = request.GET.get('search_box', None)
        print(f'query_lemma {query_lemma}')
        print(Lemma.objects.all())
        results = Lemma.objects.filter(lemma__contains=query_lemma)
        context = {'results': results}
        if results:
            texts = read_text_template('dictionary/texts/resultbox.json')
            context['texts'] = texts[LANGUAGE]
        for r in results:
            dic = {}
            for definition in r.definition.all():
                print(definition.singular)
                for t in definition.collocations.all():
                    print('----')
                    print(t.collocation)
    return render(request, 'dictionary/resultbox.html', context)

class lemma_list(viewsets.ModelViewSet):

    queryset = Lemma.objects.all()
    serializer_class = LemmaSerializer
