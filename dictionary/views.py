from django.shortcuts import render, redirect
from .models import Lemma, Definition
import json
from django.views.generic import ListView
# Create your views here.
from django.http import HttpResponse
from .serializers import LemmaSerializer
from rest_framework import viewsets
from .levenshtein import Levenshtein
import math
distance = Levenshtein().find_distance

LANGUAGE = 'en' #placeholder for language
ALL_LEMMAS = sorted([l.lemma.lower() for l in Lemma.objects.all()])


def list_lemmas_alphabetically(queried_lemma):
    q_lower = queried_lemma.lower()

    sorted_dict = {"before": [], "after": [], "queried": queried_lemma, "found": False}
    for lemma in ALL_LEMMAS:
        if sorted_dict['found'] == False:
            if lemma != q_lower:
                sorted_dict["before"].append(lemma)
            else:
                sorted_dict['found'] = True
        else:
            sorted_dict["after"].append(lemma)
    if not sorted_dict['found']:
        sorted_dict["after"] = sorted_dict['before'][math.floor(len(sorted_dict['before'])/2):]
        sorted_dict["before"] = sorted_dict["before"][:math.floor(len(sorted_dict['before'])/2)]

    return sorted_dict

def read_text_template(path):
     with open(path, 'r') as j:
        return json.load(j)

def index(request):
    return redirect("/home/")   
    
def home(request):
    context = {}
    texts = read_text_template('dictionary/texts/home.json')
    context['texts'] = texts[LANGUAGE]
    context['all_lemmas'] = ALL_LEMMAS
    return render(request, 'dictionary/home.html', context)

def return_alphabeticallysorted(queried_lemma):
    return

def search_lemma(request):
    if request.method == 'GET': # If the form is submitted
        results = []
        context = {}
        mode = ''
        found = True
        query_lemma = request.GET.get('search_box', None)
        if query_lemma != '':
            lowered = query_lemma.lower()
            
            print(f'query_lemma {query_lemma}')
            if query_lemma != '' and lowered in ALL_LEMMAS:
                results = Lemma.objects.all().filter(lemma=lowered)
                if not results:
                    results = Lemma.objects.all().filter(lemma=query_lemma) 
                mode = 'exact match'
            if not results:
                close_matches = distance(word_1=lowered, lemma_list=[l for l in ALL_LEMMAS if isinstance(l, str) and len(l) < len(lowered)+2 and l[0] == lowered[0]])
                mode = 'partial match' 
                if close_matches:
                    results = Lemma.objects.all().filter(lemma__in=close_matches)
                else:
                    mode = 'the word containes the original search input'
                    results = Lemma.objects.filter(lemma__contains=lowered)
            if not results:
                results = [f'no match found for {query_lemma}']
                mode = ''
                found = False
        else:
            found = False

        context = {'results': results, 'all_lemmas': ALL_LEMMAS, 'search_type': mode, 'found': found, 'last_queried': lowered}
        texts = read_text_template('dictionary/texts/resultbox.json')
        context['texts'] = texts[LANGUAGE]
    return render(request, 'dictionary/resultbox.html', context)

class lemma_list(viewsets.ModelViewSet):

    queryset = Lemma.objects.all()
    serializer_class = LemmaSerializer
