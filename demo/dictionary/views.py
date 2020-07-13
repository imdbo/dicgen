from django.shortcuts import render, redirect
from .models import Lemma, Definition
import json
from django.views.generic import ListView
# Create your views here.
from django.http import HttpResponse
from .serializers import LemmaSerializer
from rest_framework import viewsets


LANGUAGE = 'en' #placeholder for language
ALL_LEMMAS = sorted([l.lemma.lower() for l in Lemma.objects.all()])


def list_lemmas_alphabetically(queried_lemma):
    q_lower = queried_lemma.lower()
    found = False
    sorted_dict = {"before": [], "after": [], "queried": queried_lemma, "found": found}
    for lemma in ALL_LEMMAS:
        if sorted_dict['found'] == False:
            if lemma != q_lower:
                sorted_dict["before"].append(lemma)
            else:
                sorted_dict['found'] = True
        else:
            sorted_dict["after"].append(lemma)
    print(found)
    return sorted_dict

def read_text_template(path):
     with open(path, 'r') as j:
        return json.load(j)

def index(request):
    return redirect("/home/")   
    
def home(request):
    texts = read_text_template('dictionary/texts/home.json')
    print(texts)
    return render(request, 'dictionary/home.html', texts[LANGUAGE])

def return_alphabeticallysorted(queried_lemma):
    return

def search_lemma(request):
    if request.method == 'GET': # If the form is submitted
        print(request)
        query_lemma = request.GET.get('search_box', None)
        print(f'query_lemma {query_lemma}')
        print(Lemma.objects.all())
        results = Lemma.objects.filter(lemma__contains=query_lemma.lower())
        context = {'results': results, 'all_lemas': ALL_LEMMAS, "alphabetical_search": list_lemmas_alphabetically(query_lemma)}
        if results:
            texts = read_text_template('dictionary/texts/resultbox.json')
            context['texts'] = texts[LANGUAGE]
    return render(request, 'dictionary/resultbox.html', context)

class lemma_list(viewsets.ModelViewSet):

    queryset = Lemma.objects.all()
    serializer_class = LemmaSerializer
