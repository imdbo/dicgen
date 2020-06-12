from django.shortcuts import render, redirect
from .models import Lemma
import json
# Create your views here.
from django.http import HttpResponse

def read_text_template(path):
     with open(path, 'r') as j:
        return json.load(j)

def index(request):
    return redirect("/home/")
    
def home(request):
    texts = read_text_template('dictionary/texts/home.json')
    print(texts)
    return render(request, 'dictionary/home.html', texts)


def search_lemma(request):
    if request.method == 'GET': # If the form is submitted
        query_lemma = request.GET.get('search_box', None)
        results = Lemma.objects.all().filter(lemma=query_lemma)
    return results