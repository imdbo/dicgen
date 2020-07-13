import spacy
import os
import gensim
import numpy as np
from vocabulary_multi import Vocabulary
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from tqdm import tqdm
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from django.conf import settings
import django
from demo.settings import DATABASES, INSTALLED_APPS
settings.configure(DATABASES=DATABASES, INSTALLED_APPS=INSTALLED_APPS)
django.setup()
from dictionary.models import *


'''
class Summary():
    #https://github.com/dmmiller612/bert-extractive-summarizer
    def __init__(self, Vocabulary):
        self.word_index = Vocabulary.word_index
        self.lemma_map = Vocabulary.lemma_map
        self.bert_results = {}
        self.cosine_similarity = Vocabulary.cosine_similarity
        self.sanitize_text = Vocabulary.sanitize_text
        self.cosine_similarity = Vocabulary.cosine_similarity
        self.sentences_to_matrix = Vocabulary.sentences_to_matrix
        self.calculate_similarity = Vocabulary.calculate_similarity
        
    @staticmethod
    def bert_summary(bert_model, vocabulary):
        bert_summary = {}
        
        for lemma, articles in vocabulary.items():
            for article in articles:
                summary = bert_model(article)
                bert_summary['lemma'] = lemma
                bert_summary['definitions'] = [summary]

'''
if __name__ == '__main__':  

    number_max_lemmas = 2500000
    w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec.1',  unicode_errors='ignore', binary=False, limit=number_max_lemmas)  
    #'ententen13_tt2_1.vec.2' lowercase

    map_import = {} # map with the entire   
    vocab = Vocabulary(w2v= w2v)
    vocab._n_frequent_words()
    vocab.build()
    #print(vocab.lemma_map)
    #map_vector_lemmas = vector_parser.to_definitions(number_lemmas)

    #vocab.global_pos_collocations()
    ''' 
    map_import[lemma] = {}
    map_import[lemma]["vector_lemmas"] = lemma
    map_import[lemma]['definitions'] = text_rank_summaries[lemma]['definitions']
    map_import[lemma]['inflection'] = vocab.lemma_inflection[lemma]
    map_import[lemma]['collocations'] = vocab.collocations[lemma]

        save stats:
        1.number of lemmas
        2.number of dqefinitions
        3.number of lemmas with inflection %
    '''

    stats = { "number_lemmas": 0, "total_definitions": 0, "lemmas_inflected": 0, "disambiguations":0}
    for lemma in vocab.lemma_map:
        print(vocab.lemma_map[lemma])

    for lemma, data in vocab.lemma_map.items():
        stats['number_lemmas'] += 1
        if data['inflection']:
            singular = data['inflection'][0]
            plural = data['inflection'][1]
            stats["lemmas_inflected"] += 1

        collocations = [Collocation.objects.get_or_create(collocation=c) for c in data['collocations']]

        definitions = [Definition.objects.get_or_create(
                definition = d['definition'],
                singular = singular,
                plural = plural,
                local_pos_tag = d['pos_stats']) for d in data["definitions"] ]

        stats["total_definitions"]+= len(definitions)
        #new_def.collocations.add(Lemma.objects.get(collocation=c) for callable in collocations)
        print(definitions)
        print('----')

        le = Lemma.objects.get_or_create(
            lemma = lemma
            )
        #Places.objects.get(name='kansas')
        #print place.id
        lemma_lower = le[0].lemma.lower()

        top_n_lemmas = [l for l in vocab.top_n_lemmas(lemma_lower, n=5, negative=False)]
        if top_n_lemmas:
             for l in top_n_lemmas:
                context_token = Context_token.objects.get_or_create(
                    token = l[0],
                    similarity = l[1]
                )
                le[0].positive_lemma.add(context_token[0].id)

        negative_n_lemmas = [l for l in vocab.top_n_lemmas(lemma_lower, n=-5, negative=True)]
        if negative_n_lemmas:
             for l in negative_n_lemmas:
                context_token = Context_token.objects.get_or_create(
                    token = l[0],
                    similarity = l[1]
                )
                le[0].negative_lemma.add(context_token[0].id)


        disambiguations = [Definition.objects.get_or_create(
            definition = d['definition'],
            singular = singular,
            plural = plural,
            local_pos_tag = d['pos_stats']
            ) for d in data["disambiguations"]]
        stats["disambiguations"] += len(disambiguations)


        global_pos_tags = [PoS_tag.objects.get_or_create(
            pos = k,
            absolute_frequency = data['pos_freq'][k]
        ) for k in data['pos_freq']]
        
        for global_pos in global_pos_tags:
            le[0].global_pos_tag.add(global_pos[0].id)
        for c in collocations:
            print(c[0].collocation)
            le[0].collocations.add(c[0].id)
        for disambiguation in disambiguations:
            le[0].disambiguations.add(disambiguation[0].id)
        for d in definitions:
            le[0].definition.add(d[0].id) 

        le[0].frequency_w2v = data['frequency_w2v']

print(stats)