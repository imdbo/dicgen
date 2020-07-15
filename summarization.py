import os
import gensim
from vocabulary_multi import Vocabulary
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd 
from django.conf import settings
import django
from demo.settings import DATABASES, INSTALLED_APPS
settings.configure(DATABASES=DATABASES, INSTALLED_APPS=INSTALLED_APPS)
django.setup()
import math
from dictionary.models import *
from multiprocessing import Process

number_max_lemmas = 100000
chunksize = 10000
cores = 3
if __name__ == '__main__':  
    w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec.1',  unicode_errors='ignore', binary=False, limit=number_max_lemmas)  
    vocab = Vocabulary(w2v= w2v)
    vocab._n_frequent_words()

    imported = []

    stats = { "number_lemmas": 0, "total_definitions": 0, "lemmas_inflected": 0, "disambiguations":0}
    for df in pd.read_csv('data.csv', low_memory=False, skiprows=range(1,100000), chunksize=chunksize):


        chunkedchunk = math.floor(chunksize/cores)
        first_new_row = 0 
        headwords = list(df['headword'])
        full_articles = list(df['long_entry'])
        vocab.build([headwords, full_articles])
        '''
        split_df = []
        to_pass = 0
        
        while first_new_row < chunksize:
            print(chunkedchunk)
            split_df.append([headwords[first_new_row:first_new_row+chunkedchunk], full_articles[first_new_row:first_new_row+chunkedchunk]])
            first_new_row += chunkedchunk
            parallel_retrieval = Process(target = vocab.build, args=(split_df[to_pass],))
            parallel_retrieval.start()
            to_pass += 1
            parallel_retrieval.join()
        '''
        for lemma, data in vocab.lemma_map.items():
            if lemma not in imported:
                imported.append(lemma)
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

                negative_n_lemmas = [l for l in vocab.top_n_lemmas(lemma_lower, n=5, negative=True)]
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