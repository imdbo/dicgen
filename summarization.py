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
import time 
number_max_lemmas = 300000
cores = 6
make_smaller_df = False
'''
    to retrieve examples and examples from the entire
    wikipedia, once the lemmas have been generated.
'''

if __name__ == '__main__':
    w2v = KeyedVectors.load_word2vec_format(
        'ententen13_tt2_1.vec.1',  unicode_errors='ignore', binary=False, limit=number_max_lemmas)
    vocab = Vocabulary(w2v=w2v)
    vocab.word2index = {token: token_index for token_index, token in enumerate(w2v.index2word)} 
    imported = []

    stats = {"number_lemmas": 0, "total_definitions": 0,
             "lemmas_inflected": 0, "disambiguations": 0}
             
    vocab.lemma_map = {}      
    if make_smaller_df:
        lemmas = list(w2v.vocab)
        smaller_df = {'headword': [], 'long_entry': []}
        for hw, entry in zip(df['headword'], df['long_entry']):
            if isinstance(hw, str):
                hw = hw.lower() 
                if hw in lemmas:
                        print(hw)
                        smaller_df['headword'].append(hw)
                        smaller_df['long_entry'].append(entry)
        pd.DataFrame.from_dict(smaller_df).to_csv('smaller_df.csv')

    df = pd.read_csv('smaller_df.csv', low_memory=False)
    headwords = df['headword']
    full_articles = df['long_entry']
    vocab.generate_def(headwords, full_articles)

    ''' 
    total_length = len(headwords)
    chunkedchunk = math.floor(total_length/cores)
    first_new_row = 0 
    split_df = []
    to_pass = 0
    
    while first_new_row < total_length:
        print(chunkedchunk)
        split_df.append([headwords[first_new_row:first_new_row+chunkedchunk], full_articles[first_new_row:first_new_row+chunkedchunk]])
        first_new_row += chunkedchunk
        parallel_retrieval = Process(target = vocab.generate_def, args=(split_df[to_pass][0],split_df[to_pass][1]))
        parallel_retrieval.start()
        to_pass += 1
    '''
    for lemma, data in vocab.lemma_map.items():
        if lemma not in imported:
            imported.append(lemma)
            if 'definitions' in data:
                stats['number_lemmas'] += 1
                if 'inflection' in data:
                    singular = data['inflection'][0]
                    plural = data['inflection'][1]
                    stats["lemmas_inflected"] += 1
                    
                definitions = [Definition.objects.get_or_create(
                    definition=d['definition'],
                    singular=singular,
                    plural=plural,
                    local_pos_tag=d['pos_stats']) for d in data["definitions"]]

                stats["total_definitions"] += len(definitions)
                #new_def.examples.add(Lemma.objects.get(example=c) for callable in examples)
                print(lemma)
                for d in definitions:
                    print(d[0].definition)
                print('----')

                le = Lemma.objects.get_or_create(
                    lemma=lemma
                )
                # Places.objects.get(name='kansas')
                # print place.id
                lemma_lower = le[0].lemma.lower()

                top_n_lemmas = [l for l in vocab.top_n_lemmas(
                    lemma_lower, n=5, negative=False)]
                if top_n_lemmas:
                    for l in top_n_lemmas:
                        context_token = Context_token.objects.get_or_create(
                            token=l[0],
                            similarity=l[1]
                        )
                        le[0].positive_lemma.add(context_token[0].id)

                negative_n_lemmas = [l for l in vocab.top_n_lemmas(
                    lemma_lower, n=5, negative=True)]
                if negative_n_lemmas:
                    for l in negative_n_lemmas:
                        context_token = Context_token.objects.get_or_create(
                            token=l[0],
                            similarity=l[1]
                        )
                        le[0].negative_lemma.add(context_token[0].id)

                disambiguations = [Definition.objects.get_or_create(
                    definition=d['definition'],
                    singular=singular,
                    plural=plural,
                    local_pos_tag=d['pos_stats']
                ) for d in data["disambiguations"]]
                stats["disambiguations"] += len(disambiguations)

                for disambiguation in disambiguations:
                    le[0].disambiguations.add(disambiguation[0].id)
                for d in definitions:
                    le[0].definition.add(d[0].id)

                le[0].frequency_w2v = data['frequency_w2v']

'''   
vocab.global_pos_examples()
for lemma, data in vocab.lemma_map.items():
global_pos_tags = []
le = Lemma.objects.get_or_create(lemma = lemma)

examples = [Example.objects.get_or_create(example=c) for c in data['examples']]

for c in examples:
    print(c[0].example)
    le[0].examples.add(c[0].id)
    print(stats)

tags_to_import = []
for k in data['pos_freq']:
    global_pos_tag = PoS_tag.objects.get_or_create (
    pos = k)
    tags_to_import.append(global_pos_tag)

for k in data['pos_freq']:
    k = PoS_tag.objects.get(pos=k)
    pos_freqs.append(Pos_frequency.objects.get_or_create (
    pos = k,
    absolute_frequency = data['pos_freq'][k.pos]))


for tag in pos_freqs:
    le[0].global_pos_tag.add(tag[0].id)
'''