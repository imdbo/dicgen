import spacy
import os
import gensim
import numpy as np
from vocabulary import Vocabulary
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

#https://github.com/dmmiller612/bert-extractive-summarizer

"""
    different techniques
    all models included must be pre-trained. Training our own models is out of reach.
    1- textrank text summarization from wikipedia's article.
    x- bert-based summarization of wikipedia's article. // Not really relevant
    4- w2v based. We construct the definition by means of cosine distance and attending to the unique words in the embedding when compared to the closest lemma(case-sensitive lemma-based w2v model)
    an ontology is needed? NLTK's wordnet?
"""
class Vectors():
    """
        vector based defition construction class.
        Method 4
    """
    def __init__(self, Vocabulary, w2v, pagerank_damping= 0.85):
        self.w2v = w2v
        self.word_index = Vocabulary.word_index
        self.lemma_map = Vocabulary.lemma_map
        self.lemma_inflection = Vocabulary.lemma_inflection
        self.pagerank_damping = pagerank_damping

    def top_n_lemma(self, lemma:str, n: int):
        """we take the most negative and most positive words from each vector 
            and construct a description around it
        """
        return  self.w2v.similar_by_word(lemma, topn=n)

    def _load_ontoloy(self):
        # TODO: find some ontology (WordNet) to Improve definitions and include synonymity
        return
 
    def ont_category(self, lemma: str):
        return 
 
    def to_definitions(self, n:int):
        '''
            for now we take the top n closest lemmas in the vector of the lemma and add return them
            the main definition comes from a summary of wikipedia's article on the lemma.
        '''
        map_top_n = {}
        for lemma in self.lemma_map:
            if lemma not in map_top_n:
                map_top_n[lemma] = self.top_n_lemma(lemma,n)
        return map_top_n

class Summary():
    def __init__(self, Vocabulary):
        #https://iq.opengenus.org/textrank-for-text-summarization/
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

    def summarize_vocabulary(self):
        results = {}
        keys = list(self.lemma_map.keys())

        for l in tqdm(range(len(keys))):
            lemma = keys[l]
            for article in self.lemma_map[lemma]:
                ranked_text = self.calculate_similarity(self.sentences_to_matrix(article))
                print(ranked_text)
                if lemma not in results:
                    results[lemma] = {}
                    results[lemma]['definitions'] = ['.'.join([article[k] for k in ranked_text])]
                else:
                    results[lemma]['definitions'].append('.'.join([article[k] for k in ranked_text]))

        return results


if __name__ == '__main__':  

    number_lemmas = 3000
    w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec.1',  unicode_errors='ignore', binary=False, limit=number_lemmas)  
    #'ententen13_tt2_1.vec.2' lowercase

    map_import = {} # map with the entire   
    vocab = Vocabulary()
    vocab._n_frequent_words(w2v=w2v, number_lemmas=number_lemmas)
    vocab.build()
    vector_parser = Vectors(vocab,w2v)

    map_vector_lemmas = vector_parser.to_definitions(1000)

    text_rank = Summary(vocab)
    text_rank_summaries = text_rank.summarize_vocabulary()

    for lemma in map_vector_lemmas:
        map_import[lemma] = {}
        map_import[lemma]["vector_lemmas"] = map_vector_lemmas[lemma]
        map_import[lemma]['definitions'] = text_rank_summaries[lemma]['definitions']
        map_import[lemma]['inflection'] = vocab.lemma_inflection[lemma]
        map_import[lemma]['collocations'] = vocab.collocations[lemma]

    for lemma, data in map_import.items():
        singular = data['inflection']
        plural = data['inflection']
        collocations = [Collocation.objects.get_or_create(collocation=c) for c in data['collocations']]

        for d in data['definitions']:
            new_def = Definition.objects.get_or_create(
                definition = d,
                singular = singular,
                plural = plural,
            )
            #new_def.collocations.add(Lemma.objects.get(collocation=c) for callable in collocations)
            print(new_def)
            print('----')
            new_def = new_def[0]
            print(new_def.id)
            for c in collocations:
                new_def.collocations.add(c)
            #Places.objects.get(name='kansas')
            #print place.id
            le = Lemma.objects.get_or_create(
                lemma = lemma, 
                )
            le[0].definition.add(new_def) 
            print(le)
    #vector lemmas after all imported

    for lemma, data in map_import.items():
        vector_lemmas = [l for l in data['vector_lemmas'] if l in map_import]
        if vector_lemmas:
            lemma = Lemma.objects.get(lemma=lemma)
            lemma.vector_lemmas.add(Lemma.objects.get(lemma=l) for l in vector_lemmas)