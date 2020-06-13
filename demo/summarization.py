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
    def __init__(self, Vocabulary, w2v):
        self.w2v = w2v
        self.word_index = Vocabulary.word_index
        self.lemma_map = Vocabulary.lemma_map
        self.lemma_inflection = Vocabulary.lemma_inflection

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

    def cosine_similarity(self, x, y):
        if len(x) > 0 and len(y) > 0:
            dot = np.dot(x, y)
            #euclidian normalization
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            return dot / (norm_x * norm_y)
        else:
            return 0.

    def calculate_similarity(self, matrix):
        '''
            generate similarity matrix for all sentences in the article
            where x is the array index and each y column represents one of the indexed sentences
            yxxxxxxxxxxxxxxxx
            yxxxxxxxxxxxxxxxx
            yxxxxxxxxxxxxxxxx
            yxxxxxxxxxxxxxxxx
            yxxxxxxxxxxxxxxxx
            yxxxxxxxxxxxxxxxx
        '''
        # TODO: fix here
        import time
        similarity_matrix = np.zeros((len(matrix), len(matrix)),dtype=np.float32)
        for i in range(len(matrix)):
            array = matrix[i]
            for j in range(len(matrix)):
                _array = matrix[j]
                similarity = self.cosine_similarity(array, _array)
                #print(f'similarity {similarity}')
                similarity_matrix[i][j] = similarity
        sort = {x: sum(similarity_matrix[x])/len(similarity_matrix[x]) for x in range(len(similarity_matrix))}
        results = {k: v for k, v in sorted(sort.items(), key=lambda item: item[1], reverse=True)}
        sorted_results = []
        for k in list(results.keys())[:3]:
            #sorted_results[k] = similarity_matrix[k]
            sorted_results.append(k)
        #we return sentences sorted by highest score
        return sorted(sorted_results)

    def sentences_to_matrix(self, article):
        #find longest sentence for matrix
        split_sentences = []
        size_sentence = 0
        for sentence in article:
            if len(sentence) > 1:
                sentence = sentence.split(' ')
                split_sentences.append(sentence)
                sentence_len = len(sentence)

                if sentence_len > size_sentence:
                    size_sentence = sentence_len 
            '''
                create words to fit all sentences padding smaller ones
                we pass this to calculate similarity
            '''
        #fit sentences to numpy matrix
        matrix = np.zeros((len(split_sentences), size_sentence), dtype=int)
        for i in range(len(split_sentences)):
            sentence = split_sentences[i]
            for j in range(len(sentence)):
                token = sentence[j]
                matrix[i][j] = self.word_index[token]
        return matrix

w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec',  unicode_errors='ignore', binary=False, limit=500000)  
#'ententen13_tt2_1.vec.2' lowercase

map_import = {} # map with the entire   
vocab = Vocabulary()
vocab._n_frequent_words(w2v=w2v)
vocab.build()
vector_parser = Vectors(vocab,w2v)

map_vector_lemmas = vector_parser.to_definitions(10)

text_rank = Summary(vocab)
text_rank_summaries = text_rank.summarize_vocabulary()

for lemma in map_vector_lemmas:
    map_import[lemma] = {}
    map_import[lemma]["vector_lemmas"] = map_vector_lemmas[lemma]
    map_import[lemma]['definitions'] = text_rank_summaries[lemma]['definitions']
    map_import[lemma]['inflection'] = vocab.lemma_inflection[lemma]
    map_import[lemma]['collocations'] = ['test', 'test']

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

        #Places.objects.get(name='kansas')
        #print place.id
        le = Lemma.objects.get_or_create(
            lemma = lemma, 
            )
        le[0].definition.add(new_def) 

#vector lemmas after all imported

for lemma, data in map_import.items():
    vector_lemmas = [l for l in data['vector_lemmas'] if l in map_import]
    if vector_lemmas:
        lemma = Lemma.objects.get(lemma=lemma)
        lemma.vector_lemmas.add(Lemma.objects.get(lemma=l) for l in vector_lemmas)