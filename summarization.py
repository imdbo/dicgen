import spacy
import os
import gensim
import numpy as np
from vocabulary import Vocabulary
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim import matutils
from tqdm import tqdm
import time
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
#https://github.com/dmmiller612/bert-extractive-summarizer

"""
    different techniques
    all models included must be pre-trained. Training our own models is out of reach.
    1- simpler textrank text summarization from wikipedia's article.
    2- bert-based summarization of wikipedia's article.
    3- make use of the  wikipedia API (https://pypi.org/project/wikipedia/) to obtain a smaller/already summarized text for the entry 
       and  look for fixed syntactic structures that can fit a dictionary definition. DET + LEMMA + VERB (the dog has, a dog is)
    these 4 techniques take coreference into account.
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

    def top_n_lemma(self, lemma):
        """we take the most negative and most positive words from each vector 
            and construct a description around it
        """
        return  self.w2v.similar_by_word(lemma, topn=3)

    def _load_ontoloy(self):
        return
 
    def ont_category(self, lemma):
        return 
 
    def to_definitions(self):
        dictionary = {}
        for lemma in self.lemma_map:
            if lemma not in dictionary:
                dictionary[lemma] = []
            ontology = self.ont_category(lemma)
            definition = f'{ontology}. Closely related to {",".join(most_frequent)}'
            dictionary[lemma].append(definition)

        return dictionary

class Summary():
    def __init__(self, Vocabulary):
        #https://iq.opengenus.org/textrank-for-text-summarization/
        self.word_index = Vocabulary.word_index
        self.lemma_map = Vocabulary.lemma_map
        self.bert_results = {}
        
    @staticmethod
    def bert_summary(bert_model, vocabulary):
        bert_summary = {'lemma': [], 'definition': []}
        
        for lemma, articles in vocabulary.items():
            for article in articles:
                summary = bert_model(article)
                bert_summary['lemma'] = lemma
                bert_summary['definition'] = summary

    def summarize_vocabulary(self):
        results = {'lemma': [], 'definition': []}
        keys = list(self.lemma_map.keys())

        for l in tqdm(range(len(keys))):
            lemma = keys[l]
            for article in self.lemma_map[lemma]:
                ranked_text = self.calculate_similarity(self.sentences_to_matrix(article))
                results['lemma'].append(lemma)
                results['definition'].append('.'.join([article[k] for k in ranked_text]))

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
        print(results)
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

tryxs = Vocabulary()
tryxs._n_frequent_words(w2v=w2v)
tryxs.build()

text_rank = Summary(tryxs)
print(text_rank.summarize_vocabulary())