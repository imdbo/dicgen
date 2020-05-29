import spacy
import os
import gensim
import numpy as np
from transformers import *
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
#https://github.com/dmmiller612/bert-extractive-summarizer

"""
    different techniques
    all models included must be pre-trained. Training our own models is out of reach.
    1- simpler textrank text summarization from wikipedia's article.
    2- bert-based summarization of wikipedia's article.
    3- filter out sentences without the headword. Construct smaller article, apply text summarization techniques on it.
    4- make use of the  wikipedia API (https://pypi.org/project/wikipedia/) to obtain a smaller/already summarized text for the entry 
       and  look for fixed syntactic structures that can fit a dictionary definition. DET + LEMMA + VERB (the dog has, a dog is)
    these 4 techniques take coreference into account.
    4- w2v based. We construct the definition by means of cosine distance and attending to the unique words in the embedding when compared to the closest lemma(case-sensitive lemma-based w2v model)
    an ontology is needed? NLTK's wordnet?
"""

class Vector_def():
    """
        vector based defition construction class.
        Method 4
    """
    def __init__(self, path_w2v = 'w2v/ententen13_tt2_1.vec'):
        self.w2v = KeyedVectors.load_word2vec_format(datapath(path_w2v), binary=False)  
    def top_n_lemmas(self):
        """we take top n words and their pretrained vectors
           then we extract the first part of the article from the wikipedia
           filter sentences with the headword
        """
        return





class Summarization():
    def __init__(self, max_len_definition: int = 10, custom_model: str = 'elgeish/cs224n-squad2.0-albert-large-v2', dirpath: str = 'texts/', encoding: str = 'utf-8', max_size_vocab: int = 10000 ):
        # https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
        #---lexicon---#
        self.entries = []
        self.max_size_vocab = max_size_vocab
        self.dirpath = dirpath
        self.diccionario = []
        self.encoding = encoding
        self.lemmas = np.empty([max_size_vocab,1], dtype=object)# headword of each entry
        #--summarizing module--#
        handler = CoreferenceHandler(greedyness=.4)
        custom_config = AutoConfig.from_pretrained(custom_model)
        custom_config.output_hidden_states=True
        custom_tokenizer = AutoTokenizer.from_pretrained(custom_model)
        custom_model = AutoModel.from_pretrained(custom_model, config=custom_config)
        self.model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer, sentence_handler=handler)
        

    def generate_data(self):
        # read all the files in the directory assuming the following structure
        # headword\n\n text \nheadword\n\n text
        index = 0
        print(self.dirpath)
        for root, dirs, files in os.walk(self.dirpath):
            for name in files:
                path = (root + "/" + name)
                print(path)
                with open(path, 'r', encoding=self.encoding) as f:
                    text = f.read()
                    text = text.split(' ')
                    # first word in the file is always the headword
                    headword = text[0]
                    entry = ' '.join(text[1:])
                    if index <= self.max_size_vocab:
                        self.lemmas[index] = headword
                        self.entries.append(entry)
                        index += 1
                    else:
                        break

    def summary(self):
        for i in range(len(self.entries)):
            summary = self.model(self.entries[i])
            self.diccionario.append([i, summary])
            print(summary)
            print("---------------")

    def textrank_summary(self):
        """
        añadir simplificacion texto
        """
        return

    def summary_into_def(self):
        """
            we process the summarized text produced by the model into a 
            simpler, sytantic based ruled definition.
        """
        return


Summary = Summarization(dirpath= 'texts/')
Summary.generate_data()
Summary.summary()