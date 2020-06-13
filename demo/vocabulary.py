import re
import numpy as np
import pandas as pd
import spacy
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from dataclasses import dataclass
from tqdm import tqdm
from nltk.corpus import stopwords
#model.wv.index2entity[:100]
from lemminflect import getLemma, getInflection
spacy_parser = spacy.load('en_core_web_lg')
spacy.prefer_gpu() #enable cuda if available.

@dataclass
class Vocabulary():
    """
        common class holding all the necessary lexical information that needs to be used in definitions
        here we create the list of words that will be added to the dictionary, and fit all the data
        top_n lemmas to be converted into dictionary format.
        we need:
            -lemma list
            -wikipedia articles for each lemma for text summarization techniques

    """
    def __init__(self,
                encoding: str = 'utf-8', 
                max_lemmas: int = 30000, 
                size_sentence: int = 15, 
                size_short_article:int = 350,
                top_n_sentences_lemma: int = 10, 
                w2v_path:str = 'w2v/ententen13_tt2_1.vec', #w2v keyedvectors path
                data_df: str = 'data.csv'):
        #--common--#
        self.encoding = encoding
        self.word_index = {'<#PAD#>':0, '.': 1, ',':2} #word index for retrieval 
        self.lemma_index = {} # index of all lemas
        self.lemma_count = 0
        self.word_count = 2
        self.size_short_article = size_short_article
        self.max_lemmas = max_lemmas
        self.full_articles = np.empty([max_lemmas, top_n_sentences_lemma],dtype=int)
        self.lemma_map = {}
        self.lemma_inflection = {}
        #--data--#
        self.data_df = data_df
        self.stop_words = set(stopwords.words('english'))

    
    def inflect(self, lemma, article):
        '''
            we check the most likely POS of the lemma
            and the inflect the form if it is a noun or adjective
            of course this is biased and we need to, at some point, account for different 
            definitions of a lemma, which may affect inflection.
            Since we are using Wikipedia to generate articles, we won't find articles as lemmas
        '''
        pos_occurences = {} #store all pos of the lemma inside the article. We take the most frequent one for the inflection.
        most_common = 'NN' # Noun by default to have something to add. 

        parsed_article = spacy_parser(article)
        for token in parsed_article:
            if token.lemma_ == lemma:
                if token.tag_ not in pos_occurences:
                    pos_occurences[token.tag_] = 1
                else:
                    pos_occurences[token.tag_] += 1
        if pos_occurences:
            most_common = max(pos_occurences, key=lambda k: pos_occurences[k])
            #if most_common == 'VERB' or most_common == 'NOUN':

        inflection = getInflection((lemma), tag=most_common)
        #https://pypi.org/project/inflect/0.2.4/
        return inflection

    def most_common_construction(self):
        '''
            return the most common construction. Easy way to retrieve collocations
        '''
        colocalltions = []
        # TODO: Process every article. return most common sentences that include the lemma, as an example.
        return colocalltions

    def _n_frequent_words(self, w2v:object, binary: bool = True, number_lemmas:int = 1000):
        '''
            return list of most frequent tokens in the w2v model loaded.
        '''
        lemmas = list(w2v.vocab)
        for lemma in lemmas:
            if len(self.lemma_index) < self.max_lemmas:
                    if lemma not in self.stop_words:
                        self.lemma_count += 1
                        self.lemma_index[lemma] = self.lemma_count
            else:
                print(self.lemma_index)
                return

    def build(self):
        words_only = re.compile(r'\w+')
        full_df = pd.read_csv(self.data_df, low_memory=False, chunksize=300000)
        for df in full_df:
            headwords = df['headword'].to_numpy()
            full_articles = df['long_entry'].to_numpy()
            '''
            index table for all words
            we store articles up to the limit specified
            create dataframe with lemmas and all entries for each
            '''
            for i in tqdm(range(len(headwords))):
                hw = headwords[i]
                l_entry = full_articles[i]

                if isinstance(hw, str) and hw in self.lemma_index:
                    if hw not in self.lemma_map:
                        self.lemma_map[hw] = []
                    max_size_f_entry = 0 
                    cut_entry = []

                    l_entry = l_entry.replace('\n', ' ')
                    l_entry = l_entry[:self.size_short_article]
                    l_entry = l_entry.split('.')

                    for sentence in l_entry:
                        if max_size_f_entry < self.size_short_article:
                            max_size_f_entry += len(sentence)
                            cut_entry.append(sentence)
                    l_entry = cut_entry 
                    
                    for i in range(len(l_entry)):
                        sentence = re.findall(words_only, l_entry[i])
                        for word in sentence:
                            if word not in self.word_index:
                                self.word_count += 1
                                self.word_index[word] = self.word_count
                        l_entry[i] = ' '.join(sentence)
                        
                    self.lemma_map[hw].append(l_entry)
                    if hw not in self.lemma_inflection:
                        self.lemma_inflection[hw] = self.inflect(hw, full_articles[i])
                '''
                for article in self.lemma_map['A']:
                    for l in article:
                        print(l)
                        print('------------')
                '''

    @staticmethod
    def reverse_index(word_index) -> dict: return word_index.__class__(map(reversed, word_index.items()))

    def index_word(self, word:str, lemma:bool):
        if lemma == False:
            if word not in self.word_index:
                self.word_count += 1
                self.word_index[word] = self.word_count
        else:
            if word not in self.word_index:
                self.word_index[word] = self.word_count
                self.lemma_count += 1
                

if __name__ == "__main__":
    tryxs = Vocabulary()
    tryxs.build()
