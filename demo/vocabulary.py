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
from multiprocessing import Process
from lemminflect import getLemma, getInflection
spacy_parser = spacy.load('en_core_web_lg')
spacy.prefer_gpu() #enable cuda if available.
import math
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
                max_lemmas: int = 300, 
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
        self.collocations = {} 
        self.pos_tags = {}
        self.words_only = re.compile(r'\w+')

    def inflect(self, lemma, article):
        '''
            we check the most likely POS of the lemma
            and the inflect the form if it is a noun or adjective
            of course this is biased and we need to, at some point, account for different 
            definitions of a lemma, which may affect inflection.
            Since we are using Wikipedia to generate articles, we won't find articles as lemmas
        '''
        pos_occurrences = {} #store all pos of the lemma inside the article. We take the most frequent one for the inflection.
        most_common = 'NN' # Noun by default to have something to add. 
        tag_occurrences = {}
        parsed_article = spacy_parser(article)
        for token in parsed_article:
            if token.lemma_ == lemma:
                if token.tag_ not in pos_occurrences:
                    pos_occurrences[token.pos_] = 1
                    tag_occurrences[token.tag_] = 1
                else:
                    pos_occurrences[token.pos_] += 1
                    tag_occurrences[token.tag_] += 1

        if tag_occurrences:
            most_common_tags =  max(tag_occurrences, key=lambda k: tag_occurrences[k])
            #if most_common == 'VERB' or most_common == 'NOUN':

            inflection = getInflection((lemma), tag=most_common_tags)
        else:
            inflection = ''
        self.pos_tags[lemma] = pos_occurrences
        #https://pypi.org/project/inflect/0.2.4/
        return inflection

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

    def for_collocations(self, lemma, full_article):
        '''
            return the most common construction. Easy way to retrieve collocations
        '''
        for sentence in full_article:
            for t in sentence.split():
                if t in self.lemma_map:
                    if t in self.collocations:
                        self.collocations[t].append(sentence)  
                    else:
                        self.collocations[t] = [sentence]

    def extract_collocations(self, limit:int = 3):
        for lemma, sentences in self.collocations.items():
            sorted_sentences = self.calculate_similarity(self.sentences_to_matrix(sentences))
            self.collocations[lemma] = sorted_sentences[:limit]
     
    @staticmethod
    def sanitize_text(text):
        split = text.split()
        to_remove = []
        for i in range(len(split)):
            if i+1 <= len(split) and split[i] == split[i+1]:
                to_remove.append(i)
        split =[split[i] for i in range(len(split)) if i not in to_remove]
        split[0] = split[0].title()
        return split.join()

    def _n_frequent_words(self, w2v:object, binary: bool = True, number_lemmas:int = 1000):
        '''
            return list of most frequent tokens in the w2v model loaded.
        '''
        lemmas = list(w2v.vocab)
        for lemma in lemmas:
            if len(self.lemma_index) < self.max_lemmas and lemma.isnumeric() == False:
                    if lemma not in self.stop_words:
                        self.lemma_count += 1
                        self.lemma_index[lemma] = self.lemma_count

        print(self.lemma_index)
    def parse_df(self, df):
        headwords = df['headword']
        full_articles = df['long_entry']
        '''
        index table for all words
        we store articles up to the limit specified
        create dataframe with lemmas and all entries for each
        '''
        for i in tqdm(range(len(headwords))):
            hws = headwords[i]
            l_entry = full_articles[i]
            if isinstance(hws, str):
                hws = hws.split()
                for hw in hws:
                    if hw in self.lemma_index:
                        if hw not in self.lemma_map:
                            self.lemma_map[hw] = []
                            self.collocations[hw] = []
                            
                        max_size_f_entry = 0 
                        cut_entry = []
                        l_entry = l_entry.replace('\n', ' ')
                        l_entry = l_entry.split('.')
                        self.for_collocations(hw, l_entry)
                        for sentence in l_entry:
                            if max_size_f_entry < self.size_short_article:
                                max_size_f_entry += len(sentence)
                                cut_entry.append(sentence)
                        l_entry = cut_entry 
                        
                        for i in range(len(l_entry)):
                            sentence = re.findall(self.words_only, l_entry[i])
                            for word in sentence:
                                if word not in self.word_index:
                                    self.word_count += 1
                                    self.word_index[word] = self.word_count
                            l_entry[i] = ' '.join(sentence)
                        
                        self.lemma_map[hw].append(l_entry)
                        if hw not in self.lemma_inflection:
                            self.lemma_inflection[hw] = self.inflect(hw, full_articles[i])

    def build(self, cores:int= 10, chunksize:int=100000):
        full_df = pd.read_csv(self.data_df, low_memory=False, chunksize=chunksize)
        chonky = math.floor(chunksize/cores) # amount of lines passed to each process. governed by number of cores
        print(chonky)
        for df in full_df:
            chopped_df = {}
            headwords = list(df['headword'])
            full_articles = list(df['long_entry'])
            core = 1
            while core <= cores:
                print (core )
                chopped_df[core] = {
                    'headword' : list(headwords[(core-1)*chonky: core*chonky]),
                    'long_entry' : list(full_articles[(core-1)*chonky: core*chonky])
                    }
                p = Process(target = self.parse_df, args= (chopped_df[core],))
                p.start()
                core += 1
                p.join()
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
