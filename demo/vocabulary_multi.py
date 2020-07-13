import re
import numpy as np
import pandas as pd
import spacy
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from dataclasses import dataclass
from tqdm import tqdm
from nltk.corpus import stopwords

from lemminflect import getLemma, getInflection
spacy_parser = spacy.load('en_core_web_lg')
spacy.prefer_gpu() #enable cuda if available.
import math
from multiprocessing import Process
import sys

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
                w2v:object, #w2v model loaded
                encoding: str = 'utf-8', 
                max_lemmas: int = 5000, 
                size_sentence: int = 15, 
                size_short_article:int = 750,
                top_n_sentences_lemma: int = 10, 
                limit_collocations: int = 3,
                data_df: str = 'data.csv'
                ):
        #--common--#
        self.encoding = encoding
        self.w2v = w2v
        self.word_index = {'<#PAD#>':0, '.': 1, ',':2} #word index for retrieval 
        self.lemma_index = {} # index of all lemas
        self.word_count = 2
        self.size_short_article = size_short_article
        self.max_lemmas = max_lemmas
        self.full_articles = np.empty([max_lemmas, top_n_sentences_lemma],dtype=int)
        self.lemma_map = {}
        self.lemma_inflection = {}
        self.limit_collocations = limit_collocations
        self.data_df = data_df
        self.stop_words = set(stopwords.words('english'))
        self.collocations = {} 
        self.pos_tags = {}
        self.words_only = re.compile(r'\w+')
        self.regex_disambiguation = re.compile(r'\(\w+\)')
        self.disambiguations = {}
        #self.ignore_if_in = ["may also refer to", "can refer to"]

    def top_n_lemmas(self, lemma:str, n: int, negative=False):
        """we take the most negative and most positive words from each vector 
            and construct a description around it
        """
        try:
            if negative == False:
                return  list(self.w2v.most_similar(lemma, topn=sys.maxsize))[:n]
            else:
                all_sims = self.w2v.most_similar(lemma, topn=sys.maxsize)
                last_n = list(reversed(all_sims[-n:]))
                return last_n
        except:
            return []

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

    def global_pos_collocations(self):
        '''
            1.iterate all text stored for collocations
            2.get stats for pos of each lemma in all sentences
            3.return textranked collocations and stats pos
        '''
        for lemma in self.collocations:
            pos_freqs = {}
            for t in spacy_parser('.'.join(self.collocations[lemma])): #retrieve global PoS
                if t.text.lower() == lemma.lower():
                    if t.pos_ not in pos_freqs:
                        pos_freqs[t.pos_] = 1
                    else:
                        pos_freqs[t.pos_] += 1
            self.lemma_map[lemma]['pos_freq'] = pos_freqs # global pos frequency
            #textrank collocations
            temp_collocations = self.calculate_similarity(self.sentences_to_matrix(self.collocations[lemma]), n_sentences=4)
            print(temp_collocations)
            if lemma in self.disambiguations:
                self.collocations[lemma] = [self.collocations[lemma][sentence] for sentence in temp_collocations if sentence not in self.lemma_map[lemma] and sentence not in self.disambiguations[lemma]]
            else:
                 self.collocations[lemma] = [self.collocations[lemma][sentence]  for sentence in temp_collocations if sentence not in self.lemma_map[lemma]]
            self.lemma_map[lemma]['collocations'] = self.collocations[lemma][:self.limit_collocations]

    def inflect_pos(self, hw, article):
        '''
            we check the most likely POS of the lemma
            and the inflect the form if it is a noun or adjective
            of course this is biased and we need to, at some point, account for different 
            definitions of a lemma, which may affect inflection.
            Since we are using Wikipedia to generate articles, we won't find articles as lemmas
        '''
        try:
            freq_lemma = self.lemma_index[hw.lower()]
        except:
            freq_lemma = 9999999999999999999999999
        if hw not in self.lemma_map:
            self.lemma_map[hw] = {"pos_freq": {}, "disambiguations": {}, "frequency_w2v":freq_lemma}
            self.collocations[hw] = []

        pos_occurrences = {} #store all pos of the lemma inside the article. We take the most frequent one for the inflection.
        #most_common = 'NN' # Noun by default to have something to add. 
        
        singular = hw
        plural = hw
        tag_occurrences = {}
        parsed_article = spacy_parser(article)

        tag_occurrences = {}
        for word in article.split(' '):
            if word not in self.word_index:
                self.word_count += 1
                self.word_index[word] = self.word_count
        parsed_article = spacy_parser(article)
        for token in parsed_article:
            if token.text not in self.word_index:
                self.word_count += 1
                self.word_index[token.text] = self.word_count

            if token.text.lower() == hw.lower():
                if token.tag_ not in pos_occurrences:
                    pos_occurrences[token.pos_] = 1
                    tag_occurrences[token.tag_] = 1
                else:
                    pos_occurrences[token.pos_] += 1
                    tag_occurrences[token.tag_] += 1
                lemma = token
       
        if tag_occurrences and lemma:
            most_frequent_tag = max(tag_occurrences, key=lambda k: tag_occurrences[k])
            if most_frequent_tag == 'NNS' or most_frequent_tag == 'NNPS':
                    most_frequent_tag = most_frequent_tag[:-1]
                    singular = lemma._.inflect(most_frequent_tag)
            else:
                most_frequent_tag = f'{most_frequent_tag}S'
                plural = lemma._.inflect(most_frequent_tag)

        if pos_occurrences:
            most_frequent_pos = max(pos_occurrences, key=lambda k: pos_occurrences[k])
        else:
            most_frequent_pos = 'NOUN' #by defect we assume it's a noun.
        #https://pypi.org/project/inflect/0.2.4/
        return [singular,plural], most_frequent_pos

    def cosine_similarity(self, x, y):
        if len(x) > 0 and len(y) > 0:
            dot = np.dot(x, y)
            #euclidian normalization
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            return dot / (norm_x * norm_y)
        else:
            return 0.

    def calculate_similarity(self, matrix, n_sentences:int = 3):
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
        for k in list(results.keys())[:n_sentences]:
            #sorted_results[k] = similarity_matrix[k]
            sorted_results.append(k)
        #we return sentences sorted by highest score
        return sorted(sorted_results)

    def sentences_to_matrix(self, article:list):
        #find longest sentence for matrix
        split_sentences = []
        size_sentence = 0
        for sentence in article:
            if len(sentence) > 1:
                split_sentences.append(sentence.split(' '))
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
                if len(token) >= 1:
                    matrix[i][j] = self.word_index[token]
                else:
                    continue
        return matrix

    def for_collocations(self, full_article:list):
        '''
            return the most common construction. Easy way to retrieve collocations
        '''

        for sentence in full_article:
            for t in sentence.split():
                if t not in self.word_index:
                    self.word_count += 1
                    self.word_index[t] = self.word_count
                if t in self.lemma_map:
                    if t in self.collocations:
                        self.collocations[t].append(sentence)  
                    else:
                        self.collocations[t] = [sentence]

    def sanitize_text(self, text):
        split = text.split()
        to_remove = []
        for i in range(len(split)):
            try:
                if i+1 <= len(split) and split[i] == split[i+1]:
                    to_remove.append(i)
            except:
                continue
        split =[split[i] for i in range(len(split)) if i not in to_remove]
        split[0] = split[0].title()
        return ' '.join(split) 

    def _n_frequent_words(self, binary: bool = True):
        '''
            return list of most frequent tokens in the w2v model loaded.
        '''
        lemma_count = 0
        lemmas = list(self.w2v.vocab)
        for lemma in lemmas:
            lemma_count += 1
            self.lemma_index[lemma] = lemma_count
        print(self.lemma_index)

    def generate_def(self, headwords, full_articles):
        print(headwords)
        '''
        1. Find PoS tag of headword in the article
        2. Add new definition which includes
        {
            "lemma" : {
                "frequency_w2v": 1,
                "pos_freq" {
                    "NOUN": 93.
                    "VERB": 10  
                },
                "disambiguations": [same structure but inside]
                "definition": [{"pos": "NOUN", definition: "----"},{}]
                "collocations": [],
            }
        }
        '''
        for i in tqdm(range(len(headwords))):
            hw = headwords[i]
            l_entry = full_articles[i]
            disambiguation = False

            if isinstance(hw, str) and hw.lower() in self.lemma_index:
                if re.search(self.regex_disambiguation, hw):
                    disambiguation = True

                hw_split = hw.split()
                #if hw_split[0] in self.lemma_index:
                size_entry = 0 
                cut_entry = []
                full_article = l_entry
                l_entry = l_entry.replace('\n', ' ')
                l_entry = l_entry.replace(',', ' ')
                l_entry = self.sanitize_text(l_entry)
                inflection, current_pos  = self.inflect_pos(hw, l_entry)

                l_entry = l_entry.split('.')

                for sentence in l_entry:
                    if size_entry < self.size_short_article and "may refer to" not in sentence:
                        size_entry += len(sentence)
                        cut_entry.append(sentence)

                self.for_collocations(l_entry)

                for i in range(len(cut_entry)):
                    sentence = re.findall(self.words_only, cut_entry[i])
                    for word in sentence:
                        if word not in self.word_index:
                            self.word_count += 1
                            self.word_index[word] = self.word_count
                    cut_entry[i] = ' '.join(sentence)
                #cut_entry = '.'.join(cut_entry)
                new_entry = {"definition": '.'.join([cut_entry[k] for k in self.calculate_similarity(self.sentences_to_matrix(cut_entry))]),
                    "pos_stats": current_pos
                }

                if disambiguation == False:
                    if "definitions" in self.lemma_map[hw]:
                        self.lemma_map[hw]["definitions"].append(new_entry)
                    else:
                        self.lemma_map[hw]["definitions"] = [new_entry]
                else:
                    if "definitions" not in  self.lemma_map[hw]["disambiguations"]:
                        self.disambiguations[hw]["disambiguations"]["definitions"] = new_entry
                    else:
                        self.disambiguations[hw]["disambiguations"]["definitions"].append(new_entry)
                if 'inflection' not in self.lemma_map[hw]:
                    self.lemma_map[hw]['inflection'] = inflection

    def build(self, cores:int= 3, chunksize:int=10000):
        '''
        1. read buffer of csv with pandas
        2. TODO split into processes to speed up
        3. send to generate_def
        4. after parsing whole dataframe. Find collocations
        5. retrieve close lemmas from word2vec model
        '''
        full_df = pd.read_csv(self.data_df, low_memory=False, nrows=chunksize)
        #for df in full_df:
        print(df)
        headwords = list(df['headword'])
        full_articles = list(df['long_entry'])
        #for df in full_df:
        self.generate_def(headwords=headwords, full_articles=full_articles)
        '''
        call collocation generation and pos tagging for all the dataframe parsed
        '''
        self.global_pos_collocations()

    @staticmethod
    def reverse_index(word_index) -> dict: return word_index.__class__(map(reversed, word_index.items()))

if __name__ == "__main__":
    tryxs = Vocabulary()
    tryxs.build()
