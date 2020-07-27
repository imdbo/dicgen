import nltk
import sys
import math
import re
import numpy as np
import pandas as pd
import spacy
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from dataclasses import dataclass
from tqdm import tqdm
from nltk.corpus import stopwords
import numba as nb
from lemminflect import getLemma, getInflection
spacy_parser = spacy.load('en_core_web_lg')
spacy.prefer_gpu()  # enable cuda if available.

@nb.jit(nopython=True, fastmath=True)
def cosine_similarity(x, y):
    xx,yy,xy=0.0,0.0,0.0
    for i in range(len(x)):
        xx+=x[i]*x[i]
        yy+=y[i]*y[i]
        xy+=x[i]*y[i]
    return 1.0-xy/np.sqrt(xx*yy)

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
                 w2v: object,  # w2v model loaded
                 encoding: str = 'utf-8',
                 size_sentence: int = 15,
                 size_short_article: int = 2000,
                 top_n_sentences_lemma: int = 10,
                 limit_examples: int = 3,
                 ):
        #--common--#
        self.encoding = encoding
        self.w2v = w2v
        self.word_index = {'<#PAD#>': 0, '.': 1,',': 2}  # word index for retrieval
        self.lemma_index = {}  # index of all lemas
        self.word_count = 2
        self.size_short_article = size_short_article
        self.full_articles = []
        self.lemma_map = {}
        self.word2index = {}
        self.lemma_inflection = {}
        self.limit_examples = limit_examples
        self.stop_words = set(stopwords.words('english'))
        self.examples = {}
        self.pos_tags = {}
        self.words_only = re.compile(r'\w+')
        self.regex_disambiguation = re.compile(r'\(\w+\)')
        self.digits_filter = re.compile(r'\d+')
        self.disambiguations = {}
        #self.ignore_if_in = ["may also refer to", "can refer to"]

    def nltk_collocation(self, lemma, article):
        for lemma, in self.lemma_map:
            article = self.lemma_map[lemma]
            nltkd = nltk.Text(tkn for tkn in sentence.split()
                              for sentence in article.split('.'))

    def top_n_lemmas(self, lemma: str, n: int, negative=False):
        """we take the most negative and most positive words from each vector 
            and construct a description around it
        """
        try:
            if negative == False:
                return list(self.w2v.most_similar(lemma, topn=sys.maxsize))[:n]
            else:
                all_sims = self.w2v.most_similar(lemma, topn=sys.maxsize)
                last_n = list(reversed(all_sims))[-n:]
                return last_n
        except:
            return []

    def to_definitions(self, n: int):
        '''
            for now we take the top n closest lemmas in the vector of the lemma and add return them
            the main definition comes from a summary of wikipedia's article on the lemma.
        '''
        map_top_n = {}
        for lemma in self.lemma_map:
            if lemma not in map_top_n:
                map_top_n[lemma] = self.top_n_lemma(lemma, n)
        return map_top_n

    def global_pos_examples(self):
        '''
            1.iterate all text stored for examples
            2.get stats for pos of each lemma in all sentences
            3.return textranked examples and stats pos
        '''
        for lemma in self.examples:
            pos_freqs = {}
            for sentence in self.examples[lemma]:
                for t in spacy_parser(sentence):  # retrieve global PoS
                    if t.text.lower() == lemma:
                        if t.pos_ not in pos_freqs:
                            print(t.pos_)
                            pos_freqs[t.pos_] = 1
                        else:
                            pos_freqs[t.pos_] += 1
            # global pos frequency
            self.lemma_map[lemma]['pos_freq'] = pos_freqs
            # textrank examples
            temp_examples = self.calculate_similarity(
                self.sentences_to_matrix(self.examples[lemma]), n_sentences=self.limit_examples)
            if lemma in self.disambiguations:
                self.examples[lemma] = [self.examples[lemma][sentence]
                                            for sentence in temp_examples if sentence not in self.lemma_map[lemma] and sentence not in self.disambiguations[lemma]]
            else:
                self.examples[lemma] = [self.examples[lemma][sentence]
                                            for sentence in temp_examples if sentence not in self.lemma_map[lemma]]
            self.lemma_map[lemma]['examples'] = self.examples[lemma]

    def inflect_pos(self, hw, article):
        '''
            we check the most likely POS of the lemma
            and the inflect the form if it is a noun or adjective
            of course this is biased and we need to, at some point, account for different 
            definitions of a lemma, which may affect inflection.
            Since we are using Wikipedia to generate articles, we won't find articles as lemmas
        '''
        try:
            freq_lemma = self.word2index[hw]
        except:
            freq_lemma = 9999999999999999999999999
        if hw not in self.lemma_map:
            self.lemma_map[hw] = {
                "pos_freq": {}, "disambiguations": {}, "frequency_w2v": freq_lemma}
            self.examples[hw] = []

        # store all pos of the lemma inside the article. We take the most frequent one for the inflection.
        pos_occurrences = {}
        # most_common = 'NN' # Noun by default to have something to add.

        singular = hw
        plural = hw
        tag_occurrences = {}

        tag_occurrences = {}
        parsed_article = spacy_parser(article)
        for token in parsed_article:
            if token.text not in self.word_index:
                self.word_count += 1
                self.word_index[token.text] = self.word_count

            if token.text.lower() == hw:
                if token.tag_ not in pos_occurrences:
                    pos_occurrences[token.pos_] = 1
                    tag_occurrences[token.tag_] = 1
                else:
                    pos_occurrences[token.pos_] += 1
                    tag_occurrences[token.tag_] += 1
                lemma = token

        if tag_occurrences and lemma:
            most_frequent_tag = max(
                tag_occurrences, key=lambda k: tag_occurrences[k])
            if most_frequent_tag == 'NNS' or most_frequent_tag == 'NNPS':
                most_frequent_tag = most_frequent_tag[:-1]
                singular = lemma._.inflect(most_frequent_tag)
            else:
                most_frequent_tag = f'{most_frequent_tag}S'
                plural = lemma._.inflect(most_frequent_tag)

        if pos_occurrences:
            most_frequent_pos = max(
                pos_occurrences, key=lambda k: pos_occurrences[k])
        else:
            most_frequent_pos = 'NOUN'  # by defect we assume it's a noun.
        # https://pypi.org/project/inflect/0.2.4/
        return [singular, plural], most_frequent_pos

    def cosine_similarity(self, x, y):
        if len(x) > 0 and len(y) > 0:
            dot = np.dot(x, y)
            # euclidian normalization
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            return dot / (norm_x * norm_y)
        else:
            return 0.

    def calculate_similarity(self, matrix, n_sentences: int = 3):
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
        similarity_matrix = np.zeros(
            (len(matrix), len(matrix)), dtype=np.float32)
        for i in range(len(matrix)):
            array = matrix[i]
            for j in range(len(matrix)):
                _array = matrix[j]
                similarity = cosine_similarity(array, _array)
                similarity_matrix[i][j] = similarity
        sort = {x: sum(similarity_matrix[x])/len(similarity_matrix[x])
                for x in range(len(similarity_matrix))}
        results = {k: v for k, v in sorted(
            sort.items(), key=lambda item: item[1], reverse=True)}
        sorted_results = []
        for k in list(results.keys())[:n_sentences]:
            #sorted_results[k] = similarity_matrix[k]
            sorted_results.append(k)
        # we return sentences sorted by highest score
        return sorted(sorted_results)

    def sentences_to_matrix(self, article: list):
        # find longest sentence for matrix
        split_sentences = []
        size_sentence = 0
        for sentence in article:
            if len(sentence) > 1:
                split = sentence.split(' ')
                for w in split:
                    if w not in self.word_index:
                        self.word_count += 1
                        self.word_index[w] = self.word_count
                split_sentences.append(split)
                sentence_len = len(sentence)

                if sentence_len > size_sentence:
                    size_sentence = sentence_len
            '''
                create words to fit all sentences padding smaller ones
                we pass this to calculate similarity
            '''
        # fit sentences to numpy matrix
        matrix = np.zeros((len(split_sentences), size_sentence), dtype=int)
        for i in range(len(split_sentences)):
            sentence = split_sentences[i]
            for j in range(len(sentence)):
                token = sentence[j]
                matrix[i][j] = self.word_index[token]
        return matrix

    def for_examples(self, full_article: list):
        '''
            return the most common construction. Easy way to retrieve examples
        '''

        for sentence in full_article:
            for t in sentence.split():
                if t not in self.word_index:
                    self.word_count += 1
                    self.word_index[t] = self.word_count
                if t in self.lemma_map:
                    if t in self.examples:
                        self.examples[t].append(sentence)
                    else:
                        self.examples[t] = [sentence]

    def sanitize_text(self, text):
        to_remove = []
        text = text.split('.')
        for sentence in text:
            for i in range(len(sentence.split())):
                try:
                    if i+1 <= len(split) and split[i] == split[i+1]:
                        to_remove.append(i)
                    elif i+2 < len(split) and split[i] == split[i+2]:
                        to_remove.append(i)
                except:
                    continue
            split = [text[i] for i in range(len(text)) if i not in to_remove]
            split[0] = split[0].title()
            return '.'.join(split)

    def generate_def(self, headwords, full_articles):
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
                "examples": [],
            }
        }
        '''
        for i in tqdm(range(len(headwords))):
            hw = headwords[i]
            l_entry = full_articles[i]
            disambiguation = False

            if len(l_entry) > 30:
                l_entry = l_entry.replace('\n', ' ')
                l_entry = l_entry.replace(',', ' ')
                l_entry = l_entry.split('.')

                for s in l_entry:
                    s = ' '.join(re.findall( self.words_only, s))
                l_entry = '.'.join(l_entry)
                self.for_examples(l_entry.split('.'))
                
                if isinstance(hw, str) and not re.search(self.digits_filter, hw) and len(hw) > 2:
                    hw = hw.lower()
                    if hw not in self.lemma_index:
                        self.lemma_index[hw] = {}
                        self.disambiguations[hw] = {}
                    if re.search(self.regex_disambiguation, hw):
                        disambiguation = True
                        hw = hw.split()[0]

                    size_entry = 0
                    cut_entry = []
                    full_article = l_entry
                    inflection, current_pos = self.inflect_pos(hw, l_entry)
                    l_entry = l_entry.split('.')

                    for sentence in l_entry:
                        if size_entry < self.size_short_article and "may refer to" not in sentence:
                            size_entry += len(sentence)
                            cut_entry.append(sentence)

                    for i in range(len(cut_entry)):
                        sentence = re.findall(
                            self.words_only, cut_entry[i])

                        for word in sentence:
                            if word not in self.word_index:
                                self.word_count += 1
                                self.word_index[word] = self.word_count
                        cut_entry[i] = ' '.join(sentence)
                    #cut_entry = '.'.join(cut_entry)
                    new_entry = {"definition": '.'.join([cut_entry[k] for k in self.calculate_similarity(self.sentences_to_matrix(cut_entry))]),
                                    "pos_stats": current_pos
                                    }

                    if len(new_entry["definition"]) > 5:
                        new_entry["definition"] = self.sanitize_text(new_entry["definition"])
                        if disambiguation == False:
                            if "definitions" in self.lemma_map[hw]:
                                self.lemma_map[hw]["definitions"].append(
                                    new_entry)
                            else:
                                self.lemma_map[hw]["definitions"] = [new_entry]
                        else:
                            if "definitions" not in self.lemma_map[hw]["disambiguations"]:
                                self.disambiguations[hw]["definitions"] = new_entry
                            else:
                                self.disambiguations[hw]["definitions"].append(
                                    new_entry)
                        if 'inflection' not in self.lemma_map[hw]:
                            self.lemma_map[hw]['inflection'] = inflection

    @staticmethod
    def reverse_index(
        word_index) -> dict: return word_index.__class__(map(reversed, word_index.items()))
