import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


#model.wv.index2entity[:100]

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
    def __init__(self, build: bool = True, 
                encoding: str = 'utf-8', 
                max_lemmas: int = 100, 
                size_sentence: int = 15, 
                top_n_sentences_lemma: int = 10, 
                w2v_path:str = 'w2v/ententen13_tt2_1.vec', #w2v keyedvectors path
                stats_df: str = 'wiki_stats.csv',
                data_dir: str = 'texts/'):
        #--common--#
        self.encoding = encoding
        self.widx = {'<#PAD#>':0} #word index for retrieval 
        self.word_count = 1
        #--data--#
        self.data_dir = data_dir
        self.stats_df = stats_df
        #--sentences--#
        self.sentences = {} # index of all sentences
        self.top_sentences_lemas = np.empty([max_lemmas, top_n_sentences_lemma],dtype=int)
        #--summarization--#
        self.eidx = {} # full size entries indexed
        self.seidx = {}  #short entries indexed

        if build == True:
            self.build()

        def build(self):
            #extract parameters from stats dataframe and create arrays: word frequency, lemmas, number of articles per lemma and max size article.
            full_entries = 0
            short_entries = 0
            sentence_count = 0

            max_size_f_entry = 0 
            max_size_s_entry = 0
            self.word_count += 1
            stats = pd.read(self.stats_df)
            
            for hw, size_l_entry, s_l_entry, size_s_entry, s_s_en try in zip(stats['headword'], stats['size_long_entry'], stats['sentences_long_entry'], stats['size_short_entry']):
                if self.word_count < self.max_lemmas:
                    self.widx[hw] = self.word_count

                    if max_size_f_entry < size_l_entry:
                        max_size_f_entry = size_l_entry

                    if max_size_s_entry < size_s_entry:
                        max_size_s_entry = size_s_entry 

            self.eidx = np.zeros(full_entries, max_size_f_entry)
            self.seidx = np.zeros()
            #import texts
            for root, dirs, files in os.walk(self.dirpath):
                for name in files:
                    path = (root + "/" + name)
                    df = pd.read_csv(path)
                    for hw, l_entry, s_entry in zip(df['headword'], df['long_entry'], df['short_entry']):
                        if hw in self.widx:
                            
                        else:
                            continue

    def index_word(self, word):
        if word not in self.widx:
            self.word_count += 1
            self.widx[word] = self.word_count
            self.entries[word] = []
        return
    
    def store_entry(self, index_lemma):
        self.entries
        return
    
    def entries_to_sentences(self):
        return

    def short_entries(self):
        return


