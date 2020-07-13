
from gensim.models import KeyedVectors
number_max_lemmas = 5000
import sys
w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec.1',  unicode_errors='ignore', binary=False, limit=number_max_lemmas)  

def top_n_lemmas(lemma:str, n: int, negative=False):
    """we take the most negative and most positive words from each vector 
        and construct a description around it
    """
    try:
        if negative == False:
            return  list(w2v.most_similar(lemma, topn=sys.maxsize))[:n]
        else:
            all_sims = w2v.most_similar(lemma, topn=sys.maxsize)
            last_n = list(reversed(all_sims[-n:]))
            return last_n
    except:
        return []


x = ['Alien', 'dog']
for lemma in x:
    lemma = lemma.lower()
    xyz = [l for l in top_n_lemmas(lemma, n=5, negative=False)]
    negative_n_lemmas = [l for l in top_n_lemmas(lemma, n=5, negative=True)]
    print(xyz)
    for m in xyz:
        print(m[0])
    print(negative_n_lemmas)