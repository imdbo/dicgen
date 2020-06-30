'''from gensim.models import KeyedVectors
from gensim.test.utils import datapath


w2v = KeyedVectors.load_word2vec_format('ententen13_tt2_1.vec',  unicode_errors='ignore', binary=False, limit=500000)  

w2v.similar_by_word("rose")
w2v.doesnt_match("rose tree flower plane".split())
w2v.most_similar(positive=['rose', 'flower'], negative=['tree'])
'''
import spacy
import pytextrank

def textrank():
    #tomar sentences do exemplo no tfm e provar se da 2.273381294964029‬

    text = "Dogs are good to people. People like dogs. Cats are evil".lower()
    nlp = spacy.load("en_core_web_sm")
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    doc = nlp(text)
    # examine the top-ranked phrases in the document
    results = []
    for p in doc._.phrases:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        print(p.chunks)
        results.append(p.chunks)
    return  results

print(textrank())