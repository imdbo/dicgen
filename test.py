'''from nltk.corpus import wordnet as wn

synonyms=[]
for word in wn.words():
    print (word,end=":")
    for syn in wn.synsets(word):
      for l in syn.lemmas():
        synonyms.append(l.name())
    print(set(synonyms),end="\n")
    synonyms.clear()

#https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/

#https://www.programcreek.com/python/example/91604/nltk.corpus.wordnet.synsets

#https://spacy.io/universe/project/pyInflectw
'''

import spacy
import lemminflect

nlp = spacy.load('en_core_web_sm')
doc = nlp('I am testing this example.')
doc[2]._.lemma()  
for w in doc:       # 'test'
    print(w._.inflect('NNS'))  # 'examples'
