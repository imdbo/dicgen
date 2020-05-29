import requests
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

w2v = KeyedVectors.load_word2vec_format(datapath(path_w2v), binary=False)  
word = 'house'
r = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles={word}&rvslots=main")
r.status_code
r.headers['content-type']
r.encoding
r.text
result = r.json()
print(result['query']['pages'])
print(result["contentmodel"])