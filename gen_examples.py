import numpy as np
import pandas as pd 
from django.conf import settings
import django
from demo.settings import DATABASES, INSTALLED_APPS
settings.configure(DATABASES=DATABASES, INSTALLED_APPS=INSTALLED_APPS)
django.setup()
import math
from dictionary.models import *
import spacy
from multiprocessing import Process
from tqdm import tqdm
spacy.prefer_gpu()  # enable cuda if available.
nlp = spacy.load('en_core_web_lg')
nlp.disable_pipes("ner","parser")
nlp.add_pipe(nlp.create_pipe('sentencizer'))
lemma_map = {}
word_index = {}
cores = 3
word_count = 0
from pymysqlpool.pool import Pool
import pymysql.cursors
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def cosine_similarity(x, y):
    xx,yy,xy=0.0,0.0,0.0
    for i in range(len(x)):
        xx+=x[i]*x[i]
        yy+=y[i]*y[i]
        xy+=x[i]*y[i]
    return 1.0-xy/np.sqrt(xx*yy)

def _cosine_similarity(x, y):
    try:
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    except:
        return 0.
        
def calculate_similarity( matrix, n_sentences: int = 3):
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
        
    for i in tqdm(range(len(matrix))):
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

def sentences_to_matrix(article: list):
    global word_count
    # find longest sentence for matrix
    split_sentences = []
    size_sentence = 0
    for s in range(len(article)):
        sentence = article[s]
        if len(sentence) > 1:
            split = sentence.split(' ')
            for w in split:
                if w not in word_index:
                    word_count += 1
                    word_index[w] = word_count
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
            matrix[i][j] = word_index[token]
    return matrix

def nltk_collocation(lemma, sentences):
    nltkd = nltk.Text(tkn for tkn in sentence.split()
                        for sentence in sentences)
ao

def ______reeeeeeeeeeeee______(entries):
    for i in tqdm(range(len(entries))):
        entry = entries[i].lower()
        for sentence in entry.split('.'):
            for w in sentence.split():
                if w in lemma_map and len(lemma_map[w]["sentences"]) < 5000:
                    print(sentence)
                    lemma_map[w]["sentences"].append(sentence)


if __name__ == '__main__':
    user='root'
    pw='root'
    host="localhost"
    db="demo_dict"
    url = 'https://www.merriam-webster.com/dictionary/'

    pool=Pool(user = user,
            password = pw,
            host = host,
            db = db,
            autocommit = True,
            cursorclass = pymysql.cursors.DictCursor)
    pool.init()
    conn=pool.get_conn()

    with conn.cursor() as cursor:
        cursor.execute("select distinct dl.lemma from dictionary_lemma dl inner join dictionary_lemma_examples dle on dle.lemma_id = dl.id")
        already_examples = cursor.fetchall()
    for lemma in Lemma.objects.all(): 
        lemma_map[lemma.lemma] = {"pos_freq": {},"sentences": [], "collocations": [], "examples": []}

    df = pd.read_csv('smaller_df.csv', low_memory=False)
    headwords = list(df['headword'])
    full_articles = list(df['long_entry'])

    ______reeeeeeeeeeeee______(full_articles)

    user='root'
    pw='root'
    host="localhost"
    db="demo_dict"
    url = 'https://www.merriam-webster.com/dictionary/'

    pool=Pool(user = user,
            password = pw,
            host = host,
            db = db,
            autocommit = True,
            cursorclass = pymysql.cursors.DictCursor)
    pool.init()
    conn=pool.get_conn()

    with conn.cursor() as cursor:
        cursor.execute("select distinct dl.lemma from dictionary_lemma dl inner join dictionary_lemma_examples dle on dle.lemma_id = dl.id")
        already_examples = cursor.fetchall()
        print(already_examples)
        for lemma in lemma_map:
            if lemma not in already_examples: 
                if lemma_map[lemma]['sentences']:
                    for sentence in lemma_map[lemma]["sentences"]:
                        sentence = nlp(sentence)
                        for t in sentence:
                            word = str(t.text)
                            if word in lemma_map:
                                if t.pos_ not in lemma_map[word]['pos_freq']:
                                    lemma_map[word]['pos_freq'][t.pos_] = 1
                                else:
                                    lemma_map[word]['pos_freq'][t.pos_] += 1
                                    
                    temp_examples = calculate_similarity(
                    sentences_to_matrix(lemma_map[lemma]['sentences']), n_sentences=4)
                    temp_examples = [lemma_map[lemma]['sentences'][sentence] for sentence in temp_examples]

                    le = Lemma.objects.get_or_create(lemma = lemma)

                    examples = [Example.objects.get_or_create(example=c) for c in temp_examples]
                    for c in examples:
                        print(c[0].example)
                        le[0].examples.add(c[0].id)

                    tags_to_import = []
                    for k in lemma_map[lemma]['pos_freq']:
                        global_pos_tag = PoS_tag.objects.get_or_create (
                        pos = k)
                        tags_to_import.append(global_pos_tag)
                    
                    pos_freqs = []
                    for k in tags_to_import:
                        print(k)
                        pos_freqs.append(Pos_frequency.objects.get_or_create (
                        pos = k[0],
                        absolute_frequency = lemma_map[lemma]['pos_freq'][k[0].pos]))


                    for tag in pos_freqs:
                        le[0].global_pos_tag.add(tag[0].id)
print('end')