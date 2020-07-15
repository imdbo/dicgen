from rouge_score import rouge_scorer
from bleu import list_bleu
from requests import get
from bs4 import BeautifulSoup
from pymysqlpool.pool import Pool
import pymysql.cursors
from requests import get
import re
import nltk
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
smoothing = SmoothingFunction()
'''
    We calculate the F1 score of 100 random articles in the database 
    against  the same lemmas in merriam webster's online dictionary
'''


def f1_score(hypothesis:str, ref:str):
    #rougue
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge = rouge.score(hypothesis,ref)
    rouge = rouge['rouge1'].precision
    #bleu
    bleu = nltk.translate.bleu_score.sentence_bleu([ref],hypothesis, smoothing_function=smoothing.method4)
    try:
        #f1 h. avg.
        return  rouge, bleu, 2 * (bleu * rouge) / (bleu + rouge)
    except:
        return 0.0, 0.0, 0.0

def scrape_merriam(lemma:str, article:str, div_name:str):
    response = get(article)
    scraper = BeautifulSoup(response.text, 'html.parser')
    definitions = scraper.find_all('div',{'id': re.compile(div_name)})
    return '.'.join([d.text for d in definitions])


if __name__ == "__main__":
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
        results = {'total': {
            'rouge':[],
            'bleu':[],
            'f1': []  
        }}
        cursor.execute("SELECT DL.lemma, DD.definition FROM dictionary_lemma DL INNER JOIN dictionary_lemma_definition DLD ON DL.id = DLD.lemma_id INNER JOIN dictionary_definition DD ON DD.id = DLD.definition_id WHERE DD.definition NOT LIKE '%may refer to%' AND DD.definition NOT LIKE '%may also refer to%' AND DD.definition NOT LIKE '%refers to' AND char_length(DD.definition) > 5 ORDER BY RAND()")
        entries = cursor.fetchall()
        for entry in tqdm(range(len(entries))):
            entry = entries[entry]
            lemma = entry['lemma'].lower()
            definition = entry['definition']
            related_lemmas = cursor.execute(f"SELECT DCT.token  \
            FROM dictionary_lemma DL \
            INNER JOIN   dictionary_lemma_positive_lemma DLPL ON DLPL.lemma_id = DL.id \
            INNER JOIN dictionary_context_token DCT ON DCT.id = DLPL.context_token_id \
            WHERE DL.lemma = '{lemma}'")
            definition = definition + '.'.join(token['token'] for token in cursor.fetchall())
            examples = f"SELECT DC.collocation \
            FROM dictionary_lemma DL \
            INNER JOIN   dictionary_lemma_collocations DLC ON DLC.lemma_id = DL.id \
            INNER JOIN dictionary_collocation DC ON DC.id = DLC.collocation_id \
            WHERE DL.lemma = '{lemma}'"
            definition = definition + '.'.join(collocation['collocation'] for collocation in cursor.fetchall())
            #print(definition)
            article = url + lemma
            merriam_text = scrape_merriam(lemma, article,"-entry-")
            rouge, bleu, f1 = f1_score(definition.lower(), merriam_text.lower())
            #print(rouge, bleu, f1)
            if len(results) < 100:
                if rouge > 0.01 :#filter broken links/articles
                    print(len(results))
                    results[lemma] = {}
                    results[lemma]['rouge'] = round(rouge, 2)
                    results[lemma]['bleu'] = round(bleu, 2)
                    results[lemma]['f1'] = round(f1, 2)

                    results['total']['rouge'].append(results[lemma]['rouge'])
                    results['total']['bleu'].append(results[lemma]['bleu'])
                    results['total']['f1'].append(results[lemma]['f1'])
                    print(results[lemma])
            else:
                results['total']['rouge'] = sum([v for v in results['total']['rouge']]) / len(results['total']['rouge'])
                results['total']['bleu'] = sum([v for v in results['total']['bleu']]) / len(results['total']['bleu'])
                results['total']['f1'] = sum([v for v in results['total']['f1']]) / len(results['total']['f1'])
                break
        conn.close()
pd.DataFrame.from_dict(results, orient='index').to_csv('quantitative_analysis_3.csv')
