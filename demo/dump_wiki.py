# extraction part based on https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c
import requests# Library for parsing HTML
from bs4 import BeautifulSoup
import mwparserfromhell
import pandas as pd
import subprocess
import re
import os
import sys
import json
if len(sys.argv) > 1:
    folder = sys.argv[0]
    short_article_len = sys.argv[1]
else:
    folder = 'text/'
    short_article_len = 512
#uncomment to download
'''
base_url = 'https://dumps.wikimedia.org/enwiki/'
index = requests.get(base_url).text
soup_index = BeautifulSoup(index, 'html.parser')# Find the links on the page
dumps = [a['href'] for a in soup_index.find_all('a') if 
         a.has_attr('href')]

dump_date = dumps[-2].replace('/', '')
dump_url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles-multistream.xml.bz2"

os.system(f'/bin/bash -c "wget {dump_url}"')
os.system(f"python3 wikiextractor/WikiExtractor.py -cb 250K -o extracted {dump_url}")
'''

to_df = {"headword": [], "long_entry": []} #dict to parse into dataframe with pandas
stats_df = {"longest full article": 0, "number of articles": 0}

file_count = 0 
for root, dirs, files in os.walk(folder):
    for name in files:
            path=(root + "/" + name)
            file_count += 1
            with open(path, 'r', encoding= 'utf-8') as articles:
                    for line in articles:
                        try:
                            article = json.loads(line)
                            article_len = len(article['text'])
                            print(f'{file_count}--{article["title"]}', end='\r')
                            if article_len > stats_df['longest full article']:
                                stats_df['longest full article'] = article_len

                            to_df['headword'].append(article['title'])
                            to_df['long_entry'].append(article['text'])
                        except Exception as e:
                            print(f'{e}--\n{line}')

            if file_count >= 100:
                df_articles = pd.DataFrame.from_dict(to_df)
                df_articles.to_csv('data.csv', mode='a+')
                to_df = {"headword": [], "long_entry": []} #dict to parse into dataframe with pandas
                file_count = 0 
                