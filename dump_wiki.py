# extraction part based on https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c
import requests# Library for parsing HTML
from bs4 import BeautifulSoup
import mwparserfromhell
import pandas as pd
import subprocess
import xml.sax
import re
import os


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._pages.append((self._values['title'], self._values['text']))


base_url = 'https://dumps.wikimedia.org/enwiki/'
index = requests.get(base_url).text
soup_index = BeautifulSoup(index, 'html.parser')# Find the links on the page
dumps = [a['href'] for a in soup_index.find_all('a') if 
         a.has_attr('href')]

dump_date = dumps[-2].replace('/', '')
dump_url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/enwiki-{dump_date}-pages-articles-multistream.xml.bz2"

#uncomment to download
os.system(f'/bin/bash -c "wget {dump_url}"')
os.system(f"python3 wikiextractor/WikiExtractor.py -cb 250K -o extracted {dump_url}")

# Object for handling xml
handler = WikiXmlHandler()# Parsing object
parser = xml.sax.make_parser()
parser.setContentHandler(handler)

to_df = {"headword": [], "long_entry": [], "short_entry": []} #dict to parse into dataframe with pandas
regex_section = re.compile(r'==(=)?.*==(=)?') #regex to split article into sections

data_path = 'enwiki-20200520-pages-articles-multistream.xml.bz2' # Iterate through compressed file one line at a time
with open('test.txt', 'w+', encoding='utf-8') as t:
    for line in subprocess.Popen(['bzcat'], 
                                stdin = open(data_path), 
                                stdout = subprocess.PIPE).stdout:
        parser.feed(line)
        #t.write(str(line.decode("utf-8") ) + "\n")
        # Stop when 3 articles have been found
        if len(handler._pages) > 2:
            break
        
    for p in handler._pages:
        to_df['headword'].append(p[0])
        cleanup = mwparserfromhell.parse(p[1]).strip_code().strip()
        cleanup = re.split(regex_section, cleanup)
        to_df['long_entry'].append(cleanup)
        to_df['short_entry'].append('faket')
        
    df = pd.DataFrame.from_dict(to_df)
    df.to_csv("stats_dump.csv")
    for el in handler._pages[1]:
        print(f"{el} \n------------------------------------------------------------------")