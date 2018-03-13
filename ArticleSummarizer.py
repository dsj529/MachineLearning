from collections import defaultdict
from heapq import nlargest
from string import punctuation

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import requests

class FrequencySummarizer:
    """This class and supplementary functions will get the text of a Washington Post
    article, find the wordfrequencies within (excluding stopwords), then determines the
    top N sentences to summarize the article's content based on word frequencies)"""

    def __init__(self, floor=0.1, ceil=0.9):
        self.floor = floor
        self.ceil = ceil
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    def _compute_frequencies(self, sents):
        freq = defaultdict(int)
        for sent in sents:
            for word in sent:
                if word not in self._stopwords:
                    freq[word] += 1

        m = float(max(freq.values()))

        for w in list(freq.keys()):
            freq[w] = freq[w]/m
            if freq[w] >= self.ceil or freq[w] <= self.floor:
                del freq[w]
        return freq

    def summarize(self, text, n):
        sents = sent_tokenize(text)
        assert n <= len(sents)
        words_sents = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(words_sents)
        ranking = defaultdict(int)
        for i, sent in enumerate(words_sents):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        top_sents = nlargest(n, ranking, key=ranking.get)
        return [sents[j] for j in top_sents]

####

def parse_wapo_article(url):
    src = requests.get(url)
    full_txt = src.content

    soup = BeautifulSoup(full_txt)
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))

    soup2 = BeautifulSoup(text)
    text = ' '.join(map(lambda p: p.text, soup2.find_all('p')))

    return soup.title.text, text

###

test_url = ('https://www.washingtonpost.com/news/the-switch/wp/2015/08/06/'
            'why-kids-are-meeting-more-strangers-online-than-ever-before/')

url_text = parse_wapo_article(test_url)

fs = FrequencySummarizer()

summary = fs.summarize(url_text[1], 3)
