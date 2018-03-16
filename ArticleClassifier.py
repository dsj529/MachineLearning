from math import log
from heapq import nlargest
from string import punctuation
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class FrequencySummarizer:
    def __init__(self, floor=0.1, ceil=0.9):
        self._floor = floor
        self._ceil = ceil
        self._stopwords = set(stopwords.words('english') +
                              list(punctuation) +
                              [u"'s", '"'])

    def _computeFrequencies(self, sents, customStopwords=None):
        ''' calculates term frequencies, and converts them into percentages
        based on the highest-occurring term in the text.  Strips out terms with
        occurrence percentages outside set floor and ceiling thresholds.'''

        freq = defaultdict(int)
        if customStopwords is None:
            stopwords = self._stopwords
        else:
            stopwords = set(customStopwords).union(self._stopwords)
        for sent in sents:
            for word in sent:
                if word not in stopwords:
                    freq[word] += 1

        m = float(max(freq.values()))

        for w in list(freq.keys()):
            freq[w] = freq[w]/m
            if freq[w] >= self._ceil or freq[w] <= self._floor:
                del freq[w]
        return freq

    def extractFeatures(self, article, n, customStopwords=None):
        ''' Returns the top n terms of the article's text
         assume articles are passed in as a tuple (text, title) '''

        text, title = article[0], article[1]
        artSents = sent_tokenize(text)
        artWords = [word_tokenize(s.lower()) for s in artSents]
        self._freq = self._computeFrequencies(artWords, customStopwords)

        if n < 0:
            # if request a negative number of features, return all words
            return nlargest(len(self._freq.keys()), self._freq, key=self._freq.get)
        else:
            return nlargest(n, self._freq, key=self._freq.get)

    def extractFrequencies(self, article):
        ''' Returns the raw term frequencies for the words in the article's text'''

        text, title = article[0], article[1]
        artSents = sent_tokenize(text)
        artWords = [word_tokenize(s.lower()) for s in artSents]
        freq = defaultdict(int)
        for sent in artWords:
            for word in sent:
                if word not in self._stopwords:
                    freq[word] += 1
        return freq

    def summarize(self, article, n):
        text, title = article[0], article[1]
        artSents = sent_tokenize(text)
        artWords = [word_tokenize(s.lower()) for s in artSents]
        self._freq = self._computeFrequencies(artWords)
        ranking = defaultdict(int)
        for i, sent in enumerate(artWords):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        top_sents = nlargest(n, ranking, key=ranking.get)

        return [artSents[j] for j in top_sents]

###

# collect the articles for classification

class articleSet:
    def __init__(self, url, dateline, source, token, label):
        if source == 'WaPo':
            scraper = self._getWaPoText
        else:
            scraper = self._getNYTText

        self.data = self.scrapeArticles(url, scraper, dateline, token)
        self.label = label

    @staticmethod
    def _getWaPoText(url, token):
        ''' Text scraper for Washington Post articles.

            parameters:
            url: the url of the article to be scraped.
            token: a token that indicates the article's text.  Defaults to "article".

            returns the article text and title'''

        try:
            page = requests.get(url)
        except:
            # if article can't be retreived, return Title=None, Article=None.
            return(None, None)

        soup = BeautifulSoup(page.content, 'lxml')
        if soup is None:
            return(None, None)
        text = ""
        if soup.find_all(token) is not None:
            # search the page code for a tag that indicates article text
            # such as <article> </article>
            text = ''.join(map(lambda p: p.text, soup.find_all(token)))
            # Join all the text contained between <article> tags
            soup2 = BeautifulSoup(text, 'lxml')
            # Now, parse the article's text to filter out non-text
            if soup2.find_all('p') != []:
                text = ''.join(map(lambda p: p.text, soup2.find_all('p')))
                # assuming there are <p>aragraph tags within the <article>,
                # string together all the paragraph texts.
        return text, soup.title.text

    @staticmethod
    def _getNYTText(url, token):
        ''' Scraper function for NY Times articles.

            parameters:
            url: the url of the article
            token: a token indicating article body text.

            returns the article text and title'''

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'lxml')
        title = soup.find('title').text
        mydivs = soup.findAll('p', {"class":"story-body-text story-content"})
        text = ''.join(map(lambda p: p.text, mydivs))
        return text, title

    @staticmethod
    def scrapeArticles(url, scraper, magic='2018', token='None'):
        articles = {}
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        numErrors = 0
        count = 0
        for a in soup.findAll('a'):
            count += 1
            try:
                artURL = a['href']
                if ((artURL not in articles) and
                    (magic and ((magic in artURL) or str(int(magic)-1) in artURL))):
                    # use the Magic token to check for a byline date in the URL
                    body = scraper(artURL, token)
                    if body and len(body) > 0:
                        articles[artURL] = body
#                    print(artURL)
            except:
                numErrors += 1
                # track parsing errors
        print('{} links inspected, {} faulty links found'.format(count, numErrors))
        return articles

waPoTechURL = "https://www.washingtonpost.com/business/technology"
waPoNonTechURL = "https://www.washingtonpost.com/sports"
NYT_TechURL = "http://www.nytimes.com/pages/technology/index.html"
NYT_NonTechURL = "https://www.nytimes.com/pages/sports/index.html"

waPo_tech = articleSet(waPoTechURL, '2018', 'WaPo', 'article', 'tech')
waPo_nontech = articleSet(waPoNonTechURL, '2018', 'WaPo', 'article', 'non-tech')
NYT_tech = articleSet(NYT_TechURL, '2018', 'NYT', None, 'tech')
NYT_nontech = articleSet(NYT_NonTechURL, '2018', 'NYT', None, 'non-tech')

###

# Turn the article collections into feature vectors

def processArticleDict(summarizer, summary_d, art_d, label):
    for artURL, artData in art_d.items():
        if artData[0]:
            summary = summarizer.extractFeatures(artData, 25)
            summary_d[artURL] = {'feature-vector': summary, 'label': label}
    return summary_d


article_summaries = {}
fs = FrequencySummarizer()
for artSet in [waPo_tech, waPo_nontech, NYT_tech, NYT_nontech]:
    processArticleDict(fs, article_summaries, artSet.data, artSet.label)

###

# Now, for some test data

def getTestText(testURL, token):
    response = requests.get(testURL)
    soup = BeautifulSoup(response.content, 'lxml')
    title = soup.find('title').text
    mydivs = soup.findAll('div', {'class':token})
    text = ''.join(map(lambda p: p.text, mydivs))
    return text, title

testURL = 'http://doxydonkey.blogspot.in'
testTxt = getTestText(testURL, "post-body")

fs = FrequencySummarizer()
testSumm = fs.extractFeatures(testTxt, 25)

###

# first text similarity test

similarities = {}
for artURL, artData in article_summaries.items():
    artSumm = artData['feature-vector']
    similarities[artURL] = len(set(testSumm).intersection(set(artSumm)))

labels = defaultdict(int)
knn = nlargest(5, similarities, key=similarities.get)
for neighbor in knn:
    labels[article_summaries[neighbor]['label']] += 1

nlargest(1, labels, key=labels.get)

###

cumulativeTF = {'Tech': defaultdict(int), 'Non-Tech': defaultdict(int)}
trainingData = {'Tech': NYT_tech, 'Non-Tech': NYT_nontech}
for label in trainingData:
    for  artURL, artData in trainingData[label].data.items():
        if artData[0]:
            fs = FrequencySummarizer()
            docTF = fs.extractFrequencies(artData)
            for term in docTF:
                cumulativeTF[label][term] += docTF[term]

###

# next, some Naive-Bayes evaluation

scores = [1.0, 1.0] # [Non-Tech, Tech]
for word in testSumm:
    if word in cumulativeTF['Tech']:
        scores[1] *= (1e3 * cumulativeTF['Tech'][word] /
                      float(sum(cumulativeTF['Tech'].values())))
        # multiply the assumed techiness of each word by its probability
        # of occurring in a tech article, based on the training data
    else:
        scores[1] /= 1e3
        # if the word is not found in the training set,
        # assume a 1/1000 probability of the term occurring in the wild.
        #
        # Then, repeat for non-tech scores.
    if word in cumulativeTF['Non-Tech']:
        scores[0] *= (1e3 * cumulativeTF['Non-Tech'][word] /
                      float(sum(cumulativeTF['Non-Tech'].values())))
    else:
        scores[0] /= 1e3

scores[0] *= (float(sum(cumulativeTF['Non-Tech'].values())) /
              (float(sum(cumulativeTF['Tech'].values())) +
               float(sum(cumulativeTF['Non-Tech'].values()))))

scores[1] *= (float(sum(cumulativeTF['Tech'].values())) /
              (float(sum(cumulativeTF['Tech'].values())) +
               float(sum(cumulativeTF['Non-Tech'].values()))))
# scale each probability score by the overall probability of Tech/Non-Tech words

if scores[1] > scores[0]:
    label = 'Tech'
else:
    label = 'Non-Tech'
print(label, scores)

####
# Next, TFIDF and KMeans clustering

def getBlogPosts(url, links):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'lxml')
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == 'Older Posts':
                links.append(url)
                getBlogPosts(url, links)
        except:
            title = ''
    return

blogURL = 'http://doxydonkey.blogspot.in'
links = []
getBlogPosts(blogURL, links)

testData = {}
for link in links:
    testData[link] = getTestText(link, 'post-body')

corpus = []
for post in testData.values():
    corpus.append(post[0])

vector = TfidfVectorizer(max_df=0.75, min_df=2, stop_words='english')
X = vector.fit_transform(corpus)
clf = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=True)
clf.fit(X)

keywords={}
for i, cluster in enumerate(clf.labels_):
    doc = corpus[i]
    fs = FrequencySummarizer()
    summary = fs.extractFeatures((doc, ''), 100, None)
#                                 [u"according", u"also", u"billion", u"like",
#                                  u"new", u"one", u"year", u"first", u"last"])
    if cluster not in keywords:
        keywords[cluster] = set(summary)
    else:
        keywords[cluster] = keywords[cluster].intersection(set(summary))
