# -*- coding: utf-8 -*-
'''
Created: 14 March 2018
Author: Dale Josephs (dsj529)

This project instantiates two different classifiers, Naive Bayes and SGD,
to predict the positive/negative sentiment scores of twitter data.

This project was based on examples provided by LooneyCorn, but has been refactored and
revised to refelct a much more coherent methodology of software development.

Additionally, the code refactoring allowed the opportunity to implement better login
security for the Twitter authentication credentials.
'''

from string import punctuation
import csv
import time

import twitter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import re2 as re

class TweetAnalyzer:
    def __init__(self, verify=False):
        self.conn = self._connect(verify)
        self._raw_data = []
        self.training_data = []
        self.svm_data = []
        self.word_features = []
        self.features = None
        self._stopwords = set(stopwords.words('english') +
                              list(punctuation) +
                              ['AT_USER', 'URL'])

    @staticmethod
    def _connect(verify):
        ''' As a security measure, I have isolated the access to the Twitter
        login credentials here, in a self-contained function with no user-manipulable
        parts.  If I choose to pursue this app further, I'd plan to make the
        credentials file locked to only be readable by the application code, and likely
        add some measure of file encryption.

        This is one of the largest deviations I made from the example code I based this
        project on.  The instructors had the access tokens in plaintext copied from
        the Twitter website and pasted directly in their code.  I could not in good
        conscience duplicate that in my own version.'''

        tokens = []
        with open('twitter_creds') as creds:
            for line in creds:
                tokens.append(line.split('\t')[1][:-1])

        conn = twitter.Api(consumer_key=tokens[0],
                           consumer_secret=tokens[1],
                           access_token_key=tokens[2],
                           access_token_secret=tokens[3])

        if verify:
            print(conn.VerifyCredentials())
        return conn

    def create_test_data(self, search_term):
        try:
            tweets_fetched = self.conn.GetSearch(search_term, count=100)
            print("Success! {} tweets were fetched for the term {}"
                  .format(len(tweets_fetched), search_term))
            return [{'text':status.text, "label":None} for status in tweets_fetched]
        except:
            print('An error occurred.  Please reboot universe and try again.')
            return None

    def buildCorpus(self, sourceFile, destFile, makeDest):
        ''' Takes the corpus outline provided by Niek Sanders and downloads the twitter
        data for each tweet.  If the corpus has already been downloaded, the code only
        returns the data object, rather than re-downloading the tweets.

        parameters
        sourceFile: location of the corpus outline
        destFile: location where the full twitter data should be written.
        makeDest: boolean value, indicates if the tweet data needs to be downloaded before
                  creating the data object for analysis.

        returns
        tweet_data: a list of dictionaries, [{tweet_id, label, topic, date, text}]'''

        if makeDest:
            self._download_corpus(sourceFile, destFile)
        else:
            with open(destFile, 'r') as csvF:
                dr = csv.DictReader(csvF, delimiter=',', quotechar='"')
                for row in dr:
                    if row['Label'] == 'irrelevant':
                        row['Label'] = 'neutral'
                    self._raw_data.append(row)

    def _download_corpus(self, src, dest):
        sleep_time = 900/180
        corpus = []
        with open(src) as csvF:
            reader = csv.reader(csvF, delimiter=',', quotechar='"')
            for row in reader:
                corpus.append({'Tweet_id':row[2], 'Label':row[1], 'Topic':row[0]})

        for tweet in corpus:
            try:
                status = self.conn.GetStatus(tweet['tweet_id'])
#                print('Tweet fetched: {}'.format(status.text))
                tweet['Text'] = status.text
                tweet['Date'] = status.created_at
                if tweet['Label'] == 'irrelevant':
                    tweet['Label'] = 'neutral'
                self._raw_data.append(tweet)
                time.sleep(sleep_time) # avoid Twitter's rate limiting
            except:
                continue

        with open(dest, 'w') as outF:
            fieldnames = ['Topic', 'Label', 'Tweet_id', 'Date', 'Text']
            dw = csv.DictWriter(outF, fieldnames=fieldnames)
            dw.writeheader()
            for tweet in self._raw_data:
                try:
                    dw.writerow(tweet)
                except Exception as e:
                    print('An error occurred: {}'.format(e))

    def pre_process(self):
        def process_tweet(tweet, stopwords):
            # convert to lowercase
            tweet = tweet.lower()
            # replace any links with "URL"
            tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
            # replace "@username" references with "AT_USER"
            tweet = re.sub(r'@[\S]+', 'AT_USER', tweet)
            # replace hashtags:  #word -> word
            tweet = re.sub(r'#([\S]+)', r'\1', tweet)

            tweet = word_tokenize(tweet)
            return [word for word in tweet if word not in stopwords]

        def build_vocabulary(data):
            all_words = []
            for (words, sentiment) in data:
                all_words.extend(words)
            word_list = nltk.FreqDist(all_words)
            return word_list.keys()

        for tweet in self._raw_data:
            self.training_data.append((process_tweet(tweet['Text'], self._stopwords),
                                       tweet['Label']))
            self.word_features = build_vocabulary(self.training_data)
            self.features = \
                 nltk.classify.apply_features(self.extract_features, self.training_data)
            self.svm_data = [' '.join(tweet[0]) for tweet in self.training_data]

    def extract_features(self, tweet):
        tweet_words = set(tweet)
        features = {}
        for word in self.word_features:
            features['contains({})'.format(word)] = (word in tweet_words)
        return features

    def label_data(self):
        _labels = {'positive':1, 'negative':2, 'neutral':3}
        return [_labels[tweet[1]] for tweet in self.training_data]

def main():
    source = '/home/dsj529/Documents/PyProjects/Sanders_Twitter/corpus.csv'
    dest = '/home/dsj529/Documents/PyProjects/Sanders_Twitter/full-corpus.csv'

    analyzer = TweetAnalyzer()
    analyzer.buildCorpus(source, dest, False)
    analyzer.pre_process()

    # term = input('So....what do you want to look up today? ')
    # testData = createTestData(term)

    ## time to start deploying the classifiers
    NB_classifier = nltk.NaiveBayesClassifier.train(analyzer.features)

    count_vec = CountVectorizer(min_df=1)
    X = count_vec.fit_transform(analyzer.svm_data).toarray()
    vocabulary = count_vec.get_feature_names()

    swn_weights = []
    for word in vocabulary:
        try:
            synset = list(swn.senti_synsets(word))
            common_meaning = synset[0]
            if common_meaning.pos_score() > common_meaning.neg_score():
                weight = common_meaning.pos_score()
            elif common_meaning.neg_score() > common_meaning.pos_score():
                weight = -common_meaning.neg_score()
            else:
                weight = 0
        except:
            weight = 0
        swn_weights.append(weight)

    swn_X = []
    for row in X:
        swn_X.append(np.multiply(row, np.array(swn_weights)))
    swn_X = np.vstack(swn_X)

    y = np.array(analyzer.label_data())

    clf = SGDClassifier()
    clf.fit(swn_X, y)

    NB_preds = [NB_classifier.classify(analyzer.extract_features(tweet[0]))
                for tweet in analyzer.training_data]

    SGD_preds = []
    for tweet in analyzer.training_data:
        tweet_sentence = ' '.join(tweet[0])
        sgd_features = np.multiply(count_vec.transform([tweet_sentence]).toarray(),
                                   np.array(swn_weights))
        SGD_preds.append(clf.predict(sgd_features)[0])

    ## find majority vote and display sentiment score
    if NB_preds.count('positive') > NB_preds.count('negative'):
        print('NB Result: Positive ({}%)'
                .format(100*NB_preds.count('positive')/len(NB_preds)))
    else:
        print('NB Result: Negative ({}%)'
                .format(100*NB_preds.count('negative')/len(NB_preds)))

    if SGD_preds.count(1) > SGD_preds.count(2):
        print('SGD Result: Positive({}%)'
                .format(100*SGD_preds.count(1)/len(SGD_preds)))
    else:
        print('SGD Result: Negative({}%)'
                .format(100*SGD_preds.count(2)/len(SGD_preds)))

if __name__ == '__main__':
    main()
