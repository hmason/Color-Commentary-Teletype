#!/usr/bin/env python
# encoding: utf-8
"""
teletype.py

Created by Hilary Mason on 2010-03-09.
Copyright (c) 2010 Hilary Mason. All rights reserved.
"""

import sys, os
import pickle
import time
import random

import nltk # nltk
from nltk.tokenize import *
from nltk.corpus import movie_reviews

import tweepy # Twitter API class: http://github.com/joshthecoder/tweepy

# things you need to configure
TWITTER_USERNAME = 'ccteletype'
TWITTER_PASSWORD = 's33s33teletype'
SLEEP_INTERVAL = 1 # number of seconds to rest between items
DEBUG = True # debuggy statements to stdout
SNARK_FILE = 'snarkiness.txt' # one snark per line, please
CHANCE_OF_SNARK = .05 # probability between 0 and 1

# code!
class Teletype(object):
    def __init__(self, username, password, cache_file="teletype_cache"):
        api = self.init_twitter(username, password)
        
        try:
            (self.seen, self.users) = pickle.load(open(cache_file, 'r'))
        except IOError:
            self.seen = []
            self.users = {}
        self.cache_file = cache_file

        self.updates = [m for m in api.mentions() if m.id not in self.seen]

    def next(self):
        try:      
            update = self.updates.pop()
        except IndexError:
            self.save_status() # we're at the end of this run, save our status - a bit hacky
            return None
            
        if update.id not in self.seen:
            if DEBUG:
                print "Update: %s, %s, %s" % (update.id, update.text, update.user.screen_name)
            self.seen.append(update.id)
            self.users[update.user.screen_name] = self.users.get(update.user.screen_name, 0) + 1
            return update
        
        return None

    def save_status(self):
        pickle.dump((self.seen, self.users), open(self.cache_file, 'w'))
		
    def init_twitter(self, username, password):
        auth = tweepy.BasicAuthHandler(username, password)
        api = tweepy.API(auth)
        return api

class Sentimental(object):
    def __init__(self, sentiment_brain='sentiment_brain'):
        try:
            (self.classifier, self.word_features) = pickle.load(open(sentiment_brain, 'r'))
        except IOError:
            self.classifier = self.build_classifier()
            pickle.dump((self.classifier, self.word_features), open(sentiment_brain, 'w'))
            
        # self.classifier.show_most_informative_features()
        
    def build_classifier(self):
        documents = [(' '.join(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
        random.shuffle(documents)
        
        all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words() if len(w) > 2)
        self.word_features = all_words.keys()[:2000]
        
        featuresets = [(self.document_features(d), c) for (d,c) in documents]
        classifier = nltk.NaiveBayesClassifier.train(featuresets)
        return classifier

    def document_features(self, document):
        document_words = word_tokenize(document)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words) # feature format follows canonical example
        return features
        
    def analyze(self, text):
        page_prob = self.classifier.prob_classify(self.document_features(text))
        pos = page_prob.prob('pos')
        neg = page_prob.prob('neg')
        return (pos, neg)

def delta(pos):
    current = (10000000 * pos) - 3
    return 10*random.random()*current

if __name__ == '__main__':
    s = Sentimental()
    snarkiness = [snark.strip() for snark in open(SNARK_FILE, 'r').readlines()]
    current_sentiment = 128
    
    while(True):
        if DEBUG:
            print "iterating"
        t = Teletype(TWITTER_USERNAME, TWITTER_PASSWORD, 'teletype_cache')
        
        for i in range(0,len(t.updates)+1):
            update = None
            u = t.next()
            if u:
                update = u.text
            else:
                t.updates = []
                update = None
                if random.random() < CHANCE_OF_SNARK:
                    update = random.choice(snarkiness)

            if update:        
                print update # TODO: print to teletype
                (pos, neg) = s.analyze(update)
                current_sentiment += delta(pos)
                if DEBUG: 
                    print "Current sentiment: %s" % (current_sentiment) # TODO: print to seismograph
                            
            time.sleep(SLEEP_INTERVAL)
            
        time.sleep(SLEEP_INTERVAL)