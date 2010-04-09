#!/usr/bin/env python
# encoding: utf-8
"""
teletype.py

Created by NYC Resistor.
"""

import sys, os
import pickle
import time
import random
import serial
import re
import string

import nltk # nltk
from nltk.tokenize import *
from nltk.corpus import movie_reviews

import tweepy # Twitter API class: http://github.com/joshthecoder/tweepy

# things you need to configure
TWITTER_USERNAME = 'ccteletype'
TWITTER_PASSWORD = 'xxx'
SLEEP_INTERVAL = 30 # number of seconds to rest between items
DEBUG = True # debuggy statements to stdout
SNARK_FILE = 'snarkiness.txt' # one snark per line, please
CHANCE_OF_SNARK = .1 # probability between 0 and 1

TELETYPE_TRANS = string.maketrans( '!$-&#\'()"/:?,.', 'fdaghjklzxcbnm') # define allowed characters

teletype = serial.Serial('/dev/cu.usbserial-A9007QM2') # define your serial interface
# init teletype carriage
teletype.write("\r")
teletype.write("+")

# code!
def twit_split( str ): 
	groups = re.split("(.{0,65})(\s|$)",str)
	return [groups[x*3+1] for x in range(len(groups)/3)]

def teletype_encode( str ):
	str = str.lower()
	str = re.sub(r'[:/<>.,]+'," ", str)
	str = re.sub(r'[^a-z\s]',"", str)
	return str
	
class Teletype(object):
    def __init__(self, username, password, cache_file="teletype_cache"):
        api = self.init_twitter(username, password)
        
        try:
            (self.seen, self.users) = pickle.load(open(cache_file, 'r'))
        except IOError:
            self.seen = []
            self.users = {}

        self.cache_file = cache_file
        self.updates = [m for m in api.search('eyebeam') if m.id not in self.seen]
    def next(self):
        try:      
            update = self.updates.pop()
        except IndexError:
            self.save_status() # we're at the end of this run, save our status - a bit hacky
            return None
            
        if update.id not in self.seen:
            self.seen.append(update.id)
            return update
        
        return None

    def save_status(self):
        pickle.dump((self.seen, self.users), open(self.cache_file, 'w'))
		
    def init_twitter(self, username, password):
        auth = tweepy.BasicAuthHandler(username, password)
        api = tweepy.API(auth)
        return api

class Sentimental(object):
    """
    Sentimental is a generic, super quick, very hacky sentiment analysis engine using NLTK
    """
    def __init__(self, sentiment_brain='sentiment_brain'):
        try:
            (self.classifier, self.word_features) = pickle.load(open(sentiment_brain, 'r'))
        except IOError:
            self.classifier = self.build_classifier()
            pickle.dump((self.classifier, self.word_features), open(sentiment_brain, 'w'))
            
        
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
                update = teletype_encode(u.text)
  
            else:
                t.updates = []
                update = None
                if random.random() < CHANCE_OF_SNARK:
                    update = random.choice(snarkiness)

            if update:        
                print update
                for line in twit_split(str(update)):
                	line = teletype_encode( line )
                	print "TELETYPE: ", line 
                	teletype.write(line.lower()) # print to teletype
                	teletype.write("\r")
                	teletype.write("+")
                	time.sleep( len(line)/4 )
                
                teletype.write("+")
                (pos, neg) = s.analyze(update)
                current_sentiment += delta(pos)
                if current_sentiment < 0 or current_sentiment > 255:
                	current_sentiment = 128
                if DEBUG: 
                    print "Current sentiment: %s" % (current_sentiment) 
                graphserial.write( str(int(current_sentiment))+"." ) # print to seismograph     
                print str(int(current_sentiment))+"." 
            time.sleep(SLEEP_INTERVAL)
            
        time.sleep(SLEEP_INTERVAL)
