from Interfaces.PreProcessing import PreProcessingInterface
import numpy as np
import pandas as pd
import re
import nltk


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class PreProcessing(PreProcessingInterface):
    def __init__(self, data):
        self.data = data

    def emptyValue(self):
        self.data = self.data.dropna(subset=['tweet'])

    def lowerCase(self):
        self.data["tweet"] = self.data.tweet.str.lower()

    def numericValue(self):
        self.data['tweet'] = self.data['tweet'].apply(lambda x: re.sub(r'\b\d+\b|\d+', '', x))

    def hastag(self):
        self.data['tweet'] = self.data['tweet'].apply(lambda x: re.sub(r'\b(?:@|http|www|#)\S+\b', '', x))

    def punctuation(self):
        self.data['tweet'] = self.data['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    def dublicatedData(self):
        self.data.drop_duplicates(subset=['tweet', 'type'], inplace=True)


    def stemmer(self):
        stemmer = PorterStemmer()

        for index, row in self.data.iterrows():
            tweet = row['tweet']
            tokens = word_tokenize(tweet)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]

            stemmed_tweet = ' '.join(stemmed_tokens)

            self.data.at[index, 'tweet'] = stemmed_tweet


    def stopWord(self):
        stop_words = set(stopwords.words('english'))
        self.data['tweet'] = self.data['tweet'].apply(
            lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


    def oneOrTwo(self):
        self.data['tweet'] = self.data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))

    def allStep(self):
        self.emptyValue()
        self.dublicatedData()
        self.lowerCase()
        self.numericValue()
        self.hastag()
        self.punctuation()
        self.stemmer()
        self.dublicatedData()
        self.stopWord()
        self.dublicatedData()
        self.oneOrTwo()
        self.dublicatedData()
        return self.data