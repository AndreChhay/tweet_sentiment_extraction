# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
from nltk.tokenize import TweetTokenizer
import pickle

#embedding paths
glove_path = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/glove.twitter.27B.25d.txt"

def read_train(PathToFile):
    train=pd.read_csv(PathToFile)
    train['textID']=train['textID'].astype(str)
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    train['sentiment']=train['sentiment'].astype(str)
    return train

def read_test(PathToFile):
    test=pd.read_csv(PathToFile)
    test['text']=test['text'].astype(str)
    return test

def read_submission(PathToFile):
    test=pd.read_csv(PathToFile)
    return test

#Train dataset
train_df = read_train("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/train.csv")
print(train_df.head())

#Train dataset
test_df = read_test("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/test.csv")
print(test_df.head())

#Read submission
submission_df = read_submission("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/sample_submission.csv")
print(submission_df.head())

# label index : 3 possibilités neutral(0), negative(1) et positive(2)
class_index = {k:v for k,v in enumerate(train_df['sentiment'].unique())}
print(class_index)

#Un tweet est une série de tokens individuels
tknzr = TweetTokenizer()
#tokenisation de chaque tweet
train_df['text']=train_df['text'].apply(lambda y:tknzr.tokenize(y))
print(train_df['text'])


words_set = set()

def build_index(row):
    global words_set
    tokens = tknzr.tokenize(row.lower())
    words_set = words_set.union(set(tokens))
    return tokens

train_df['text'] = train_df['text'].apply(build_index)

words_index = {k+1:v for k,v in enumerate(words_set)}
print(words_index)

with open("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/index.pickle", 'wb') as out:
    pickle.dump(words_index, out)
