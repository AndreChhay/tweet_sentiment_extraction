# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
#from index import get_words_index
#from data import read_train, read_test, read_submission, build_sequences, read_embeddings
from keras.layers import Input, Embedding, Dense
from keras.models import Model, Input
from nltk.tokenize import TweetTokenizer
import pickle

#------------------
#------------------
## data.py
def build_sequences(documents, index):
    tknzr = TweetTokenizer()
    sequences = []
    for doc in documents:
        sequences.append([index[w] for w in tknzr.tokenize(doc.lower())])
    return sequences

def read_train(PathToFile, echantillon=None):
    train=pd.read_csv(PathToFile, nrows= echantillon)
    train['textID']=train['textID'].astype(str)
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    train['sentiment']=train['sentiment'].astype(str)
    return train

def read_test(PathToFile):
    test=pd.read_csv(PathToFile)
    test['text'] = test['text'].astype(str)
    return test

def read_submission(PathToFile):
    submission = pd.read_csv(PathToFile)
    return submission

def read_embeddings (data_path, index, build=True, save=True):
    #Ouput embeddings matrix
    embeddings = np.zeros((len(index)+ 1, 25))
    #Reverse index
    reverse_index = {v:k for k,v in index.items()}
    if build:
        with open(data_path, 'r', encoding="utf-8") as file:
            for line in file:
                tokens = line.split(" ")
                w = tokens[0]
                zw =  tokens[1:]
                try:
                    embeddings[reverse_index[w]] = np.array(float(zj) for zj in zw)
                except KeyError:
                    continue
        if save:
            with open(data_path,'wb') as out:
                pickle.dump(embeddings, out)
    else:
        with open(data_path, 'rb') as out:
            embeddings = pickle.load(out)
    return embeddings

#------------------
#------------------

#Train dataset
train_df = read_train("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/train.csv", echantillon=4)
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
#tknzr = TweetTokenizer()
#tokenisation de chaque tweet
#train_df['text']=train_df['text'].apply(lambda y:tknzr.tokenize(y))
#print(train_df['text'])

#--------------------------------------
#--------------------------------------
## Index.py
words_set = set()

def get_words_index(documents, build=True, save = True, index_path = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/index.pickle"):
    '''
    documents : pd.Series, string
    '''

    #Words
    tknzr = TweetTokenizer()

    def build_index(row):
        global words_set
        #tokenisation de chaque tweet en minuscule
        tokens = tknzr.tokenize(row.lower())
        #intégration de chaque token dans la listewords_set
        words_set = words_set.union(set(tokens))
        return tokens

    if build:
        documents.apply(build_index)
        words_index = {k+1:v for k,v in enumerate(words_set)}
        if save:
            with open(index_path, 'wb') as out:
                pickle.dump(words_index, out)
    else:
        with open(index_path,'rb') as out:
            words_index = pickle.load(out, encoding="utf-8")
    return words_index

#--------------------------------------
#--------------------------------------
#words index
#train_df['text'] = train_df['text'].apply(build_index)
words_index = get_words_index(train_df['text'], build=False, save=False)
reverse_words_index = { v:k for k,v in words_index.items()}
print(words_index)
print("-----")
print(reverse_words_index)

print(words_index.items())

glove_path = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/glove.twitter.27B.25d.txt"
#Embeddings
z = read_embeddings("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/glove.twitter.27B.25d.txt", index=words_index, build=False, save=False)
print(z)

#sequences
sequences = build_sequences(train_df['text'], reverse_words_index)
print(np.max([len(s) for s in sequences]))
print(np.mean([len(s) for s in sequences]))

# Padding
T = 8

x = pad_sequences(sequences, maxlen = T, padding="post")

#Keras models
input = Input(shape=[T])
z = Embedding(input_dim=len(words_index)+1, output_dim=25, weights=[embeddings_matrix])(input)
y= Dense(1)
model = Model(inputs=input, outputs=[y1,y2])

print(model.summary())
