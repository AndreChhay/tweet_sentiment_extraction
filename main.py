# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
from index import get_words_index
from data import read_train, read_test, read_submission, build_sequences, read_embeddings
from keras.layers import Input, Embedding, Dense
from keras.models import Model, Input

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
