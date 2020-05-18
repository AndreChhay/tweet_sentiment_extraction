# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Librairies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re #regex

from data import read_train, read_test, read_submission

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Train dataset

train_df = read_train("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/train.csv")
print(train_df.head())
print(train_df.shape)

#Test dataset
test_df = read_test("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/test.csv")
#print(test_df.head())
print(test_df.shape)


print("Text preprocessing")
#Recherche de valeurs manquantes sur le dataset train
print(f'Training null Values:\n{train_df.isnull().sum()}\n')
print(f'Test null Values:\n{test_df.isnull().sum()}')
#si c'est le cas, on drop les observations
train_df.dropna(axis=0, inplace=True)

#Data analyse
class_index = {k:v for k,v in enumerate(train_df['sentiment'].unique())}
print(class_index)

train_df.replace({"sentiment":{"positive":2, "neutral":1, "negative":0}}, inplace=True)
test_df.replace({"sentiment":{"positive":2, "neutral":1, "negative":0}}, inplace=True)

#repartition de la variable 'sentiment'
fig, axs = plt.subplots(1,2)
sns.countplot(train_df.sentiment,ax=axs[0])
sns.countplot(test_df.sentiment,ax=axs[1])
axs[0].set_title("Train Dataset")
axs[1].set_title("Test Dataset")
#plt.show()

def text_preprocessing(text):
    # rend le texte en minuscules
    text = text.lower()
    # supprime le texte entre crochets
    text = re.sub('\[.*?\]', '', text)
    # supprime le texte entre parenthèses
    text = re.sub('<.*?>+', '', text)
    # supprime les caractères de saut de ligne
    text = re.sub('\n', ' ', text)

    stop_words = set(stopwords.words('english'))
    return text

print("Data preprocessing")
train_df['clean_text'] = train_df['text'].apply(lambda x: text_preprocessing(x))
test_df['clean_text'] = test_df['text'].apply(lambda x: text_preprocessing(x))
print(train_df['clean_text'].head())

X_train = train_df.clean_text
y_train = train_df.sentiment
X_test = test_df.clean_text
y_test=test_df.sentiment
tokenize = Tokenizer()
tokenize.fit_on_texts(X_train.values)

X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)

max_length= max([len(s.split()) for s in train_df['clean_text']])
X_train = pad_sequences(X_train, max_length)
X_test = pad_sequences(X_test, max_length)

max_features = len(tokenize.word_index)+1
num_classes = 3


model=Sequential()
model.add(Embedding(max_features,100,input_length=max_length))
model.add(LSTM(64,dropout=0.2,return_sequences=True))
model.add(LSTM(32,dropout=0.3,return_sequences=False))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,batch_size=500, epochs=10,  verbose=1)

y_pred_train = model.predict_classes(X_train)
y_pred_test = model.predict_classes(X_test)

#print(y_pred_train)


print("test Result:\n")
print("Accuracy Score:\n")
print(accuracy_score(y_train, y_pred_train))
print("Confusion Matrix:\n")
print(confusion_matrix(y_train, y_pred_train))
print("======================================")
print("test Result:\n")
print("Accuracy Score:\n")
print(accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_test))

#Read submission
#submission_df = read_submission("/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/sample_submission.csv")
#print(submission_df.head())
