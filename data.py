import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
import pickle

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

#from index import get_words_index
#words_index = get_words_index(None, build = False, save = False)
#embedding paths
#glove_path = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/glove.twitter.27B.25d.txt"
#read_embeddings(glove_path, index = words_index)
