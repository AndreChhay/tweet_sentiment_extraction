from nltk.tokenize import TweetTokenizer
import pickle

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
