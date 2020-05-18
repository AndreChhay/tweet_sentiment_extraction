import numpy as np
import pandas as pd

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
