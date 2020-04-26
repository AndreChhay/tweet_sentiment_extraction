# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv

#embedding paths
glove_path = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/glove.twitter.27B.25d.txt"

#Train dataset
path_train = "/Users/Ricou/Desktop/ANDRE/machine_learning/tweet_sentiment_extraction/data/train.csv"

data_train = pd.read_csv(path_train)
print(data_train.head())
