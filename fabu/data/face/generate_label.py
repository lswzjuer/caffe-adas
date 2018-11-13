import pandas as pd
import numpy as np
import os

#os.system('find ./train -iname "*.jpg" > train.txt')
#os.system('find ./test -iname "*.jpg" > test.txt')
train = pd.read_csv("train.txt", header=None)
test = pd.read_csv("test.txt", header=None)
train.columns = ['data']
test.columns = ['data']

def find_label(string):
    label_list = ['normal', 'phone', 'drink', 'smoke']
    for idx, label in enumerate(label_list):
        if label in string:
            return idx

def string_meet(string):
    return ('hmdb' in string['data']) or ('2018' not in string['data']) or ('phone' in string['data'])

#train = train[train.apply(string_meet, axis=1)]
#train = train[not train.apply(lambda x: '2018' not in x['data'] and 'phone' in x['data'], axis=1)]
train['label'] = train['data'].apply(find_label)
test['label'] = test['data'].apply(find_label)

train.to_csv("train.txt", index=False, header=False, sep=' ')
test.to_csv("test.txt", index=False, header=False, sep=' ')
