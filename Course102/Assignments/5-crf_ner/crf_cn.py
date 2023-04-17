# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:42:30 2022

@author: hongzhihou
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pycrfsuite

# parse the training data and tag
trainpd = pd.read_json('./data/train.json',lines=True,encoding='utf-8')

def label_data(df):
    docs = []
    for _, row in df.iterrows():
            sequence = [[word,'none'] for word in ''.join(row['text'])]

            for key, value in row['label'].items():

                for token, intervals in value.items():

                    for interval in intervals:
                        
                        start = interval[0]
                        while start <=interval[1]:
                            sequence[start][1]=key
                            start +=1
                            
            docs.append(sequence)
            
    return docs


labeled_data = label_data(trainpd[0:])


def get_labels(doc):
    return [label for (token, label) in doc]

def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def word2features(doc, i):
    word = doc[i][0]

    # Common features for all words
    features = [
        'bias',
        'word='+word,
        'word.isalpha=%s' % word.isalpha(),
        'word.isdigit=%s' % word.isdigit(),
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        features.extend([
            '-1:word='+word1,
            '-1:words='+word+word1,
#            '-1:word.isalpha=%s' % word1.isalpha(),
            '-1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')
    
    if i > 2:
        word1=doc[i-1][0]
        word2=doc[i-2][0]
        features.extend([
            '-2:word='+word2,
            '-2:wordss='+word1+word2,
            '-2:wordsss='+word+word2,
            '-2:words='+word+word1+word2,
#            '-2:word.isalpha=%s' % word2.isalpha(),
            '-2:word.isdigit=%s' % word2.isdigit(),            
            
            ])
    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        features.extend([
            '+1:word='+word1,
            '+1:words='+word+word1,
#            '+1:word.isalpha=%s' % word1.isalpha(),
            '+1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')
    if i<len(doc)-2:
        word1=doc[i+1][0]
        word2=doc[i+2][0]
        features.extend([
            '+2:word='+word2,
            '+2:wordss='+word1+word2,
            '+2:words='+word+word1+word2,
             '+2:wordsss='+word+word2,           
#            '+2:word.isalpha=%s' % word2.isalpha(),
            '+2:word.isdigit=%s' % word2.isdigit(),            
            
            ])

    return features

X = [extract_features(doc) for doc in labeled_data]

y = [get_labels(doc) for doc in labeled_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# --- train ---

trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('./data/crf.model')

# --- predict/test ---

tagger = pycrfsuite.Tagger()
tagger.open('./data/crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]


# Create a mapping of labels to indices
labels = {'address': 0, 
          'book': 1,
          'company':2,
          'game':3,
          'government':4,
          'movie':5,
          'name':6,
          'none':7,
          'organization':8,
          'position':9,
          'scene':10}


# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])


# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=['address', 
          'book',
          'company',
          'game',
          'government',
          'movie',
          'name',
          'none',
          'organization',
          'position',
          'scene']))
