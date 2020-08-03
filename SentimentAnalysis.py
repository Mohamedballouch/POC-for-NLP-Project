# -*- coding: utf-8 -*-
"""
@author: BALLOUCH
"""

""" Sentiment Analysis """


# Importing Libraries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import string

#nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import tokenize
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *

#sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import scikitplot as skplt


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import word2vec

from collections import Counter
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

 
#Identifying and Remove Stop Words
#A stop word is a commonly used word (such as “the”, “a”, “an”, “in”).
#Removal of stopwords is necessary since they add noise without having any informational value in modeling.


stop = stopwords.words('english')



# Tokenize Text in Words 

nltk.download('punkt')

sent_tokenize(data.review)

features['review_tokenized'] = features.review.apply(lambda x: word_tokenize(x))


# NLTK Word Stemming

stemmer = PorterStemmer()

print(stemmer.stem('working'))

features['review_stemmed'] = features.review_tokenized.apply(lambda x: [stemmer.stem(word) for word in x])



"""

Lemmatizing Words Using WordNet
Wordnet:

WordNet is a lexical database for the English language.
It groups English words into sets of synonyms called synsets, provides short definitions and usage examples, and records a number of relations among these synonym sets or their members.
WordNet can thus be seen as a combination of dictionary and thesaurus. While it is accessible to human users via a web browser, its primary use is in automatic text analysis and artificial intelligence applications.
Lemmatization:

Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices.
It makes use of the vocabulary and does a morphological analysis to obtain the root word. Therefore, we usually prefer using lemmatization over stemming.
Example: reduce words such as “am”, “are”, and “is” to a common form such as “be”

"""

# Lemmatizing Words Using WordNet


nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

features['review_lemmatized'] = pfeatures.review_tokenized.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])



###### ****   Modelling using Machine Learning ***** #########


# Split Dataset

x=df_bow_upsampled.iloc[:,0:-1]
y=df_bow_upsampled['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



# RandomForest model

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

acc=accuracy_score(y_pred,y_test)

print('Accuracy Score',acc)

accuracy.append(acc)

y_proba=model.predict_proba(x_test)

f1_scor=f1_score_(y_proba,y_test)


# LSTM model

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
 

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))




