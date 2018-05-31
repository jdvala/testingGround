#!/usr/bin/env python

# A Simple LSTM for Sequence Classification

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import timeit

program_start = timeit.default_timer()


# fix random seed for reproducibility
numpy.random.seed(7)

# Loading the dataset
dataset = pd.read_csv('legal_dataset.csv', index_col=None)
sentences = list(dataset['Sentence'])
max_sentence_length = 250

Y = pd.get_dummies(dataset['Numerical_Label']).values

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2500)
tokenizer.fit_on_texts(sentences)

X = tokenizer.texts_to_sequences(sentences)
X = sequence.pad_sequences(X, maxlen=max_sentence_length)

# Building the LSTM Network

V = len(tokenizer.word_index) + 1
embed_dim = 100
lstm_out = 100
batch_size = 64

model = Sequential()
model.add(Embedding(V, embed_dim,input_length = X.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(32,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# Splitting the dataset into test and train

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

# Training the model
model_time = timeit.default_timer()
#Here we train the Network.
model.fit(X_train, Y_train, validation_split=.1, batch_size =batch_size, epochs = 20,  verbose = 1)

# Final evaluation of the model
scores = model.evaluate(X_valid, Y_valid, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

stop = timeit.default_timer()
print("Total Time taken by the program: {}".format(stop - program_start))
print("Time taken in model training: {}".format(stop-model_time))
