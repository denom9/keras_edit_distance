from __future__ import absolute_import
from __future__ import print_function

import nltk
import numpy as np
from keras import backend as K
from keras import losses
from keras.layers import Input, Dense, Lambda, Conv2D, ZeroPadding2D, Flatten
from keras.models import Model
from keras.models import load_model
from nltk.corpus import words
from sklearn.metrics import mean_squared_error as mse

import input_processing as w

epochs = 10
words_number = 10000
letters = 27
length_limit = 24


def euclidean_distance(vects):
    x, y = vects
    return K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(0, 1), data_format='channels_last')(input)
    x = Conv2D(64, (27, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    return Model(input, x)


def binary_accuracy(output_true, output_pred):
    return K.mean(K.equal(output_true, K.round(output_pred)), axis=-1)


words_list = words.words()
book = nltk.corpus.gutenberg.words(u'austen-persuasion.txt')
book_text = nltk.Text(book)
words_list2 = book_text.tokens

alphabet = []
for letter in range(97, 123):
    alphabet.append(chr(letter))

words_to_train1 = w.create_wordlist(words_list, words_number)
words_to_train2 = w.create_wordlist(words_list, words_number)
words_to_val1 = w.create_wordlist(words_list2, words_number)
words_to_val2 = w.create_wordlist(words_list2, words_number)
labels_train = w.create_labels(words_to_train1, words_to_train2, words_number)
labels_val = w.create_labels(words_to_val1, words_to_val2, words_number)

matrix_train1 = np.array(w.create_matrix(words_to_train1, alphabet, words_number))
matrix_train2 = np.array(w.create_matrix(words_to_train2, alphabet, words_number))
matrix_val1 = np.array(w.create_matrix(words_to_val1, alphabet, words_number))
matrix_val2 = np.array(w.create_matrix(words_to_val2, alphabet, words_number))

input_shape = (letters, length_limit, 1)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

matrix_train1 = matrix_train1.reshape(words_number, letters, length_limit, 1)
matrix_train2 = matrix_train2.reshape(words_number, letters, length_limit, 1)
matrix_val1 = matrix_train1.reshape(words_number, letters, length_limit, 1)
matrix_val2 = matrix_train2.reshape(words_number, letters, length_limit, 1)
labels_train = np.array(labels_train)
labels_val = np.array(labels_val)

# resuming model
#model = load_model("model.h5")
# train
model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=[binary_accuracy])

model.fit([matrix_train1, matrix_train2], labels_train,
          batch_size=32,
          epochs=epochs,
          validation_data=([matrix_val1, matrix_val2], labels_val))

output_pred = model.predict([matrix_train1, matrix_train2])
int_pred = np.rint(output_pred)
int_pred = int_pred.astype(int)
tr_acc = mse(labels_train, int_pred)
output_pred = model.predict([matrix_val1, matrix_val2])
int_pred = np.rint(output_pred)
int_pred = int_pred.astype(int)
te_acc = mse(labels_val, int_pred)

score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
print('Test loss evaluation:', score[0])
print('Test accuracy evaluation:', score[1])

# saving model
model.save("model.h5")

print('Errore quadratico medio sul training set: %0.2f%%' % tr_acc)
print('Errore quadratico medio sul validation set: %0.2f%%' % te_acc)
