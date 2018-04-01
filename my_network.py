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
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utilities as u
import random

epochs = 20
words_number = 50016
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
words_list2 = words.words()

alphabet = []
for letter in range(97, 123):
    alphabet.append(chr(letter))

words_to_train1 = u.create_wordlist(words_list, words_number_train)
words_to_train2 = u.create_wordlist(words_list, words_number_train)
words_to_val1 = u.create_wordlist(words_list2, words_number_val)
words_to_val2 = u.create_wordlist(words_list2, words_number_val)

'Qui è possibile cambiare il modo in cui è scelta la seconda parola della coppia per il training set'
for i in range(words_number_train):
    words_to_train2[i] = u.gen_LD(words_to_train1[i], random.randint(1, 8))

'Qui è possibile cambiare il modo in cui è scelta la seconda parola della coppia per il validation set'
for i in range(words_number_val):
    words_to_val2[i] = u.gen_LD(words_to_val1[i], random.randint(1, 8))

labels_train = w.create_labels(words_to_train1, words_to_train2, words_number)
labels_val = w.create_labels(words_to_val1, words_to_val2, words_number)

matrix_train1 = np.array(u.create_matrix(words_to_train1, alphabet, words_number))
matrix_train2 = np.array(u.create_matrix(words_to_train2, alphabet, words_number))
matrix_val1 = np.array(u.create_matrix(words_to_val1, alphabet, words_number))
matrix_val2 = np.array(u.create_matrix(words_to_val2, alphabet, words_number))

#earlystop & checkpoint
earlystop = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.00001, patience=2,
                          verbose=1, mode='auto')
checkpoint_callback = ModelCheckpoint('model_name' + '.h5', monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='min')
callbacks_list = [earlystop, checkpoint_callback]

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
matrix_val1 = matrix_val1.reshape(words_number, letters, length_limit, 1)
matrix_val2 = matrix_val2.reshape(words_number, letters, length_limit, 1)
labels_train = np.array(labels_train)
labels_val = np.array(labels_val)

'per caricare il modello precedentemente allenato'
# model = load_model("model_name.h5")

# train
model.compile(loss=losses.mean_squared_error, optimizer='sgd', metrics=[binary_accuracy])

model.fit([matrix_train1, matrix_train2], labels_train,
          batch_size=32,
          epochs=epochs,
          callbacks=callbacks_list,
          validation_data=([matrix_val1, matrix_val2], labels_val))

new_model = load_model("model_name.h5")

score = new_model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
print('Test loss evaluation:', score[0])
print('Test accuracy evaluation:', score[1])

'in fase di test questo metodo consente di mostrare accuratezza sui 3 corpora'
# u.check_accuracy('model_name.h5', alphabet)
'in fase di test questo metodo consente di accuratezza dei vari edit distance(1-8) sui 3 corpora'
# u.check_diff_ed('model_name.h5', alphabet)
