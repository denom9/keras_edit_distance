import nltk
import numpy as np
from keras.models import load_model

import input_processing as w

alphabet = []
for letter in range(97, 123):
    alphabet.append(chr(letter))

number = 10000
bible = nltk.corpus.gutenberg.words(u'bible-kjv.txt')
bibleText = nltk.Text(bible)
words_list = bibleText.tokens

words1 = w.create_wordlist(words_list, number)
words2 = w.create_wordlist(words_list, number)
labels = w.create_labels(words1, words2, number)
matrix1 = np.array(w.create_matrix(words1, alphabet, number))
matrix2 = np.array(w.create_matrix(words2, alphabet, number))

input1 = matrix1.reshape(number, 27, 24, 1)
input2 = matrix2.reshape(number, 27, 24, 1)
labels = np.array(labels)

model = load_model("model.h5")

y_pred = model.predict([input1, input2])
t = np.rint(y_pred)
t = t.astype(int)

matrix_tester = []

rows = 10
columns = 10
m = []
for i in range(rows):
    n = []
    for j in range(columns):
        n.append(0)
    m.append(n)
matrix_tester.append(m)

matrix_tester = np.array(matrix_tester)

for f in range(number):
    matrix_tester[0][t[f][0]][labels[f]] = matrix_tester[0][t[f][0]][labels[f]] + 1
print(matrix_tester)


