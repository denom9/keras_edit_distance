from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np
import string
import editdistance as ed
from keras.models import load_model
import nltk


def create_wordlist(words, number):
    wordlist = []
    for i in range(number):
        randomWord = random.choice(words)
        while not randomWord.isalpha() or len(randomWord) > 8:
            randomWord = random.choice(words)
        randomWord = randomWord.lower()
        wordlist.append(randomWord)
    return wordlist


def create_labels(wordlist1, wordlist2, number):
    labels = []
    for i in range(number):
        labels.append(ed.eval(wordlist1[i], wordlist2[i]))
    return labels


def create_matrix(wordlist, alphabet, number):
    matrix1 = []
    for p in range(number):
        rows = 27
        columns = 24
        m = []
        for i in range(rows):
            n = []
            for j in range(columns):
                if i == 0:
                    if len(wordlist[p]) > j:
                        n.append(0)
                    else:
                        n.append(1)
                else:
                    if len(wordlist[p]) > j:
                        if (wordlist[p])[j] == alphabet[i - 1]:
                            n.append(1)
                        else:
                            n.append(0)
                    else:
                        n.append(0)
            m.append(n)
        matrix1.append(m)
    return matrix1


def gen_LD(word, dist):
    word_l = list(word)
    new_word = word_l
    while ed.eval(word, new_word) != dist:
        if len(new_word) == 8:
            op = random.choice([0, 2])
        else:
            if len(new_word) == 1:
                op = random.choice([0, 1])
            else:
                op = random.choice([0, 1, 2])

        new_pos = random.randint(0, len(new_word) - 1)

        if op == 0:  # sostituzione
            new_letter = random.choice(string.ascii_lowercase)
            new_word[new_pos] = new_letter

        if op == 1:  # inserimento
            new_letter = random.choice(string.ascii_lowercase)
            new_word.insert(new_pos, new_letter)

        if op == 2:  # cancellazione
            del new_word[new_pos]

    return new_word


def tester_matrix(input1, input2, labels, model, words):
    y_pred = model.predict([input1, input2])
    t = np.rint(y_pred)
    t = t.astype(int)
    tester_matrix = []
    rows = 10
    columns = 10
    m = []
    for i in range(rows):
        n = []
        for j in range(columns):
            n.append(0)
        m.append(n)
    tester_matrix.append(m)
    tester_matrix = np.array(tester_matrix)
    for f in range(words):
        tester_matrix[0][t[f][0]][labels[f]] = tester_matrix[0][t[f][0]][labels[f]] + 1
    return tester_matrix


def check_accuracy(model_name, alphabet):
    number = 10000
    reuters = nltk.corpus.reuters.words()
    reuters_text = nltk.Text(reuters)
    reuters_words = reuters_text.tokens

    gutenberg = nltk.corpus.gutenberg.words()
    gutenberg_text = nltk.Text(gutenberg)
    gutenberg_words = gutenberg_text.tokens

    brown = nltk.corpus.brown.words()
    brown_text = nltk.Text(brown)
    brown_words = brown_text.tokens

    words_to_val1 = create_wordlist(reuters_words, number)
    words_to_val2 = create_wordlist(reuters_words, number)
    matrix_val1 = np.array(create_matrix(words_to_val1, alphabet, number))
    matrix_val2 = np.array(create_matrix(words_to_val2, alphabet, number))
    matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
    matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
    labels_val = create_labels(words_to_val1, words_to_val2, number)
    model = load_model(model_name)
    score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
    print('Testing reuters corpus')
    print('Test loss evaluation:', score[0])
    print('Test accuracy evaluation:', score[1])
    print(tester_matrix(matrix_val1, matrix_val2, labels_val, model, number))

    words_to_val1 = create_wordlist(gutenberg_words, number)
    words_to_val2 = create_wordlist(gutenberg_words, number)
    matrix_val1 = np.array(create_matrix(words_to_val1, alphabet, number))
    matrix_val2 = np.array(create_matrix(words_to_val2, alphabet, number))
    matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
    matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
    labels_val = create_labels(words_to_val1, words_to_val2, number)
    model = load_model(model_name)
    score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
    print('Testing gutenberg corpus')
    print('Test loss evaluation:', score[0])
    print('Test accuracy evaluation:', score[1])
    print(tester_matrix(matrix_val1, matrix_val2, labels_val, model, number))

    words_to_val1 = create_wordlist(brown_words, number)
    words_to_val2 = create_wordlist(brown_words, number)
    matrix_val1 = np.array(create_matrix(words_to_val1, alphabet, number))
    matrix_val2 = np.array(create_matrix(words_to_val2, alphabet, number))
    matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
    matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
    labels_val = create_labels(words_to_val1, words_to_val2, number)
    model = load_model(model_name)
    score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
    print('Testing brown corpus')
    print('Test loss evaluation:', score[0])
    print('Test accuracy evaluation:', score[1])
    print(tester_matrix(matrix_val1, matrix_val2, labels_val, model, number))

    return


def check_diff_ed(model_name, alphabet):
    number = 10000
    reuters = nltk.corpus.reuters.words()
    reuters_text = nltk.Text(reuters)
    reuters_words = reuters_text.tokens

    gutenberg = nltk.corpus.gutenberg.words()
    gutenberg_text = nltk.Text(gutenberg)
    gutenberg_words = gutenberg_text.tokens

    brown = nltk.corpus.brown.words()
    brown_text = nltk.Text(brown)
    brown_words = brown_text.tokens

    wordlist = []
    wordlist2 = []
    print('Testing reuters corpus')
    for j in range(8):
        wordlist.clear()
        wordlist2.clear()
        for i in range(number):
            randomWord = random.choice(reuters_words)
            randomWord2 = random.choice(reuters_words)
            randomWord = randomWord.lower()
            randomWord2 = randomWord2.lower()
            while not randomWord.isalpha() or not randomWord2.isalpha() or len(randomWord) > 8 or len(
                    randomWord2) > 8 or ed.eval(randomWord, randomWord2) != (j + 1):
                randomWord = random.choice(reuters_words)
                randomWord2 = random.choice(reuters_words)
                randomWord = randomWord.lower()
                randomWord2 = randomWord2.lower()
            wordlist.append(randomWord)
            wordlist2.append(randomWord2)
        matrix_val1 = np.array(create_matrix(wordlist, alphabet, number))
        matrix_val2 = np.array(create_matrix(wordlist2, alphabet, number))
        matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
        matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
        labels_val = create_labels(wordlist, wordlist2, number)
        model = load_model(model_name)
        score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
        print('Accuracy on edit distance ', j + 1, ':', score[1])

    wordlist = []
    wordlist2 = []
    print('Testing gutenberg corpus')
    for j in range(8):
        wordlist.clear()
        wordlist2.clear()
        for i in range(number):
            randomWord = random.choice(gutenberg_words)
            randomWord2 = random.choice(gutenberg_words)
            randomWord = randomWord.lower()
            randomWord2 = randomWord2.lower()
            while not randomWord.isalpha() or not randomWord2.isalpha() or len(randomWord) > 8 or len(
                    randomWord2) > 8 or ed.eval(randomWord, randomWord2) != (j + 1):
                randomWord = random.choice(gutenberg_words)
                randomWord2 = random.choice(gutenberg_words)
                randomWord = randomWord.lower()
                randomWord2 = randomWord2.lower()
            wordlist.append(randomWord)
            wordlist2.append(randomWord2)
        matrix_val1 = np.array(create_matrix(wordlist, alphabet, number))
        matrix_val2 = np.array(create_matrix(wordlist2, alphabet, number))
        matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
        matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
        labels_val = create_labels(wordlist, wordlist2, number)
        model = load_model(model_name)
        score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
        print('Accuracy on edit distance ', j + 1, ':', score[1])

    wordlist = []
    wordlist2 = []
    print('Testing brown corpus')
    for j in range(8):
        wordlist.clear()
        wordlist2.clear()
        for i in range(number):
            randomWord = random.choice(brown_words)
            randomWord2 = random.choice(brown_words)
            randomWord = randomWord.lower()
            randomWord2 = randomWord2.lower()
            while not randomWord.isalpha() or not randomWord2.isalpha() or len(randomWord) > 8 or len(
                    randomWord2) > 8 or ed.eval(randomWord, randomWord2) != (j + 1):
                randomWord = random.choice(brown_words)
                randomWord2 = random.choice(brown_words)
                randomWord = randomWord.lower()
                randomWord2 = randomWord2.lower()
            wordlist.append(randomWord)
            wordlist2.append(randomWord2)
        matrix_val1 = np.array(create_matrix(wordlist, alphabet, number))
        matrix_val2 = np.array(create_matrix(wordlist2, alphabet, number))
        matrix_val1 = matrix_val1.reshape(number, 27, 24, 1)
        matrix_val2 = matrix_val2.reshape(number, 27, 24, 1)
        labels_val = create_labels(wordlist, wordlist2, number)
        model = load_model(model_name)
        score = model.evaluate([matrix_val1, matrix_val2], labels_val, verbose=0)
        print('Accuracy on edit distance ', j + 1, ':', score[1])

        return


def gen_random_wordlist(listlen):
    wordlist = []
    for i in range(listlen):
        length = random.randint(3, 8) #lunghezza parola
        word = ''
        for _ in range(length):
            word += random.choice(string.ascii_lowercase) #genero parola a lettere casuali
        wordlist.append(word)
    return wordlist


def gen_LD_wordlist(base_list,listlen):
    scales = listlen / 8
    ld_list = []
    dist = 0
    for i in range(listlen):
        if i > scales * (dist+1): dist += 1
        #dist = random.randint(0,8)
        ld_list.append(gen_LD(base_list[i], dist))
    return ld_list
