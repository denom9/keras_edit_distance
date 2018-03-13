from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np

def LD(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    res = min([LD(s[:-1], t) + 1,
               LD(s, t[:-1]) + 1,
               LD(s[:-1], t[:-1]) + cost])
    return res


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
        labels.append(LD(wordlist1[i], wordlist2[i]))
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


def create_mInput(word, alphabet):
    rows = 27
    columns = 24
    m = []
    for i in range(rows):
        n = []
        for j in range(columns):
            if i == 0:
                if len(word) > j:
                    n.append(0)
                else:
                    n.append(1)
            else:
                if len(word) > j:
                    if (word)[j] == alphabet[i - 1]:
                        n.append(1)
                    else:
                        n.append(0)
                else:
                    n.append(0)
        m.append(n)
    return m

def gen_LD(word, dist):
    word_l = list(word)

    if dist > len(word):
        new_word = ''
        for _ in range(dist):
            new_letter = random.choice(string.ascii_lowercase)
            while new_letter in word_l:
                new_letter = random.choice(string.ascii_lowercase)
            new_word += new_letter
        return new_word

    used_indexes = []
    changes = [0] * len(word)
    new_word = list(word)

    for _ in range(dist):
        choice = random.randint(0, 1)

        new_pos = random.randint(0, len(new_word) - 1)
        if choice == 0: #sostituzione
            while changes[new_pos] == 1 or new_pos in used_indexes:
                new_pos = random.randint(0, len(new_word) - 1)
            changes[new_pos] = 1
            if new_pos not in used_indexes:
                used_indexes.append(new_pos)
            old_letter = new_word[new_pos]
            new_letter = random.choice(string.ascii_lowercase)
            while new_letter == old_letter or new_letter in word_l:
                new_letter = random.choice(string.ascii_lowercase)
            new_word[new_pos] = new_letter

        if choice == 1: #inserimento
            changes.insert(new_pos, 1)
            word_l.insert(new_pos,0)
            if new_pos not in used_indexes:
                used_indexes.append(new_pos)
            new_letter = random.choice(string.ascii_lowercase)
            while new_letter in word_l:
                new_letter = random.choice(string.ascii_lowercase)
            new_word.insert(new_pos, new_letter)

        if choice == 2: #cancellazione
            while changes[new_pos] == 1 or new_pos in used_indexes:
                new_pos = random.randint(0, len(new_word) - 1)
            del new_word[new_pos]
            del changes[new_pos]

        #print(choice)
        #print(changes)
        #print(used_indexes)

    new_word = ''.join(new_word)
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