from __future__ import absolute_import
from __future__ import print_function

import random


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
