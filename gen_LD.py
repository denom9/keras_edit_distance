import random
import string
import input_processing as w


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


#testo lo script
prec = 0
test = 500
for i in range(test):
    w_len = random.randint(4, 8)
    word = ''
    for _ in range(w_len):
        word += random.choice(string.ascii_lowercase)

    dist = random.randint(1, len(word) + 3)
    new_word = gen_LD(word,dist)
    if w.LD(word,new_word) == dist:
        prec += 1
    else:
        print(word + " - " + str(dist))
        print(new_word)
    print(i)

print("precision: " + str((prec/test)*100) + "%")

'''
print(word + " - " + str(len(word)))
print(new_word)
print("needed dist: " + str(dist))
print("obtained: " + str(w.LD(word,new_word)))
print(w.LD(word,new_word) == dist)
'''