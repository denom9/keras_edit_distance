import numpy as np


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


