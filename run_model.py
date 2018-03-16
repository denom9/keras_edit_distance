import numpy as np
from keras.models import load_model

import utilities as w

alphabet = []
for letter in range(97, 123):
    alphabet.append(chr(letter))

input1 = w.create_mInput("caso", alphabet)
input2 = w.create_mInput("catoll", alphabet)

input1 = np.array(input1)
input2 = np.array(input2)

input1 = input1.reshape(1, 27, 24, 1)
input2 = input2.reshape(1, 27, 24, 1)

model = load_model("model.h5")

result = model.predict([input1, input2])

print(result)
