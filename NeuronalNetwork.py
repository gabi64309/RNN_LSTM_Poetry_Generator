#Gabi's RRN(LSTM)
import random
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

#PREPARING DATA :p

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding = 'utf-8').lower()

text = text[200000:800000]

characters = sorted(set(text))

charToIndex = dict((c, i) for i, c in enumerate(characters)) #'c' and 'i' are coming in tuples. 'c' is the character (key). 'i' is the index (value).
indexToChar = dict((i, c) for i, c in enumerate(characters))

inputSequencyLength = 40
stepSize = 3

sentences = []
nextChar = []

for i in range(0, len(text) - inputSequencyLength, stepSize):
    sentences.append(text[i : i + inputSequencyLength])
    nextChar.append(text[i + inputSequencyLength])

#Converting the text sequences into a numerical representation that a neural network can understand: 'one-hot encoding'.
x = np.zeros((len(sentences), inputSequencyLength, len(characters)), dtype = bool)
y = np.zeros((len(sentences), len(characters)), dtype = bool)

for i, phrase in enumerate(sentences):
    for t, char in enumerate(phrase):
        x[i, t, charToIndex[char]] = 1
    y[i, charToIndex[nextChar[i]]] = 1

#BUILDING THE RECURRENT NEURONAL NETWORK

model = Sequential()
model.add(LSTM(128, input_shape = (inputSequencyLength, len(characters)))) #128 neurons
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.01))
model.fit(x, y, batch_size = 256, epochs =1) #epochs: times that the data is reviewed.

#HELPER FUNCTION (making reasonable text) from keras oficial site

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#GENERATING TEXT
def generate_text(length, temperature):
    startIndex = random.randint(0, len(text) - inputSequencyLength - 1)
    generated = ''
    sentence = text[startIndex: startIndex + inputSequencyLength]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, inputSequencyLength, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, charToIndex[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        nextIndex = sample(predictions,
                                 temperature)
        nextCharacter = indexToChar[nextIndex]

        generated += nextCharacter
        sentence = sentence[1:] + nextCharacter
    return generated

print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))
