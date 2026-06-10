import numpy as np

vocab = ['I', 'love', 'trekking','hiking', 'learning']

word = 'trekking'

vector = np.zeros(len(vocab))

vector[vocab.index(word)] = 1

print(vector)