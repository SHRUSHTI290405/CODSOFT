import numpy as np

# Load text
with open("data/handwritten_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Corpus length:", len(text))

# Unique characters
chars = sorted(list(set(text)))
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

vocab_size = len(chars)
print("Vocab size:", vocab_size)

# Create sequences
seq_length = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print("Number of sequences:", len(sentences))

# Vectorize
X = np.zeros((len(sentences), seq_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

print("Vectorization complete!")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(256, input_shape=(seq_length, vocab_size), return_sequences=True),
    Dropout(0.2),
    LSTM(256),
    Dense(vocab_size, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

# Train
model.fit(X, y, batch_size=128, epochs=30)

# Save model + mappings
model.save("model/handwriting_rnn.h5")

import json
with open("model/char_to_idx.json","w") as f:
    json.dump(char_to_idx, f)
with open("model/idx_to_char.json","w") as f:
    json.dump(idx_to_char, f)

print("Model trained and saved!")
