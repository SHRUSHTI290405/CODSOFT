import numpy as np
import json
from tensorflow.keras.models import load_model

# Load
model = load_model("model/handwriting_rnn.h5")

with open("model/char_to_idx.json") as f:
    char_to_idx = json.load(f)
with open("model/idx_to_char.json") as f:
    idx_to_char = json.load(f)

vocab_size = len(char_to_idx)

# Sampling function with temperature
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(preds)), p=preds)

# Generate text function
def generate_text(seed, length=500, temperature=0.6):
    # Keep only known characters
    seed = ''.join([c for c in seed if c in char_to_idx])

    if len(seed) < 10:
        raise ValueError("Seed text too short after cleaning. Use common letters only.")

    generated = seed
    print("Seed:", seed)

    for i in range(length):
        x_pred = np.zeros((1, len(seed), vocab_size))
        for t, char in enumerate(seed):
            x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[str(next_index)]

        generated += next_char
        seed = seed[1:] + next_char

    return generated

seed_text = "the quick brown fox jumps over the lazy dog"
print("\nGenerated text:\n")
print(generate_text(seed_text.lower(), 400, temperature=0.5))
