import streamlit as st
import json
import numpy as np
from tensorflow.keras.models import load_model

# Load model and mappings
model = load_model("model/handwriting_rnn.h5")

with open("model/char_to_idx.json") as f:
    char_to_idx = json.load(f)
with open("model/idx_to_char.json") as f:
    idx_to_char = json.load(f)

vocab_size = len(char_to_idx)

def sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(preds)), p=preds)

def generate_text(seed, length=300, temp=0.5):
    seed = ''.join([c for c in seed if c in char_to_idx])
    generated = seed

    for _ in range(length):
        x_pred = np.zeros((1, len(seed), vocab_size))
        for t, char in enumerate(seed):
            x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_char = idx_to_char[str(sample(preds, temp))]
        generated += next_char
        seed = seed[1:] + next_char

    return generated

st.title("✍️ Handwritten Style Text Generator")

seed = st.text_input("Seed Text (simple letters only)", "dear friend i hope you are well ")
temp = st.slider("Creativity (Temperature)", 0.2, 1.0, 0.5)

if st.button("Generate"):
    text = generate_text(seed.lower(), 400, temp)
    st.text_area("Generated Text", text, height=300)
