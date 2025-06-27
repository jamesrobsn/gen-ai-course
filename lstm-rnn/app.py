import numpy as np
import tensorflow as tf
import streamlit as st
import pickle
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    
    # Convert index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    
    return None

# Streamlit app
st.title("Next Word Prediction with LSTM")

input_text = st.text_input("Enter text:", "to be or not to be")
if st.button("Predict"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write(f"Predicted next word: '{next_word}'")