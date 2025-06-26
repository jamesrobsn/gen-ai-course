import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

def preprocess_text(text, max_len=500):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=max_len)
    return padded

## Prediction Function
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

## Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative):")

user_input = st.text_area("Movie Review")

if st.button("Predict"):
    sentiment, score = predict_sentiment(user_input)
    st.write(f"Review: {user_input}\nSentiment: {sentiment}, Score: {score:.4f}")
else:
    st.write("Please enter a review and click 'Predict' to see the sentiment analysis.")