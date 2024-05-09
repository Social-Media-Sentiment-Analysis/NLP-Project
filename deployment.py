import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import re
model = joblib.load(r"F:\second_year\ai\NLP-Project\Sentiment_Analysis.joblib")
vectorizer = joblib.load(r"F:\second_year\ai\NLP-Project\tf_idf_vectorizer.joblib")
st.set_page_config(
    page_title='Sentiment Classifier',
    page_icon=':heart:'
)

def clean_text(sentence):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"(https?://|www\.)\S+",' ',sentence)
    sentence = re.sub('\b[a-zA-Z]\b', ' ', sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()
    stemmer = SnowballStemmer('english')
    tokens = word_tokenize(sentence)
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    tokens = [stemmer.stem(token) for token in tokens if token.lower() not in all_stopwords]
    sentence = ' '.join(tokens)

    return sentence


st.write('# Sentiment Classifier')
st.write('---')
st.subheader('Enter your text and hashtags to analyze sentiment')
# User input
text = st.text_area("Enter your text:", height=100)
text = text 
clean_text(str(text))
if st.button("Analyze Sentiment"):
  text_vector = vectorizer.transform([text])
  y_predict = model.predict(text_vector)
  if(y_predict == 0):
    st.write("You are feeling positive emotions, Live, Love, Laugh!")
  elif(y_predict == 1):
    st.write("You are feeling nothing, Better drink a cup of coffee!")
  else:
    st.write("You are feeling plenty of negative emotions, Eat a snack bar!")

    
        
