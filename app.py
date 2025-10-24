import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Modelleri ve TF-IDF'yi yükle
@st.cache_resource
def load_models():
    models = joblib.load('toxic_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return models, tfidf

models, tfidf = load_models()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text

# Özellikleri hesapla
def extract_features(text):
    clean_comment = clean_text(text)
    
    # Numeric features
    count_star = clean_comment.count('*')
    count_question = clean_comment.count('?')
    count_exclamation = clean_comment.count('!')
    word_count = len(clean_comment.split())
    char_count = len(clean_comment)
    avg_word_len = char_count / word_count if word_count > 0 else 0
    
    num_features = np.array([[count_star, count_question, count_exclamation, word_count, char_count, avg_word_len]])
    
    # TF-IDF
    X_tfidf = tfidf.transform([clean_comment])
    
    # Combine
    X = hstack([X_tfidf, num_features])
    
    return X

# Streamlit App
st.title('Toxic Comment Classifier')

st.write('Enter a comment below to classify it into toxic categories.')

comment = st.text_area('Comment:', height=150)

if st.button('Classify'):
    if comment:
        X = extract_features(comment)
        
        predictions = {}
        for col in label_cols:
            model = models[col]
            pred = model.predict_proba(X)[0][1]
            predictions[col] = pred
        
        st.subheader('Prediction Results:')
        for category, prob in predictions.items():
            st.write(f"{category.capitalize()}: {prob:.4f} ({'Toxic' if prob > 0.5 else 'Non-Toxic'})")
        
        # Bar chart for visualization
        df_pred = pd.DataFrame(list(predictions.items()), columns=['Category', 'Probability'])
        st.bar_chart(df_pred.set_index('Category'))
    else:
        st.warning('Please enter a comment.')