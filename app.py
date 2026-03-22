import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    text = re.sub(r'@\w+|http\S+|#\w+|[^a-zA-Z\s]', '', str(text))
    text = text.lower().strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(lemmatizer.lemmatize(w) for w in words)


st.title("Sentiment Analyzer")
st.write("Enter a sentence to predict its sentiment (positive or negative).")

tweet = st.text_area("Tweet text", "")

if st.button("Analyze"):
    if not tweet:
        st.warning("Please enter a tweet to analyze.")
    else:
        clean_text = preprocess_text(tweet)
        vectorized = vectorizer.transform([clean_text])
        
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0].max()
        
        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"{label}  (Confidence: {confidence:.2%})")
        
        probs = model.predict_proba(vectorized)[0]
        classes = model.classes_
        probs_df = pd.DataFrame({
            'Sentiment': classes, 
            'Probability': probs
        })
        st.bar_chart(probs_df.set_index('Sentiment'))

        if confidence < 0.6:
            st.info("Hmm—I’m a bit unsure. This seems ambiguous.")
