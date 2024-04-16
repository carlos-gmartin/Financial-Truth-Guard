import os
import joblib
import numpy as np
from django.shortcuts import render
from .utils import preprocess_text_for_nb, map_to_traffic_light, get_model_filenames, preprocess_text_for_cnn
from .api.news import get_news
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v2")
MODELS_DIR_v3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v3")

# Function to load the Naive Bayes model
def load_naive_bayes():
    # Load the Naive Bayes model
    nb_model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
    nb_model = joblib.load(nb_model_path)
    return nb_model

# Function to load the TF-IDF vectorizer
def load_tfidf_vectorizer():
    tfidf_vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    return tfidf_vectorizer

# Function to load the CNN model
def load_cnn_model():
    # Load the CNN model
    cnn_model_path = os.path.join(MODELS_DIR_v3, 'cnn_model.h5')
    cnn_model = load_model(cnn_model_path)
    return cnn_model

# Function to load the CNN model TF-IDF tokenizer
def load_cnn_tokenizer():
    tokenizer_path = os.path.join(MODELS_DIR_v3, 'tokenizer.pkl')
    tokenizer = joblib.load(tokenizer_path)
    return tokenizer

def get_predictions(text, model_filename):
    # Naive Bayes
    if 'naive_bayes' in model_filename.lower():
        # Load Naive Bayes model and TF-IDF vectorizer
        nb_model = load_naive_bayes()
        tfidf_vectorizer = load_tfidf_vectorizer()
        
        # Preprocess text for Naive Bayes
        preprocessed_text = preprocess_text_for_nb(text)
        
        # Transform input text using the TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([preprocessed_text])

        try:
            # Make predictions using Naive Bayes
            predictions = nb_model.predict_proba(transformed_text)[0]
            print("Naive Bayes Predictions:", predictions)  # Add this line for debugging
            return predictions
        except Exception as e:
            return f"Error during prediction: {e}"
    
    # CNN model
    else:
        # Load CNN model and tokenizer
        cnn_model = load_cnn_model()
        tokenizer = load_cnn_tokenizer()
        
        # Preprocess text for CNN
        preprocessed_text = preprocess_text_for_cnn(text)
        
        # Tokenize and pad input text
        X_test_sequences = tokenizer.texts_to_sequences([preprocessed_text])
        X_test_pad = pad_sequences(X_test_sequences, padding='post', maxlen=24512)
        
        try:
            # Make predictions using CNN
            predictions = cnn_model.predict(X_test_pad)[0]
            print("CNN Predictions:", predictions)  # Add this line for debugging
            return predictions
        except Exception as e:
            return f"Error during prediction: {e}"

def extract_fake_news_words(text, predictions, threshold=0.5):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Zip tokens with corresponding predictions
    token_predictions = zip(tokens, predictions)
    
    # Filter tokens based on predictions
    fake_news_words = [token for token, pred in token_predictions if pred >= threshold]
    
    return fake_news_words

# Result page view
def result(request):
    line = request.GET.get('line', '')
    model_filename = request.GET.get('model', '')
    
    if not line or not model_filename:
        return render(request, 'error.html', {'error_message': 'Missing line or model filename'})

    result = get_predictions(line, model_filename)
    
    return render(request, 'result.html', {'result': result})

# Home page
def home(request):
    data = get_news('NVDA', '25', '03', '2024')
    articles = data.get('results', [])

    for article in articles:
        # Get predictions for the article
        predictions = get_predictions(article['description'], 'cnn_model.h5')
        
        # Calculate traffic light based on the predictions
        article['traffic_light'] = map_to_traffic_light(predictions)
        
        # Get indicative words for fake news
        indicative_words = extract_fake_news_words(article['description'], predictions)
        article['indicative_words'] = indicative_words
        
    return render(request, 'index.html', {'articles': articles})

# testing page view
def test_web(request):
    # Get all model filenames
    model_files = get_model_filenames()
    
    # Pass the model filenames to the template
    return render(request, 'testing.html', {'models': model_files})

def api(request):
    return render(request, 'api.html')
