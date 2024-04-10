import os
import joblib
import numpy as np
from django.shortcuts import render
from .utils import preprocess_text_for_nb, map_to_traffic_light, get_model_filenames
from .api.news import get_news
import re
from collections import Counter

# models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v2")

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

# Function to get top features associated with fake news
def get_top_features():
    # Load the Naive Bayes model
    nb_model = load_naive_bayes()
    
    # Load the TF-IDF vectorizer
    tfidf_vectorizer = load_tfidf_vectorizer()
    
    # Get feature probabilities for the "fake" class
    feature_probabilities = nb_model.feature_log_prob_[1]  # Assuming "fake" is the positive class
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Sort features based on probabilities
    sorted_indices = np.argsort(feature_probabilities)[::-1]  # Sort in descending order
    top_n = 1000  # Select the top 1000 features.
    
    top_features = []
    top_feature_probabilities = []
    
    for i in sorted_indices[:top_n]:
        top_features.append(feature_names[i])
        top_feature_probabilities.append(np.exp(feature_probabilities[i]))  # Convert log probabilities to actual probabilities
    
    return top_features, top_feature_probabilities

def get_predictions(text, model_filename):

    # Need to make ability for users to use different models.
    # Load the model
    model = load_naive_bayes()

    # Preprocess text based on the selected model
    if 'naive_bayes' in model_filename.lower():
        preprocessed_text = preprocess_text_for_nb(text)
    else:
        return "Model not supported"

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = load_tfidf_vectorizer()
    # Transform input text using the TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([preprocessed_text])

    try:
        # Make predictions
        predictions = model.predict_proba(transformed_text)[0]

        return predictions
    except Exception as e:
        return f"Error during prediction: {e}"

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

    # Get top features associated with fake news
    top_features, _ = get_top_features()

    for article in articles:
        # Get predictions for the article
        predictions = get_predictions(article['description'], 'naive_bayes_model.pkl')

        # Calculate traffic light based on the predictions
        article['traffic_light'] = map_to_traffic_light(predictions)

        # Initialize list to store fake news words found in the article description
        article['fake_news_words'] = []

        # Check if any top feature is present in the article description
        for word in article['description'].split():
            if word.lower() in top_features:
                article['fake_news_words'].append(word)

    return render(request, 'index.html', {'articles': articles})


# testing page view
def test_web(request):
    # Get all model filenames
    model_files = get_model_filenames()
    
    # Pass the model filenames to the template
    return render(request, 'testing.html', {'models': model_files})

def api(request):
    return render(request, 'api.html')
