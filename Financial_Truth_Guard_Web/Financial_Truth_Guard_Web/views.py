import os
import joblib
import numpy as np
from django.shortcuts import render
from .utils import preprocess_text_for_nb, map_to_traffic_light, get_model_filenames, preprocess_text_for_cnn
from .api.news import get_news
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v2")
MODELS_DIR_v3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v3/models")
MODELS_PILOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/pilot/model")

def load_naive_bayes():
    """Load and return the Naive Bayes model."""
    nb_model_path = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')
    try:
        nb_model = joblib.load(nb_model_path)
        logger.info("Naive Bayes model loaded successfully.")
        return nb_model
    except Exception as e:
        logger.error(f"Error loading Naive Bayes model: {e}")
        raise

def load_tfidf_vectorizer():
    """Load and return the TF-IDF vectorizer."""
    tfidf_vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    try:
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        logger.info("TF-IDF vectorizer loaded successfully.")
        return tfidf_vectorizer
    except Exception as e:
        logger.error(f"Error loading TF-IDF vectorizer: {e}")
        raise

def load_cnn_model():
    """Load and return the CNN model."""
    cnn_model_path = os.path.join(MODELS_DIR_v3, 'cnn_model.h5')
    try:
        cnn_model = load_model(cnn_model_path)
        logger.info("CNN model loaded successfully.")
        return cnn_model
    except Exception as e:
        logger.error(f"Error loading CNN model: {e}")
        raise

def load_cnn_tokenizer():
    """Load and return the CNN tokenizer."""
    tokenizer_path = os.path.join(MODELS_DIR_v3, 'tokenizer.pkl')
    try:
        tokenizer = joblib.load(tokenizer_path)
        logger.info("CNN tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading CNN tokenizer: {e}")
        raise

def load_pilot_model():
    """Load and return the Pilot CNN model."""
    pilot_model_path = os.path.join(MODELS_PILOT, 'cnn_model_pilot.h5')
    try:
        pilot_model = load_model(pilot_model_path)
        logger.info("Pilot CNN model loaded successfully.")
        return pilot_model
    except Exception as e:
        logger.error(f"Error loading Pilot CNN model: {e}")
        raise

def load_pilot_tokenizer():
    """Load and return the Pilot tokenizer."""
    tokenizer_path = os.path.join(MODELS_PILOT, 'tokenizer_pilot.pkl')
    try:
        tokenizer = joblib.load(tokenizer_path)
        logger.info("Pilot tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading Pilot tokenizer: {e}")
        raise

def get_predictions(text, model_filename):
    """Get predictions for the given text using the specified model."""
    try:
        if 'naive_bayes' in model_filename.lower():
            # Load Naive Bayes model and TF-IDF vectorizer
            nb_model = load_naive_bayes()
            tfidf_vectorizer = load_tfidf_vectorizer()

            # Preprocess text for Naive Bayes
            preprocessed_text = preprocess_text_for_nb(text)

            # Transform input text using the TF-IDF vectorizer
            transformed_text = tfidf_vectorizer.transform([preprocessed_text])

            # Make predictions using Naive Bayes
            predictions = nb_model.predict_proba(transformed_text)[0]
            logger.info("Naive Bayes predictions: %s", predictions)
            return predictions

        elif 'cnn_model' in model_filename.lower():
            # Load CNN model and tokenizer
            cnn_model = load_cnn_model()
            tokenizer = load_cnn_tokenizer()

            # Preprocess text for CNN
            max_sequence_length = 24512  # Adjust this value based on your model's input shape
            preprocessed_text = preprocess_text_for_cnn(text, tokenizer, max_sequence_length)

            # Make predictions using CNN
            predictions = cnn_model.predict(preprocessed_text)
            logger.info("CNN predictions: %s", predictions)
            return predictions
        
        elif 'pilot_model' in model_filename.lower():
            # Load Pilot model and tokenizer
            pilot_model = load_pilot_model()
            tokenizer = load_pilot_tokenizer()

            max_sequence_length = 3505  # Adjust this value based on your model's input shape
            preprocessed_text = preprocess_text_for_cnn(text, tokenizer, max_sequence_length)

            # Make predictions using Pilot model
            predictions = pilot_model.predict(preprocessed_text)
            logger.info("Pilot CNN predictions: %s", predictions)
            return predictions

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

def extract_fake_news_words(text, predictions, threshold=0.3):
    """Extract words from text that are indicative of fake news based on prediction thresholds."""
    try:
        # Tokenize and preprocess text
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Zip tokens with corresponding predictions and filter based on threshold
        token_predictions = zip(tokens, predictions)
        fake_news_words = [token for token, pred in token_predictions if pred >= threshold]

        logger.info("Extracted fake news words: %s", fake_news_words)
        return fake_news_words

    except Exception as e:
        logger.error(f"Error extracting fake news words: {e}")
        return []

# Result page view
def result(request):
    line = request.GET.get('line', '')
    model_filename = request.GET.get('model', '')

    if not line or not model_filename:
        logger.warning("Missing line or model filename")
        return render(request, 'error.html', {'error_message': 'Missing line or model filename'})

    # Get predictions
    predictions = get_predictions(line, model_filename)

    # Calculate traffic light based on the predictions
    traffic_light = map_to_traffic_light(predictions)

    # Get indicative words for fake news
    indicative_words = extract_fake_news_words(line, predictions)

    return render(request, 'result.html', {
        'line': line,
        'model': model_filename,
        'predictions': predictions,
        'traffic_light': traffic_light,
        'indicative_words': indicative_words
    })

# Landing page view
def landing(request):
    return render(request, 'landing.html')

# Home page
def home(request):

    model_filename = request.GET.get('model')

    if not model_filename:
        # If no model is selected, render the landing page, fix for user movement between test and home page.
        return render(request, 'landing.html')
    
    try:
        data = get_news('NVDA', '25', '03', '2024')
        articles = data.get('results', [])

        for article in articles:
            # Get predictions for the article using the specified model
            predictions = get_predictions(article['description'], model_filename)

            # Calculate traffic light based on the predictions
            article['traffic_light'] = map_to_traffic_light(predictions)

            # Get indicative words for fake news
            indicative_words = extract_fake_news_words(article['description'], predictions)
            article['indicative_words'] = indicative_words

        return render(request, 'index.html', {
            'articles': articles,
            'selected_model': model_filename  # Pass the selected model to the template
        })

    except Exception as e:
        logger.error(f"Error in home view: {e}")
        return render(request, 'error.html', {'error_message': 'An error occurred while fetching the news'})


# Testing page view
def testing(request):
    try:
        # Get all model filenames
        model_files = get_model_filenames(MODELS_DIR_v3)

        # Pass the model filenames to the template
        return render(request, 'testing.html', {'models': model_files})

    except Exception as e:
        logger.error(f"Error in testing view: {e}")
        return render(request, 'error.html', {'error_message': 'An error occurred while loading the testing page'})

# API documentation page view
def api(request):
    return render(request, 'api.html')
