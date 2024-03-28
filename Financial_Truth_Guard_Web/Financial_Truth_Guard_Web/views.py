import os
import joblib
import numpy as np
from django.shortcuts import render
from .utils import preprocess_text_for_svm, preprocess_text_for_rf, preprocess_text_for_nb, preprocess_text
from .api.news import get_news

# models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_v2")

# Function to get all model filenames from the models directory
def get_model_filenames():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return model_files

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

# Function to load a model from a filename
def load_model(filename):
    model_path = os.path.join(MODELS_DIR, filename)
    model = joblib.load(model_path)
    return model

def map_to_traffic_light(predictions, thresholds=(0.3, 0.7)):
    green_threshold, yellow_threshold = thresholds
    max_prob_index = np.argmax(predictions)
    max_prob = predictions[max_prob_index]

    if max_prob < green_threshold:
        return 'Green'
    elif max_prob < yellow_threshold:
        return 'Yellow'
    else:
        return 'Red'
    
# Function to predict label for input text
def predict(input_text, model):
    try:
        prediction = model.predict(input_text)[0]

        if(prediction == "Green"):
            return "Green"
        elif(prediction == "Yellow"):
            return "Yellow"
        else:
            return "Red"
    except Exception as e:
        return f"Error during prediction: {e}"

def get_predictions(line, model_filename):
    # Load the model
    model = load_model(model_filename)

    # Preprocess text based on the selected model
    if 'naive_bayes' in model_filename.lower():
        preprocessed_text = preprocess_text_for_nb(line)
    else:
        return "Model not supported"

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = load_tfidf_vectorizer()
    # Transform input text using the TF-IDF vectorizer
    transformed_text = tfidf_vectorizer.transform([preprocessed_text])

    try:
        # Make predictions
        predictions = model.predict_proba(transformed_text)[0]

        print("Probabilities:", predictions)

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
    for article in articles:
        result = get_predictions(article['description'], 'naive_bayes_model.pkl')

        article['result'] = result
        # Calculate angle based on the result
        article['traffic_light'] = map_to_traffic_light(result)

    return render(request, 'index.html', {'articles': articles})

# testing page view
def test_web(request):
    # Get all model filenames
    model_files = get_model_filenames()
    
    # Pass the model filenames to the template
    return render(request, 'testing.html', {'models': model_files})

def api(request):
    return render(request, 'api.html')




