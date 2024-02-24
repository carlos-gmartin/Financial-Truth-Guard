import os
import joblib
from django.shortcuts import render
from .utils import preprocess_text_for_svm, preprocess_text_for_rf

# Define the path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

# Function to get all model filenames from the models directory
def get_model_filenames():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return model_files

# Function to load a model from a filename
def load_model(filename):
    model_path = os.path.join(MODELS_DIR, filename)
    model = joblib.load(model_path)
    return model

def load_random_forest():
    # Load the Random Forest model
    rf_model = load_model('random_forest_model.pkl')
    return rf_model

def load_svm():
    # Load the SVM model
    svm_model = load_model('svm_model.pkl')
    return svm_model

def load_logistic_regression():
    # Load the Logistic Regression model
    lr_model = load_model('best_logistic_regression_model.pkl')
    return lr_model

def load_naive_bayes():
    # Load the Naive Bayes model
    nb_model = load_model('naive_bayes_model.pkl')
    return nb_model

# Prediction function
def predict(input_text, model):
    try:
        prediction = model.predict([input_text])[0]
        return "Fake News" if prediction == 1 else "True News"
    except Exception as e:
        return f"Error during prediction: {e}"

# Function to handle prediction request
def get_predictions(line, model_filename):
    if 'svm' in model_filename.lower():
        model = load_svm()
    elif 'random_forest' in model_filename.lower():
        model = load_random_forest()
    elif 'logistic_regression' in model_filename.lower():
        model = load_logistic_regression()
    elif 'naive_bayes' in model_filename.lower():
        model = load_naive_bayes()
    else:
        return "Model not supported"
    
    result = predict(line, model)
    return result

# Result page view
def result(request):
    line = request.GET.get('line', '')
    model_filename = request.GET.get('model', '')
    
    if not line or not model_filename:
        return render(request, 'error.html', {'error_message': 'Missing line or model filename'})

    result = get_predictions(line, model_filename)
    
    return render(request, 'result.html', {'result': result})

# Home page view
def home(request):
    # Get all model filenames
    model_files = get_model_filenames()
    
    # Pass the model filenames to the template
    return render(request, 'index.html', {'models': model_files})
