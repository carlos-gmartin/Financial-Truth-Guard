import os
import joblib
from django.shortcuts import render
from .utils import preprocess_text

# Function to get all model filenames from the models directory
def get_model_filenames():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    models_directory = os.path.join(current_directory, "models")
    model_files = [f for f in os.listdir(models_directory) if f.endswith(".pkl")]
    return model_files

# Function to load a model from a filename
def load_model(filename):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_directory, "models", filename)
    model = joblib.load(model_path)
    return model

# Home page view
def home(request):
    # Get all model filenames
    model_files = get_model_filenames()
    
    # Pass the model filenames to the template
    return render(request, 'index.html', {'models': model_files})

# Prediction function
def predict(input_text, model):
    preprocessed_text = preprocess_text(input_text)
    prediction = model.predict([preprocessed_text])[0]
    return "Fake News" if prediction == 1 else "True News"

# Function to handle prediction request
def getPredictions(line, model_filename):
    model = load_model(model_filename)
    result = predict(line, model)
    return result

# Result page view
def result(request):
    line = str(request.GET['line'])
    model_filename = request.GET['model']
    
    result = getPredictions(line, model_filename)
    
    return render(request, 'result.html', {'result': result})
