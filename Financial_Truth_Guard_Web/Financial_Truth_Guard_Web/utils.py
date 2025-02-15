import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def preprocess_text(text):
    text = text.lower()
    
    # Remove punctuation using regular expression
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to preprocess text for Random Forest and Naive Bayes models
def preprocess_text_for_rf(text):
    preprocessed_text = preprocess_text(text)
    return preprocessed_text

# Function to tokenize and preprocess text for Naive Bayes
def preprocess_text_for_nb(text):
    #nltk.download('stopwords')
    #nltk.download('punkt')

    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    tokens = tokenizer.tokenize(text.lower())
    filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
    preprocessed_text = ' '.join(filtered_words)

    return preprocessed_text

def preprocess_text_for_cnn(text, tokenizer, max_sequence_length):
    # Preprocess text using the general preprocess_text function
    preprocessed_text = preprocess_text(text)
    
    # Tokenize and pad input text using the tokenizer
    sequences = tokenizer.texts_to_sequences([preprocessed_text])
    X_pad = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)
    return X_pad


def map_to_traffic_light(predictions, thresholds=(0.25, 0.5)):
    green_threshold, yellow_threshold = thresholds
    max_prob = np.max(predictions)  # Get the maximum probability from the predictions

    if max_prob >= yellow_threshold:
        return 'Red'  # Automatically classify as red if probability is 0.5 or higher
    elif max_prob >= green_threshold:
        return 'Yellow'
    else:
        return 'Green'


# Function to get all model filenames from the models directory
def get_model_filenames(MODELS_DIR):
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]
    return model_files
