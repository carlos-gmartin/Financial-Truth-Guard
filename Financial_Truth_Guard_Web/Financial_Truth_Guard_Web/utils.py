import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a global TfidfVectorizer for SVM models
tfidf_vectorizer = TfidfVectorizer()

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

# Function to preprocess text for SVM models
def preprocess_text_for_svm(text):
    global tfidf_vectorizer  # Use the global tfidf_vectorizer
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    return text_tfidf

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

def preprocess_text_for_cnn(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def map_to_traffic_light(predictions, thresholds=(0.3, 0.7)):
    green_threshold, yellow_threshold = thresholds
    max_prob_index = np.argmax(predictions)
    max_prob = float(predictions[max_prob_index])  # Convert max_prob to float
    
    print("Max Probability:", max_prob)  # Add this line for debugging

    if max_prob < green_threshold:
        return 'Green'
    elif max_prob < yellow_threshold:
        return 'Yellow'
    else:
        return 'Red'


# Function to get all model filenames from the models directory
def get_model_filenames():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    return model_files
