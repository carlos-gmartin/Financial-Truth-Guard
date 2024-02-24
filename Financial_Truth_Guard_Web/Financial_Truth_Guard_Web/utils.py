import re
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

# Sample usage:
# input_text = "This is a news article text that you want to classify."
# preprocessed_text_svm = preprocess_text_for_svm(input_text)
# preprocessed_text_rf = preprocess_text_for_rf(input_text)
