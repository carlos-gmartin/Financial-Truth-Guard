import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing data
def preprocessing_data(df_true, df_fake):
    # Removing additional columns named 'Unnamed: 0' from both true and fake datasets
    if 'Unnamed: 0' in df_true.columns:
        df_true.drop(columns='Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in df_fake.columns:
        df_fake.drop(columns='Unnamed: 0', axis=1, inplace=True)

    # Adding labels to the datasets: 1 for true news, 0 for fake news
    df_true['label'] = 1
    df_fake['label'] = 0

    # Combining the true and fake datasets into one dataframe
    df = pd.concat([df_true, df_fake])
    # Removing all rows with any missing values
    df.dropna(how='any', inplace=True)
    
    # Preprocess text
    X, Y = preprocess_text(df)
    
    return X, Y  # Returning the preprocessed text data and labels

# Function to preprocess text using NLTK
def preprocess_text(df):
    # Extracting labels (Y) and text data (X) from the dataframe
    Y = df['label'].values
    X = df['text'].apply(process_text_helper)
    return X, Y  # Returning the text data and labels

def process_text_helper(text):
    # Lemmatize and remove stopwords
    return ' '.join(lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word.isalpha() and word.lower() not in stop_words)

# TF-IDF vectorization
def tfidf_vectorization(X):
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit the number of features for efficiency
    X_tfidf = vectorizer.fit_transform(X).toarray()
    vocab_size = len(vectorizer.get_feature_names_out())
    return X_tfidf, vocab_size, vectorizer

# Dense model creation.
def create_dense_model(input_shape):
    # Function to create a Dense Neural Network model for text classification
    model = Sequential()  # Initialize the model as a Sequential model, which is a linear stack of layers
    # Add an InputLayer to define the shape of the input
    model.add(InputLayer(input_shape=input_shape))
    # Add a Dense layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    # Add a Dropout layer to prevent overfitting by randomly setting 50% of the input units to 0 during training
    model.add(Dropout(0.5))
    # Add another Dense layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))
    # Add another Dropout layer
    model.add(Dropout(0.5))
    # Add a final Dense layer with 1 unit and sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model  # Return the constructed model

# Training the model.
def train_model(X_train, y_train, X_test, y_test, input_shape):
    # Function to train the Dense Neural Network model
    model = create_dense_model(input_shape)
    model.summary()

    # Define the ModelCheckpoint callback to save the model every 5 epochs
    checkpoint_path = 'model_checkpoint.h5'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=False,
                                          save_best_only=True,
                                          save_freq='epoch')

    history = model.fit(X_train, y_train, 
                        epochs=10,  # Adjust the number of epochs as needed
                        validation_data=(X_test, y_test), 
                        callbacks=[checkpoint_callback],
                        batch_size=512,  # Batch size for more efficient training with large data
                        verbose=1)
    return model, history

# Confusion Matrix and model evaluating.
def evaluate_model(model_file, vectorizer_file, X_test, y_test, save_confusion_matrix_image=True, confusion_matrix_image_file="Confusion_matrix.jpg"):
    # Load the trained model
    model = load_model(model_file)
    
    # Load the vectorizer
    vectorizer = joblib.load(vectorizer_file)
    
    # Transform the test data using the loaded vectorizer
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    
    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test_tfidf, y_test)
    
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    
    # Predict probabilities for testing data
    y_pred_probs = model.predict(X_test_tfidf)
    
    # Convert probabilities to class labels
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)  # Assuming binary classification with threshold of 0.5
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Define class labels
    class_labels = ['Fake', 'True']  # Class labels: 'Fake' for 0 and 'True' for 1
    
    # Plot confusion matrix
    if save_confusion_matrix_image:
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(confusion_matrix_image_file)
        print(f"\nConfusion matrix image saved to '{confusion_matrix_image_file}'.")
        plt.close()

# Function to read data
def read_data():
    df_true = pd.read_csv('../data/archive/DataSet_Misinfo_TRUE.csv')
    df_fake = pd.read_csv('../data/archive/DataSet_Misinfo_FAKE.csv')
    return df_true, df_fake

if __name__ == "__main__":
    # Read data
    df_true, df_fake = read_data()
    df = preprocessing_data(df_true, df_fake)
    X, Y = preprocess_text(df)

    # Use TF-IDF Vectorization
    X_tfidf, vocab_size, vectorizer = tfidf_vectorization(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.2, random_state=42)
    
    input_shape = (X_train.shape[1],)
    model_file = 'tfidf_dense_model.h5'
    vectorizer_file = 'tfidf_vectorizer.pkl'
    
    if not (os.path.exists(model_file) and os.path.exists(vectorizer_file)):
        # Train the model if the files don't exist
        model, history = train_model(X_train, y_train, X_test, y_test, input_shape)
        
        # Save the trained model and vectorizer
        model.save(model_file)
        joblib.dump(vectorizer, vectorizer_file)
    
    X_test_list = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test  # Convert X_test to a list of strings if it's a NumPy array
    evaluate_model(model_file, vectorizer_file, X_test_list, y_test)
