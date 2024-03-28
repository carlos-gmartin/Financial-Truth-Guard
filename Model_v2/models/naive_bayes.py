import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import joblib

def read_data():
    true_file_path = '../data/archive/DataSet_Misinfo_TRUE.csv'
    fake_file_path = '../data/archive/DataSet_Misinfo_FAKE.csv'
    
    # Read true dataset if path exists
    if os.path.exists(true_file_path):
        df_true = pd.read_csv(true_file_path)
    else:
        print(f"File '{true_file_path}' not found.")

    # Read fake dataset if path exists
    if os.path.exists(fake_file_path):
        df_fake = pd.read_csv(fake_file_path)
    else:
        print(f"File '{fake_file_path}' not found.")

    return df_true, df_fake

def preprocessing_data(df_true, df_fake):
    # Removing additional columns
    df_true.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
    df_fake.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)

    # Adding labels
    df_true['label'] = 1
    df_fake['label'] = 0

    # Joining datasets.
    df = pd.concat([df_true, df_fake])

    #df.info()

    # Removing all of the missing values
    df.dropna(how = 'any', inplace = True)

    #df.info()
    return df

def preprocess_text(df):

    nltk.download('stopwords')
    nltk.download('punkt')

    Y = df['label'].values
    X = []

    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    for i in df['text'].values:
        temp = []
        sentences = nltk.sent_tokenize(i)
        for sentence in sentences:
            sentence = sentence.lower()
            tokens = tokenizer.tokenize(sentence)
            filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w)>1]
            temp.extend(filtered_words)
        X.append(temp)
    del df

    #print(X[0])

    return X, Y

def vectorization(X):
    X = [" ".join(sent) for sent in X]  # Convert list of words to list of sentences
    tfidf = TfidfVectorizer(max_features=2**12)
    tfidf_matrix = tfidf.fit_transform(X)
    return tfidf, tfidf_matrix.toarray()  # Return the transformed data


def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = MultinomialNB()

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'naive_bayes_model.pkl')

    # Predict on the testing set
    predictions = model.predict(X_test)

    print(f"Accuracy: {round(accuracy_score(y_test, predictions), 2)*100}%")

    cm = confusion_matrix(y_test, predictions)
    class_names = ['Real', 'Fake']
    plt.figure(figsize=(8, 5))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, 
                annot_kws={"size": 16}, 
                fmt='g', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return model

if __name__ == "__main__":
    df_true, df_fake = read_data()
    
    df = preprocessing_data(df_true, df_fake)

    X, Y = preprocess_text(df)

    vectorizer, vector_x = vectorization(X)

    model = train_model(vector_x, Y)

    # Save the vectorizer
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
