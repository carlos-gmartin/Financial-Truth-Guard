def read_data():
    # Function to read the true and fake datasets
    true_file_path = '../data/archive/DataSet_Misinfo_TRUE.csv'
    fake_file_path = '../data/archive/DataSet_Misinfo_FAKE.csv'

    # Read true dataset
    df_true = pd.read_csv(true_file_path)

    # Read fake dataset
    df_fake = pd.read_csv(fake_file_path)

    return df_true, df_fake

def preprocessing_data(df_true, df_fake):
    # Function to preprocess the data
    # Removing additional columns
    df_true.drop(columns='Unnamed: 0', axis=1, inplace=True)
    df_fake.drop(columns='Unnamed: 0', axis=1, inplace=True)
    
    # Adding labels
    df_true['label'] = 1
    df_fake['label'] = 0
    
    # Joining datasets
    df = pd.concat([df_true, df_fake])
    
    # Removing all of the missing values
    df.dropna(how='any', inplace=True)
    
    return df

def preprocess_text(df):
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize WordNetLemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    X = []
    Y = df['label'].values

    for text in df['text'].values:
        # Tokenization
        tokens = word_tokenize(text)
        
        # Lowercase conversion and stopword removal
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
        
        # Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        # Join tokens back into text
        processed_text = ' '.join(lemmatized_tokens)
        
        X.append(processed_text)

    return X, Y

def vectorization(X):
    # Function to vectorize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    max_len = X_pad.shape[1]
    return X_pad, vocab_size, max_len, tokenizer

def create_cnn_model(vocab_size, max_len):
    # Function to create the CNN model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, vocab_size, max_len):
    # Function to train the CNN model
    model = create_cnn_model(vocab_size, max_len)
    model.summary()

    # Define the ModelCheckpoint callback to save the model every 5 epochs
    checkpoint_path = 'model_checkpoint.h5'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=False,
                                          save_best_only=True,
                                          period=5)

    history = model.fit(X_train, y_train, 
                        epochs=5, 
                        validation_data=(X_test, y_test), 
                        callbacks=[checkpoint_callback],
                        verbose=1)
    return model, history

def evaluate_model(model_file, tokenizer_file, X_test, y_test,
save_confusion_matrix_image=True, 
confusion_matrix_image_file="Confusion_matrix.jpg"):
     # Load the trained model
    model = load_model(model_file)
    
    # Load the tokenizer
    tokenizer = joblib.load(tokenizer_file)
    
    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    
    # Predict probabilities for testing data
    y_pred_probs = model.predict(X_test)
    
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
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
        xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(confusion_matrix_image_file)
        print(f"\nConfusion matrix image saved to '{confusion_matrix_image_file}'.")
        plt.close()

if __name__ == "__main__":
    # Read data
    df_true, df_fake = read_data()
    df = preprocessing_data(df_true, df_fake)
    X, Y = preprocess_text(df)
    X_pad, vocab_size, max_len, tokenizer = vectorization(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pad, Y, test_size=0.2, random_state=42)
    
    model_file = 'cnn_model.h5'
    tokenizer_file = 'tokenizer.pkl'
    
    if not (os.path.exists(model_file) and os.path.exists(tokenizer_file)):
        # Train the model if the files don't exist
        model, history = train_model(X_train, y_train, X_test, y_test, vocab_size, max_len)
        
        # Save the trained model and tokenizer
        model.save('cnn_model.h5')
        joblib.dump(tokenizer, 'tokenizer.pkl')
    
    # Evaluate the model
    evaluate_model(model_file, tokenizer_file, X_test, y_test)