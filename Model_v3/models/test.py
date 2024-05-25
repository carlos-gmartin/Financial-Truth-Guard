import json
import pickle

# Load data from pickle file
with open('tokenizer.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert data to JSON-serializable format
json_serializable_data = {
    'word_index': data.word_index,
    'word_counts': data.word_counts,
    # Add other attributes as needed
}

# Write JSON-serializable data to JSON file
with open('tokenizer.json', 'w') as json_file:
    json.dump(json_serializable_data, json_file)
