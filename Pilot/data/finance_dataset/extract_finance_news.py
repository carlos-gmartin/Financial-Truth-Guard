import pandas as pd
import re

# Read the CSV file containing news articles and labels
try:
    df = pd.read_csv('../archive/DataSet_Misinfo_TRUE.csv')
    print("News articles dataset read successfully.")
except FileNotFoundError:
    print("Error: News articles dataset file not found.")
    exit()

# Read the text file containing finance words
try:
    with open('./finance_words.txt', 'r') as file:
        finance_words_str = file.read()
    print("Finance words read successfully.")
except FileNotFoundError:
    print("Error: Finance words file not found.")
    exit()

# Process the finance words
finance_words = [re.escape(word.strip()) for word in finance_words_str.split(',')]
print("Finance words escaped successfully.")

# Filter news articles containing finance words
try:
    pattern = '|'.join(finance_words)
    finance_related_articles = df[df['text'].str.contains(pattern, case=False, na=False)]
    print("Finance-related articles filtered successfully.")
except re.error as e:
    print(f"Regex error: {e}")
    exit()

# Check if any articles were found
if finance_related_articles.empty:
    print("No finance-related articles found.")
else:
    print(f"Found {len(finance_related_articles)} finance-related articles.")

# Save the filtered finance-related articles to a new CSV file
try:
    finance_related_articles.to_csv('Finance_Related_Articles_TRUE.csv', index=False)
    print("Finance-related articles saved to 'Finance_Related_Articles_TRUE.csv' successfully.")
except Exception as e:
    print(f"Error saving file: {e}")
