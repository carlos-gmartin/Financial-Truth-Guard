# Financial Truth Guard

Financial Truth Guard is a web application designed to analyze news articles and predict whether they contain misleading or false information related to financial markets. It utilizes machine learning models, including Naive Bayes and Convolutional Neural Networks (CNN), to classify news articles based on their content.

## Features

- **Model Selection**: Choose between Naive Bayes and CNN models to classify news articles.
- **Traffic Light Indicator**: Each article is assigned a traffic light indicator (green, yellow, or red) based on the likelihood of containing fake news.
- **Fake News Words Highlighting**: Extracts words from articles that are indicative of fake news and displays them.

## Installation

To install Financial Truth Guard locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python packages using pip, possible start a new env for this:
`pip install -r requirements.txt`
3. Run the Django development server:
`python manage.py runserver`
5. Access the application in your web browser at `http://localhost:8000`.

## Usage

- Upon visiting the homepage, Financial Truth Guard retrieves recent news articles related to financial markets.
- Select a news article to view its details.
- Choose a model (Naive Bayes or CNN or pilot model) to classify the article.
- The traffic light indicator provides a visual representation of the likelihood of fake news.
- The extracted fake news words are displayed for further analysis.

## Acknowledgements

Financial Truth Guard was developed as part of the CMP2024 course project at [University of East Anglia]. Special thanks to [Farhana Liza] for guidance and support.



