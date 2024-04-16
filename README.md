# Financial Truth Guard

Financial Truth Guard is a web application designed to analyze news articles and predict whether they contain misleading or false information related to financial markets. It utilizes machine learning models, including Naive Bayes and Convolutional Neural Networks (CNN), to classify news articles based on their content.

## Features

- **Model Selection**: Choose between Naive Bayes and CNN models to classify news articles.
- **Traffic Light Indicator**: Each article is assigned a traffic light indicator (green, yellow, or red) based on the likelihood of containing fake news.
- **Fake News Words Highlighting**: Extracts words from articles that are indicative of fake news and displays them.

## Installation

To install Financial Truth Guard locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python packages using pip:
`pip install -r requirements.txt`
3. Run the Django development server:
`python manage.py runserver`
5. Access the application in your web browser at `http://localhost:8000`.

## Usage

- Upon visiting the homepage, Financial Truth Guard retrieves recent news articles related to financial markets.
- Select a news article to view its details.
- Choose a model (Naive Bayes or CNN) to classify the article.
- The traffic light indicator provides a visual representation of the likelihood of fake news.
- The extracted fake news words are displayed for further analysis.

## Contributing

Contributions to Financial Truth Guard are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/my-feature` or `git checkout -b bugfix/my-bug-fix`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to your branch: `git push origin feature/my-feature`.
5. Submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Financial Truth Guard was developed as part of the CMP2024 course project at [University of East Anglia]. Special thanks to [Farhana Liza] for guidance and support.



