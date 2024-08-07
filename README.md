## Sentiment Analysis Project in Python using NLP

This project aims to build a sentiment analysis system using Python's Natural Language Processing (NLP) libraries. The system will analyze textual data (twitter data) and classify the sentiment as positive or negative.

**Key Technologies:**

* **Python:** The programming language for development.
* **NLTK:** Natural Language Toolkit library for various NLP tasks like tokenization, stemming/lemmatization, and stop word removal.
* **Scikit-learn:** Machine learning library containing tools for building and evaluating the sentiment analysis model.
* **TF-IDF and Bag-of-Words:** Techniques to convert text data into numerical features suitable for machine learning algorithms.

**Preprocessing Steps:**
* **Data Analysis:** Analysing the dataset and finding patterns in the data. 
* **Text Cleaning:** Remove irrelevant information like punctuation, special characters, and HTML tags.
* **Tokenization:** Break down the text into individual words or sentences.
* **Stop Word Removal:** Eliminate common words ("the", "a", "an") that don't contribute to sentiment analysis.
* **Stemming/Lemmatization:** Reduce words to their base form (e.g., "running" -> "run", "better" -> "good"). Stemming might lead to grammatically incorrect words, while lemmatization aims for proper dictionary forms.

**Feature Engineering:**

* **TF-IDF or Bag-of-Words:** Convert preprocessed text data into numerical features. TF-IDF considers word importance within a document and across the corpus, while Bag-of-Words simply counts word occurrences.

**Model Building and Evaluation:**

* Train a machine learning model (Logistic Regression) on labeled sentiment data (positive, negative)
* Evaluate the model's performance using metrics like accuracy, precision, and recall.


Run app.py using the command **streamlit run app.py**
