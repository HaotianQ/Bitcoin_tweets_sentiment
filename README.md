# Bitcoin_tweets_sentiment

This project uses Bitcoin Tweets data (source: https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets/tasks?taskId=3483) to do sentiment analysis on tweets text and utilizes tfidf vectorizer to convert a collection of raw documents to a matrix of TF-IDF features, then applies multiple classifiers (SGDClassifier, MultinomialNB, Randomforestclassifier) to predict the sentiment for tweets based on the training data

Last update: December 2021.

## Prequisites: Dependencies

Different sub-projects may have different requirements, as specified in the 
_requirements.txt_ files to be found in the various folders. We recommend using
[virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to 
keep environments isolated, i.e. creating a new environment:

`python3 -m venv venv`

then activating it and installing the required dependencies:

`source venv/bin/activate`

`pip install -r requirements.txt`

## Repo Structure

The repo is organized by folder: each folder contains either resources - e.g. text corpora or slides - or Python programs, divided by type. 

As far as ML is concerned, language-related topics are typically covered through notebooks, MLSys-related concepts are covered through Python scripts (not surprisingly!).

### Data
