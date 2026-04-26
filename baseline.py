from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def get_baseline_model():
    """
    Returns the baseline model for bug report classification.
    The baseline specified by Lab 1 is Naive Bayes paired with TF-IDF.
    """
    # Create a pipeline that first extracts TF-IDF features and then trains a Naive Bayes classifier.
    # stop_words='english' removes common English words that don't add predictive value.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', MultinomialNB())
    ])
    
    return pipeline
