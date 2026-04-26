from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def get_advanced_model():
    """
    Returns our proposed advanced model for bug report classification.
    We use a Support Vector Machine (LinearSVC) paired with TF-IDF.
    SVMs are highly effective for high-dimensional, sparse text data and 
    typically outperform Naive Bayes algorithms while maintaining strong interpretability.
    """
    pipeline = Pipeline([
        # We increase n-gram range to (1, 2) to capture short phrases like "memory leak"
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),
        ('clf', LinearSVC(random_state=42, class_weight='balanced', max_iter=2000))
    ])
    
    return pipeline
