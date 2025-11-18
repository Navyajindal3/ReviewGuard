# custom_vec.py
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

class CombinedPrefitVectorizer(BaseEstimator, TransformerMixin):
    """
    Concatenate the outputs of several *already-fitted* vectorizers
    (CountVectorizer, TfidfVectorizer, HashingVectorizer, etc.).
    All vectorizers are called with .transform on the same list of texts.
    """
    def __init__(self, vectorizers):
        self.vectorizers = vectorizers  # list of fitted vectorizer objects

    def fit(self, X, y=None):
        # Vectorizers are pre-fitted; nothing to fit here.
        return self

    def transform(self, X):
        mats = [v.transform(X) for v in self.vectorizers]
        return hstack(mats).tocsr()

    def __repr__(self):
        names = [type(v).__name__ for v in self.vectorizers]
        return f"CombinedPrefitVectorizer({names})"
