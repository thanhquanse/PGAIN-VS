from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

class Generalizer(BaseEstimator):
    def __init__(self, estimator = LinearRegression()): self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X,y)
