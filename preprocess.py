from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ProbTransform:
    def __init__(self, cols):
        self._col = cols
        self._probs = None

    def fit(self, X, y=None):
        X_ = X.copy()
        X_['prob'] = 1
        self._probs = X_.groupby(self._col)['prob'].agg('size'). \
                groupby(level=self._col[0]).apply(lambda x: x / sum(x)). \
                    reset_index()
        return self

    def transform(self, X, y=None):
        return X.merge(self._probs, on=self._col, how='inner').drop(columns=self._col[1], inplace=False)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X, y=None):
        X_ = X.rename(columns={'variable': self._col[0]})
        X_ = X_.merge(self._probs, on=[self._col[0], 'prob'], how='inner').drop(columns='prob', inplace=False)
        return X_


class OneHot:
    def __init__(self, cols):
        self._col = cols
        self._orig = None
        self._melt = None

    def fit(self, X, y=None):
        self._orig = list(X.drop(columns=self._col, inplace=False).columns)
        self._melt = list(X[self._col].unique())
        return self

    def transform(self, X, y=None):
        X1 = pd.concat([X,pd.get_dummies(X[self._col])],axis=1).drop([self._col],axis=1).to_numpy()
        return X1
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X, y=None):
        X_ = pd.DataFrame(X, columns=self._orig + self._melt). \
            melt(id_vars=self._orig, value_vars=self._melt).drop(columns='value')
        return(X_)