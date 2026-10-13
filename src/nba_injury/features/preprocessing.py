"""Unfitted sklearn preprocessing factories for later training-fold use."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .historical import FEATURE_NAMES


class PredictorSchemaGuard(TransformerMixin, BaseEstimator):
    """Reject extra metadata/label columns instead of accidentally treating them as X."""
    def fit(self,X,y=None):
        self._validate(X)
        self.feature_names_in_=np.array(FEATURE_NAMES,dtype=object)
        self.n_features_in_=len(FEATURE_NAMES)
        return self

    def transform(self,X):
        self._validate(X)
        return X.loc[:,FEATURE_NAMES].copy()

    @staticmethod
    def _validate(X):
        if not isinstance(X,pd.DataFrame):
            raise TypeError("Use a named predictor DataFrame so its schema can be audited")
        if X.columns.duplicated().any() or set(X.columns)!=set(FEATURE_NAMES):
            raise ValueError("X must contain exactly the approved predictor names, without metadata/targets")
        if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
            raise TypeError("All approved predictors are numeric")
        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("Infinite predictor values")

    def get_feature_names_out(self,input_features=None):
        return np.array(FEATURE_NAMES,dtype=object)


def make_preprocessor(model_family="linear"):
    """Fit inside each later training fold; never pre-fit on all splits for CV.

    Median imputation leaves the existing any-prior-episode and attempt-mean
    variables available to identify structural missingness. No duplicate global
    missingness indicators or categorical/identity encoding are introduced.
    """
    if model_family not in {"linear","tree"}:
        raise ValueError("model_family must be 'linear' or 'tree'")
    steps=[("schema",PredictorSchemaGuard()),("impute",SimpleImputer(strategy="median",keep_empty_features=True))]
    if model_family=="linear":
        steps.append(("scale",StandardScaler()))
    return Pipeline(steps)
