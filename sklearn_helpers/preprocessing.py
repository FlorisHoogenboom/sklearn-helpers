from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d, check_array, as_float_array
import numpy as np


class EnhancedLabelEncoder(LabelEncoder):
    def __init__(self, handle_unknown = 'error'):
        """A enhanced version of Scikit's LabelEncoder that can
        handle unseen Labels without erroring.

        Parameters
        ----------
        handle_unknown : str, 'error' or 'ignore' a
        """
        self.handle_unknown = handle_unknown

    def transform(self, y):
        """Transform labels to normalized encoding. If n levels have been
        seen during training (mapped to 0 till n-1), then unseen levels are
        mapped to the value n.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        if self.handle_unknown == 'ignore':
            check_is_fitted(self, 'classes_')
            y = column_or_1d(y, warn=True)

            classes = np.unique(y)
            return np.searchsorted(self.classes_, y)
        else:
            return super(EnhancedLabelEncoder,self).transform(y)


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown = 'error', categorical_features = None):
        """Encode multiple columns at one by applying LabelEncoder to each column.

        Parameters
        ----------
        handle_unknown : str, 'error' or 'ignore'
            Whether to raise an error or ignore if unseen categorical features
            are present during transformation.

        categorical_features : Array of column indices or labels
            The columns to which Label encoding should be applied
        """
        self.columns = categorical_features
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """
        Fit the MultiColumnLabelEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        Returns
        ----------
        self
        """
        X = check_array(X, copy=True, dtype='object')

        if self.columns is None:
            self.columns = list(range(X.shape[1]))

        self.encoders = {}
        for col in self.columns:
            self.encoders[col] = EnhancedLabelEncoder(
                handle_unknown=self.handle_unknown)
            self.encoders[col].fit(X[:,col])

        self._fitted = True

        return self

    def transform(self, X, y=None):
        """
        Transforms columns of X specified by categorical_features the underlying LabelEncoder().

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        Returns
        ----------
        A copy of X with the specified columns transformed
        """
        check_is_fitted(self, '_fitted')
        output = X.copy()
        for col in self.columns:
            output[:,col] = self.encoders[col].transform(output[:,col])

        if y:
            return as_float_array(output), y
        else:
            return as_float_array(output)