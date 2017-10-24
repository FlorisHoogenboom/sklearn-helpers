from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class Transformer(BaseEstimator, TransformerMixin):
    """A helper class that turns a Python function that
    transforms some data into a Scikit-Learn transformer.

    Examples
    ----------
    Can be best used as a decorator on an existing function
        ```
        @Transformer
        def my_transforming_function(data):
            pass
        ````

    Parameters
    ----------
    func : callable that contains the transformation.

    Attributes
    ----------
    _transformer : private attribute that contains the transformer
        function used
    """
    def __init__(self, func):
        if not callable(func):
            raise ValueError('Funtion should be callable')

        self._transformer = func

    def fit(self, data):
        """Empty shell only to adhere to the scikit-learn api

        Returns
        ----------
        self : the instance of the transformer
        """
        return self

    def transform(self, data):
        """Transform using the transformer function

        Parameters
        ----------
        data : the dataset to transform
        """
        data = data.copy()
        return self._transformer(data)


class PandasTransformer(Transformer):
    """A helper class that wraps a given transformation function
    and makes it adhere to she Scikit-Learn api.
    """

    def transform(self, data):
        """Transform using the transformer function

        Parameters
        ----------
        data : the dataset to transform
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Data should be a pandas dataframe')

        return super(PandasTransformer, self).transform(data)