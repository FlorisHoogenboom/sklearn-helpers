from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
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
            raise ValueError('Function should be callable')

        self._transformer = func

    def fit(self, data, y):
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
        check_array(data, dtype=None, copy=False)
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


class ColumnSelector(Transformer):
    """A helper class that selects a specific subset of columns for
    further analysis.

    Parameters
    ----------
    columns : a list of integers that specifies which columns should be kept

    Attributes
    ----------
    columns : attribute containning the columns being selected
    """

    def __init__(self, columns=None):
        # Validate the columns, raises if an invalid value is set
        self.columns = columns

        # Call superclass with the desired transformer function
        super(ColumnSelector, self).__init__(self.transform_func)

    def transform_func(self, data):
        """Selects columns in self.columns of the given argument data.

        Parameters
        ----------
        data : A array like shape that support NumPy like indexing.
        """
        if self.columns is None:
            return data

        return data[:,self.columns]

    @property
    def columns(self):
        return self.columns_

    @columns.setter
    def columns(self, columns):
        if columns is None:
            return

        if (
            type(columns) is not list or
            not all(map(lambda x: isinstance(x, int), columns))
        ):
            raise ValueError('Columns should be a list of integers')

        self.columns_ = columns


class PandasColumnSelector(ColumnSelector):
    def transform_func(self, data):
        """Selects columns in self.columns of the given Pandas DataFrame.

        Parameters
        ----------
        data : Pandas DataFrame on which column selection should be performed.
        """
        if self.columns is None:
            return data

        return super(PandasColumnSelector, self).transform_func(
            data.iloc
        )
