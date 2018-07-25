from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import pandas as pd
import numpy as np


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
        self.skip_validation = True

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
        if not self.skip_validation:
            check_array(
                data,
                dtype=None,
                copy=False,
                force_all_finite=False
            )

        data = data.copy()
        return self._transformer(data)


class PandasTransformer(Transformer):
    """A helper class that wraps a given transformation function
    and makes it adhere to she Scikit-Learn api.
    """
    def __init__(self, func):
        super(PandasTransformer, self).__init__(func)
        self.skip_validation = True

    @staticmethod
    def check_is_pandas_dataframe(data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Data should be a pandas dataframe')

        return data

    def transform(self, data):
        """Transform using the transformer function

        Parameters
        ----------
        data : the dataset to transform
        """
        self.check_is_pandas_dataframe(data)

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
    handle_unknown : attribute whether to ignore non-existing columns,
        str: 'error' (Default) or 'ignore'
    """

    def __init__(self, columns=None, handle_unknown='error', complement=False):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.complement = complement

        # Call superclass with the desired transformer function
        super(ColumnSelector, self).__init__(self.transform_func)

    # TODO: docs....
    def _columns_to_select(self, columns, data):
        if not self.complement:
            return columns
        else:
            if type(columns) is int:
                columns = [columns]
            all = list(range(data.shape[0]))
            return [col for col in all if col not in columns]

    # TODO: docs....
    def _check_columns(self, data):
        if type(self.columns) is int:
            return self.columns
        if self.handle_unknown == 'ignore':
            return [col for col in self.columns if col < data.shape[1]]
        elif len(self.columns) > 0 and max(self.columns) >= data.shape[1]:
            raise IndexError('Could not select the desired columns')
        else:
            return self.columns

    def transform_func(self, data):
        """Selects columns in self.columns of the given argument data.

        Parameters
        ----------
        data : A array like shape that support NumPy like indexing.
        """
        if self.columns is None:
            return data

        columns = self._check_columns(data)
        columns = self._columns_to_select(columns, data)

        return data[:,columns]

    @property
    def columns(self):
        return self.columns_

    @columns.setter
    def columns(self, columns):
        if columns is None:
            self.columns_ = None
            return

        if (type(columns) is not int and (
                type(columns) is not list or
                not all(map(lambda x: isinstance(x, int), columns)
            ))
        ):
            raise ValueError(
                'Columns should be a single integer or a list of integers'
            )

        self.columns_ = columns


class PandasColumnSelector(ColumnSelector, PandasTransformer):
    # TODO: docs....
    def _columns_to_select(self, columns, data):
        if not self.complement:
            return columns
        elif not self.named_columns:
            if type(columns) is int:
                columns = [columns]
            return super(PandasColumnSelector, self)._columns_to_select(
                columns,
                data
            )
        else:
            if type(columns) is str:
                columns = [columns]
            return [col for col in data.columns if not col in columns]

    # TODO: docs....
    def _check_columns(self, data):
        if type(self.columns) in (str, int):
            return self.columns
        if self.handle_unknown == 'ignore' and self.named_columns:
            return [col for col in self.columns if col in data.columns]
        elif self.handle_unknown == 'ignore' and not self.named_columns:
            return [col for col in self.columns if col < data.shape[1]]

        # If we do not handle unkowns
        if (
            self.named_columns and
            not set(self.columns).issubset(set(data.columns))
        ):
            raise IndexError(
                'Columns {0} not contained in axis.'.format(
                    set(self.columns).difference(set(data.columns))
                )
            )
        elif(
            not self.named_columns and
            len(self.columns) > 0 and
            not max(self.columns) <= data.shape[1]
        ):
            raise IndexError('Column indicecs out of bounds')
        else:
            return self.columns

    def transform_func(self, data):
        """Selects columns in self.columns of the given Pandas DataFrame.

        Parameters
        ----------
        data : Pandas DataFrame on which column selection should be performed.
        """
        if self.columns is None:
            return data

        columns = self._check_columns(data)
        columns = self._columns_to_select(columns, data)

        if self.named_columns:
            return data[columns]
        else:
            return data.iloc[:, columns]

    @property
    def columns(self):
        return self.columns_

    @columns.setter
    def columns(self, columns):
        if columns is None:
            self.columns_ = None
            return

        if (
            type(columns) is list and
            all(map(lambda x: isinstance(x, int), columns))
        ):
            self.columns_ = columns
            self.named_columns = False
        elif (
            type(columns) is list and
            all(map(lambda x: isinstance(x, str), columns))
        ):
            self.columns_ = columns
            self.named_columns = True
        elif (
            type(columns) is int
        ):
            self.columns_ = columns
            self.named_columns = False
        elif (
            type(columns) is str
        ):
            self.columns_ = columns
            self.named_columns = True
        else:
            raise ValueError(
                'Columns should be a singe instance or a list of strings or integers'
            )


class PandasTypeSelector(ColumnSelector, PandasTransformer):
    def __init__(self, types):
        super(PandasTypeSelector, self).__init__()
        self.types = types

    def transform_func(self, data):
        if type(self.types) is str:
            types = [self.types]
        return data.select_dtypes(include=types)


class PandasCatColumnSelector(PandasColumnSelector):
    def __init__(self):
        super(PandasCatColumnSelector, self).__init__()

    def transform_func(self, data):
        cat_cols = data.select_dtypes(exclude=[np.number]).columns
        self.columns = [data.columns.get_loc(col) for col in cat_cols]

        return super(PandasCatColumnSelector, self).transform_func(data)


class PandasNonCatColumnSelector(PandasColumnSelector):
    def __init__(self):
        super(PandasNonCatColumnSelector, self).__init__()

    def transform_func(self, data):
        cat_cols = data.select_dtypes(include=[np.number]).columns
        self.columns = [data.columns.get_loc(col) for col in cat_cols]

        return super(PandasNonCatColumnSelector, self).transform_func(data)