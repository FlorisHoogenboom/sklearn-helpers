import unittest
import numpy as np
import pandas as pd

from sklearn_helpers.transformations import Transformer, PandasTransformer


class TransformerTest(unittest.TestCase):
    # TODO: implement  these
    pass


class PandasTransformerTest(unittest.TestCase):
    def test_applies_function(self):
        """Test whether the transformer function is applied correctly"""
        @PandasTransformer
        def trans(data):
            data['a'] = 0
            return data

        df = pd.DataFrame(
            np.ones((10, 2)),
            columns=['a', 'b']
        )

        df = trans.transform(df)

        self.assertTrue(
            (df['a'] == 0).all()
        )

        self.assertTrue(
            (df['b'] == 1).all()
        )

    def test_accepts_only_pandas(self):
        """A PandasTransformer should only accept a pandas dataframe"""
        @PandasTransformer
        def trans(data):
            return data

        data = np.ones((10, 2))

        self.assertRaises(ValueError, lambda: trans.transform(data))

