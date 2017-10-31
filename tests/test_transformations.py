import unittest
import numpy as np
import pandas as pd

from sklearn_helpers.transformations import \
    Transformer, \
    PandasTransformer, \
    ColumnSelector, \
    PandasColumnSelector, \
    PandasCatColumnSelector, \
    PandasNonCatColumnSelector


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


class ColumnSelectorTest(unittest.TestCase):
    def test_selects_correct_columns(self):
        """The column selector should only return the selected columns"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[1])

        result = cs.transform(data)

        self.assertTrue(
            (data[:,1] == result[:,0]).all()
        )

    def test_selects_all(self):
        """It should keep all columns if no columns argument is passed"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector()

        result = cs.transform(data)

        self.assertTrue(
            (data == result).all()
        )


class PandasColumnSelectorTest(unittest.TestCase):
    def test_selects_right_columns_numeric(self):
        """It should select the right columns when specified as indices"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=[1,2])

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,1] == selected.iloc[:,0]).all()
        )

        self.assertTrue(
            (df.iloc[:,2] == selected.iloc[:,1]).all()
        )

    def test_selects_right_columns_str(self):
        """It should select the right columns when specified as a string"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=['b', 'c'])

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,1] == selected.iloc[:,0]).all()
        )

        self.assertTrue(
            (df.iloc[:,2] == selected.iloc[:,1]).all()
        )


class PandasCatColumnsSelectorTest(unittest.TestCase):
    def test_select_right_columns(self):
        """It should select only categorical columns"""
        df = pd.DataFrame(
            np.array([
                [1,'b','a'],
                [3,'c','b']
            ]),
            columns=['a', 'b', 'c']
        )

        df['a'] = df['a'].astype('int64')

        ccs = PandasCatColumnSelector()
        selected = ccs.transform(df)

        self.assertTrue(selected.shape[1] == 2)

        self.assertTrue(
            all(selected == df.iloc[:,[1,2]])
        )


class PandasNonCatColumnsSelectorTest(unittest.TestCase):
    def test_select_right_columns(self):
        """It should select only categorical columns"""
        df = pd.DataFrame(
            np.array([
                [1,'b','a'],
                [3,'c','b']
            ]),
            columns=['a', 'b', 'c']
        )

        df['a'] = df['a'].astype('int64')

        nccs = PandasNonCatColumnSelector()
        selected = nccs.transform(df)

        self.assertTrue(selected.shape[1] == 1)

        self.assertTrue(
            all(selected == df.iloc[:,[0]])
        )