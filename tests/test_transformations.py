import unittest
import numpy as np
import pandas as pd

from sklearn_helpers.transformations import \
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

    def test_accepts_dataframe_with_missing(self):
        """The PandasTransformer should accept a DataFrame with mising of np.inf values"""
        df = pd.DataFrame(
            [
                [np.inf, np.nan],
                [1,2]
            ],
            columns=['a', 'b']
        )

        @PandasTransformer
        def trans(data):
            return data.dropna()

        self.assertTrue(
            trans.transform(df).shape[0] == 1
        )


class ColumnSelectorTest(unittest.TestCase):
    def test_selects_correct_columns(self):
        """The column selector should only return the selected columns"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[0,1])

        result = cs.transform(data)

        self.assertTrue(
            (data == result).all()
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

    def test_selects_vector(self):
        """If only  a single index is passed it should return a vector"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=0)

        result = cs.transform(data)

        self.assertTrue(
            (data[:,0] == result).all()
        )

    def test_handle_unkown_error(self):
        """It should raise and error if an unkown column is passed"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[1,2], handle_unknown='error')

        self.assertRaises(
            IndexError,
            lambda: cs.transform(data)
        )

    def test_handle_unkown_ignore(self):
        """It should ignore unknown columns"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[1,3], handle_unknown='ignore')

        result = cs.transform(data)

        self.assertTrue(
            (data[:,1] == result[:,0]).all()
        )

    def test_selects_complement(self):
        """The column selector should only return the selected columns"""
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[1], complement=True)

        result = cs.transform(data)

        self.assertTrue(
            (data[:,0] == result[:,0]).all()
        )

    def test_returns_empty_when_desired(self):
        """
        When an emtpy set of columns is specified it should return no columns
        """
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[])

        result = cs.transform(data)

        self.assertEqual(
            result.shape[1],
            0
        )

    def test_returns_full_when_selecting_empty_complement(self):
        """
        It should return the full df when selecting the emtpy complement.
        """
        data = np.array([
            [1,2],
            [3,4]
        ])

        cs = ColumnSelector(columns=[], complement=True)

        result = cs.transform(data)

        self.assertEqual(
            result.shape[1],
            2
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

    def test_select_series_numeric(self):
        """It should return a series if a single value is passed"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=0)

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,0] == selected).all()
        )

    def test_select_series_string(self):
        """It should return a series if a single value is passed"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns='b')

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,1] == selected).all()
        )

    def test_handle_unkown_error(self):
        """It should throw an error if an unknown column is passed"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=['b', 'd'], handle_unknown='error')

        self.assertRaises(
            IndexError,
            lambda: pcs.transform(df)
        )

    def test_handle_unkown_ignore(self):
        """It should ignore unknown columns"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=['b', 'd'], handle_unknown='ignore')

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,1] == selected.iloc[:,0]).all()
        )

        self.assertTrue(
            selected.shape[1] == 1
        )

    def test_handle_complement(self):
        """If complement is set to true it should select the complement columns"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=['b', 'c'], complement=True)

        selected = pcs.transform(df)

        self.assertTrue(
            (df.iloc[:,0] == selected.iloc[:,0]).all()
        )

        self.assertTrue(
            selected.shape[1] == 1
        )

    def test_selects_empty(self):
        """When an empty list of columns is passed, it should return DF without columns"""
        df = pd.DataFrame(
            np.array([
                [1,2,4],
                [3,4,5]
            ]),
            columns=['a', 'b', 'c']
        )

        pcs = PandasColumnSelector(columns=[])

        selected = pcs.transform(df)

        self.assertEqual(
            selected.shape[1],
            0
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