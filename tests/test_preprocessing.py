import unittest
import numpy as np

from sklearn_helpers.preprocessing import \
    EnhancedLabelEncoder, MultiColumnLabelEncoder


class EnhancedLabelEncoderTest(unittest.TestCase):
    def test_accepts_only_1d(self):
        """It should only accept only a 1d array"""
        ehe = EnhancedLabelEncoder()
        train = np.array([
            [1,2],
            [2,1]
        ])

        self.assertRaises(ValueError, lambda: ehe.fit(train))

        # If it is flattened, it should not raise.
        train = train.flatten()
        ehe.fit(train)


    def test_handle_unknown_error(self):
        """If handle_unkown is 'error' it should throw on unseen labels"""
        ehe = EnhancedLabelEncoder(handle_unknown='error')
        train = np.array(['a', 'b', 'a'])
        test = np.array(['a','c'])

        ehe.fit(train)

        # Check that a ValueError is raised on transform
        self.assertRaises(ValueError, lambda: ehe.transform(test))

    def test_handle_unknown_ignore(self):
        """If handle_unknown is 'ignore' it should map unseen labels to a new value"""
        ehe = EnhancedLabelEncoder(handle_unknown='ignore')
        train = np.array(['a', 'b', 'a'])
        test = np.array(['a','c'])

        ehe.fit(train)

        # Check that the new label is mapped to the next value
        self.assertTrue(
            (np.array([0,2]) == ehe.transform(test)).all()
        )


class MultiColumnLabelEncoderTest(unittest.TestCase):
    def test_handle_ignore(self):
        """If handle_unknown is 'ignore' it should map unseen labels to a new value"""

        mce = MultiColumnLabelEncoder(handle_unknown='ignore')
        train = np.array([
            ['a', 'b'],
            ['c', 'a']
        ])
        test = np.array([
            ['a', 'd'],
            ['c', 'd']
        ])

        mce.fit(train)

        test_transformed = np.array([
            [0.,2.],
            [1.,2.]
        ])

        self.assertTrue(
            (mce.transform(test) == test_transformed).all()
        )
