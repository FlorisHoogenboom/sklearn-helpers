# sklearn-helpers
This repository contains some helpers I use when working with scikit-learn.Currently, it contains 
- ~~Some workarounds for handeling categorical data until https://github.com/scikit-learn/scikit-learn/pull/9151 is finalized.~~ [9151](https://github.com/scikit-learn/scikit-learn/pull/9151) has been merged and is set to be released in version 0.20. From then on sklearn.preprocessing.CategoricalEncoder should be used since it is more feature complete.
- Some utility functions I use to incorporate transformations in a Scikit-Learn pipleline.
    - Some transformations classes that can be used as decorators on function performing some data manipulation
    - ~~A column selector class that I'll use untill [3886](https://github.com/scikit-learn/scikit-learn/pull/3886) gets merged.~~ [3886](https://github.com/scikit-learn/scikit-learn/pull/3886) was closed in favor of [9012](https://github.com/scikit-learn/scikit-learn/pull/9012). However the ColumnTransformer there focusses on a slightly different target than the classes in this project.
