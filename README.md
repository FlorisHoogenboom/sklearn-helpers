# sklearn-helpers
This repository contains some helpers I use when working with scikit-learn.Currently, it contains 
- Some workarounds for handeling categorical data until https://github.com/scikit-learn/scikit-learn/pull/9151 is finalized.
- Some utility functions I use to incorporate transformations in a Scikit-Learn pipleline.
    - Some transformations classes that can be used as decorators on function performing some data manipulation
    - A column selector class that I'll use untill https://github.com/scikit-learn/scikit-learn/pull/3886 gets merged.