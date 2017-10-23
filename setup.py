from distutils.core import setup

setup(
    name='sklearn_helpers',
    version='0.1',
    packages=['sklearn_helpers'],
    url='https://github.com/FlorisHoogenboom/sklearn-helpers',
    license='MIT',
    author='Floris Hoogenboom',
    author_email='floris.hoogenboom@futurefacts.nl',
    description='Some helpers I use when working with scikit-learn',
    install_requires=[
        'scikit-learn',
        'numpy'
    ],
    tests_require=[
        'nose',
        'pandas'
    ]
)
