from setuptools import setup

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
        'scipy',
        'scikit-learn',
        'numpy',
        'pandas'
    ],
    setup_requires=[
        'nose>=1.0'
    ]
)
