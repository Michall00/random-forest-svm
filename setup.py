from setuptools import setup, find_packages

setup(
    name='random_forest_svm',
    version='0.1',
    packages=find_packages(include=['random_forest_svm', 'random_forest_svm.*'])
)