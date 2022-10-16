import pytest
import pandas as pd
import os
import sys

print(os.getcwd())
sys.path.append('../starter/ml')
from starter.ml.data import process_data
path = os.getcwd()

@pytest.fixture
def data():
    df = pd.read_csv(path+ '/../data/clean_census.csv')
    return df

@pytest.fixture
def cat_features():
    return  [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

def test_cat_features(data, cat_features):
    """ Checks the categorical features are categorical. """
    print(data.columns)
    assert set(cat_features).issubset(set(data.columns))

def test_process_data(data,cat_features):
    """Test if results of the process_data function are not none"""
    X, y, encoder, lb = process_data(data, cat_features, label = 'salary')
    assert X is not None
    assert y is not None
    assert encoder is not None
    assert lb is not None


def test_cat_features_valid(data, cat_features):
    """ Control if there is a numeric feature among categorical ones """
    df = data.drop(columns = ['salary'])
    columns = df.columns
    numeric_columns = set(df._get_numeric_data().columns)
    categorical_columns = (set(columns) - set(numeric_columns))
    assert set(cat_features) == categorical_columns

def test_size_for_training(data):
    """ Checks the size of the dataset that will be used for training"""
    assert 1000 < data.shape[0]
