from pyexpat import model
import pandas as pd
from ml.data import *
from ml.model import *
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

path = os.path.abspath(os.getcwd())
data = pd.read_csv(path + "/../data/clean_census.csv")
train, test = train_test_split(data, test_size=0.2, random_state=42)
encoder = pickle.load(open(path + "/../model/encoder_random_forest.pkl", 'rb'))
model = pickle.load(open(path + "/../model/random_forest_classifier.pkl", 'rb'))
lb = pickle.load(open(path + "/../model/lb.pkl", 'rb'))
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
#Extract calculate performance for values given in feature and category
def test_on_slice(test, category, feature):
    test_slice = test[test[category] == feature]
    X_test, y_test, _, _ = process_data(test_slice, categorical_features=categorical_features,
                                        label="salary", encoder=encoder, lb = lb, training=False)
    y_pred = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    print(f"precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f} on category {category} and feature {feature}")

#Extract all features and show performance for all features
def partial_categories(test, category):
    features = test[category].unique()

    for feature in features:
        test_on_slice(test, category, feature)

#Calculate performances for per slice on categorical features
def slice_summary(test):

    for category in categorical_features:
        partial_categories(test, category)


if __name__ == "__main__":
    slice_summary(test)
