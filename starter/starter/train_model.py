# Script to train machine learning model.

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle
#load in the data.
path = os.path.abspath(os.getcwd())
data = pd.read_csv(path+"/../data/clean_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
x_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, encoder=encoder,lb=lb,label="salary", training=False
)
# Train and save a model.
model = train_model(X_train, y_train)
#Save both encoder and the model weights
pickle.dump(model, open(path + "/../model/random_forest_classifier.pkl", 'wb'))
pickle.dump(encoder, open(path + "/../model/encoder_random_forest.pkl", 'wb'))
pickle.dump(lb, open(path + "/../model/lb.pkl", 'wb'))
#Test model scores on test data
predict = inference(model, x_test)
precision, recall, f1 = compute_model_metrics(y_test, predict)

print(f"Precision on test data is {precision}")
print(f"Recall on test data is {recall}")
print(f"F1 score on test data is {f1}")
