from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app, CensusRequest, CensusBatchRequest

# Instantiate the testing client with our app.
client = TestClient(app)

def test_api_get():
    response = client.get("/")
    assert response.status_code == 200
    response = response.json()
    assert response["initial_message"] == "Welcome to the Census Salary Prediction Service built by Çağlar Çankaya"


def test_api_post():
    response = client.post("/predict/", data=json.dumps(CensusRequest.Config.schema_extra['example']))
    assert response.status_code == 200
    response = response.json()
    assert response['salary'] == '<=50K'


def test_api_post_example_input():
    response = client.post("/predict_batch/", data=json.dumps({
        "people": [CensusRequest.Config.schema_extra['example'], CensusRequest.Config.schema_extra['example2']]}))

    assert response.status_code == 200
    response = response.json()
    assert response['salary'] == ['<=50K', '>50K']
