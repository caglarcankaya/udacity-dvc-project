import requests
import json
URL = "https://afternoon-dawn-56307.herokuapp.com/predict/"

sample_request = {"age": 27,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Women",
  "capital-gain": 4000,
  "capital-loss": 0,
  "hours-per-week": 36,
  "native-country": "Greece"
}
sample_request_json = json.dumps(sample_request)
print(f"Sending request input {sample_request_json}")
r = requests.post(url = URL, data = sample_request_json)
print(f"Status code = {r.status_code}")
data = r.json()
print(f"Response = {data}")
