import json

import requests

# send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000/")

# print the status code
print("Status Code:", r.status_code)
# print the welcome message
print("Results:", r.json().get("message"))

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

# send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data)

# print the status code and result
print("Status Code:", r.status_code)
try:
    print("Result:", r.json().get("result"))
except Exception:
    print("Raw Body:", r.text)
