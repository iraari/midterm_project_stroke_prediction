#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

user = {'gender': 'male',
 'work_type': 'govt_job',
 'residence_type': 'urban',
 'smoking_status': 'never_smoked',
 'age': 51.0,
 'hypertension': 1,
 'heart_disease': 0,
 'ever_married': 1,
 'avg_glucose_level': 82.2,
 'bmi': 34.2}

response = requests.post(url, json=user).json()
print(response)

# after 1)gunicorn running 2)midterm_project % python3 predict-test.py
# {'stroke': False, 'stroke_probability': 0.4219938265207051}

# NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, 
# currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. 
# See: https://github.com/urllib3/urllib3/issues/3020