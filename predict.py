import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_model = 'model_LR.pkl'
input_dv = 'dv.pkl'

with open(input_model, 'rb') as m_in: 
    model = pickle.load(m_in)

with open(input_dv, 'rb') as dv_in: 
    dv = pickle.load(dv_in)

app = Flask('stroke')

@app.route('/predict', methods=['POST'])
def predict():
    user = request.get_json()

    X = dv.transform([user])
    y_pred = model.predict_proba(X)[0, 1]
    stroke = y_pred >= 0.5

    result = {
        'stroke_probability': float(y_pred),
        'stroke': bool(stroke)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

# 1) midterm_project % gunicorn --bind 0.0.0.0:9696 predict:app
# 2) midterm_project % python3 predict-test.py
