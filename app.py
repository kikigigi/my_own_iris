from flask import Flask, request
import numpy as np
import pickle
#from sklearn.datasets import load_iris

app = Flask(__name__)

with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

labels = ['setosa', 'versicolor', 'virginica']

#pred = model.predict([[1,2,3,4]])[0]
#print('prediction is {}'.format(labels[pred]))



features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

@app.route('/hi')
def hi():
    return "Hi"

@app.route('/api_predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    test_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(test_data)[0]
    return str(labels[pred])

if __name__ == "__main__":
    app.run()
