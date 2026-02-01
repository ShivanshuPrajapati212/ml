import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

## Data Cleaning
data = pd.read_csv("data.csv")

data = data.drop(["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"], axis=1)

data["Sex"] = data["Sex"].map({'male': 0, 'female': 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q":2})

data = data.dropna()
print(data)

## Predict Y 
def predict(X, w, b):
    return 1 / (1 + np.exp(-(np.dot(X, w) + b)))

## Cost Function
def cost_function(X, y, w, b):
   m = X.shape[0]
   f_wb = predict(X, w, b)

   cost = (-1/m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))

   return cost

## Gradient Descent 
def gradient_descent_step(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    err  = y_hat - y

    dj_dw = (1/m) * np.dot(X.T, err)
    dj_db = (1/m) * np.sum(err)

    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, epoches):
    for i in range(epoches):    
        dj_dw, dj_db = gradient_descent_step(X, y, w, b)

        w = w - dj_dw * alpha
        b = b - dj_db * alpha

        if i % 50 == 0:
            print("Epoche", i, ":", cost_function(X, y, w, b))
 
    return w, b

## Score Check
def get_score(y, y_hat):
    res = y == y_hat
    return (np.sum(res) * 100) / y.shape[0]

## Usage / Testing
X = data.drop("Survived", axis=1).to_numpy()
y = data["Survived"].to_numpy()

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

w = np.zeros(X.shape[1])
b = 0 

w, b = gradient_descent(X, y, w, b, 0.01, 2000)

y_hat = predict(X, w, b)

y_hat = (y_hat >= 0.5).astype(int)

data["Prediction"] = y_hat

print(data)

print("Score:", get_score(y, y_hat), "%")
