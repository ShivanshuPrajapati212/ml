import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

## Data Cleaning
data = pd.read_csv("data.csv")

data = data.drop(["Name", "Ticket", "Cabin", "PassengerId", "Age", "SibSp", "Parch"], axis=1)

data["Sex"] = data["Sex"].map({'male': 0, 'female': 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q":2})

data = data.dropna()
print(data)

## Predict Y 
def pridict(X, w, b):
    return 1 / (1 + np.exp(-(np.dot(X, w) + b)))

## Cost Function
def cost_function(X, y, w, b):
   m = X.shape[0]
   f_wb = pridict(X, w, b)

   cost = (-1/m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))

   return cost



## Usage / Testing
X = data.drop("Survived", axis=1).to_numpy()
y = data["Survived"].to_numpy()

w = np.zeros(X.shape[1])
b = 0 

print(cost_function(X, y, w, b))
