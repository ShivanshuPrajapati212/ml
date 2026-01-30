import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

## Data Cleaning

data = pd.read_csv("housing.csv")
bedrooms = data['total_bedrooms']
data = data.drop(["total_bedrooms", "ocean_proximity"], axis=1)


data["bedrooms"] = bedrooms
data = data.dropna()


## Loss Function 

def pridict_y(X, w, b):
    return np.dot(X, w)  + b


def loss_function(X, y, w, b): 
    m = len(y) 

    y_hat = pridict_y(X, w, b)

    return np.sum((y_hat-y)**2) / (2 * m)
    

## Gradient Descent 

def gradient_descnet_step(X, y, w, b):
    m = X.shape[0]
    y_hat = pridict_y(X, w, b)
    err = y_hat- y

    dj_dw = (1/m) * np.dot(X.T, err) 
    dj_db = (1/m) * np.sum(err)

    return dj_dw, dj_db

history = [] 

def gradient_descnet(X, y,w , b,epoches=1000, alpha=0.001):

    for i in range(epoches):

        if i % 50 == 0:
            print("Epoche:", i, ", Error:", loss_function(X, y, w, b))
            history.append((i,loss_function(X, y,w, b)))

        dj_dw, dj_db = gradient_descnet_step(X, y, w,b)
        w = w - dj_dw * alpha
        b = b - dj_db * alpha

    return w, b


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = X.to_numpy()
y = y.to_numpy()

w = np.zeros(X.shape[1])
b = 0.0
alpha = 0.1
epoches = 1000


w, b = gradient_descnet(X, y, w, b, epoches, alpha) 

print(w, b)

idx, error = zip(*history) 
for i in range(len(history)):

    print(f"Epoche {idx[i]} : {error[i]}")

plt.plot(idx, error)
plt.show()

y_hat = pridict_y(X, w, b)
data["Model Prediction"] = y_hat

plt.figure(figsize=(10, 6))

# 1. Plot the Predicted points
# We use alpha=0.5 because there are many points; this makes overlaps visible
plt.scatter(y, y_hat, color='#3498db', alpha=0.5, label='Model Predictions')

# 2. Plot the "Perfect Line" (where Actual == Predicted)
# We find the min and max of the actual data to draw a straight diagonal
line_range = [y.min(), y.max()]
plt.plot(line_range, line_range, color='#e74c3c', linewidth=3, label='Perfect Fit (y = y_hat)')

# Formatting
plt.title('Comparison: Actual vs Predicted Housing Values', fontsize=14)
plt.xlabel('Actual Values (y)', fontsize=12)
plt.ylabel('Predicted Values (y_hat)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

print(data)
