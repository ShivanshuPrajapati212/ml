import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

def loss_function(w, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].wind_speed
        y = points.iloc[i].temperature
        total_error +=  (y - (w * x + b)) ** 2
    
    total_error = total_error / len(points)
    
    return  total_error


def gradient_descent(now_w, now_b, points, learning_rate):
    w_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].wind_speed
        y = points.iloc[i].temperature


        w_gradient += -(2/n) * x * (y - (now_w * x + now_b))
        b_gradient += -(2/n) * (y - (now_w * x + now_b))
    
    w = now_w - w_gradient * learning_rate
    b = now_b - b_gradient * learning_rate

    return w, b

def predict(w, b, wind_speed):
    return w*wind_speed + b

w = 0
b = 0
learning_rate = 0.002
epoches = 250 
errors = []

for i in range(epoches):
    if i%50 == 0:
        print("Epoche: ", i)
    if i%100 == 0:
        errors.append((i,loss_function(w, b, df)))
    w, b = gradient_descent(w, b, df, learning_rate)

print("Weight: ", w, "Bias: ",  b)
print("MSE Error: ", loss_function(w, b, df))

plt.scatter(df.wind_speed, df.temperature)
plt.plot(list(range(0, 25)), [w * x + b for x in range(0, 25)], color="black")
plt.show()

x, y = zip(*errors)

plt.plot(x, y)
plt.show()


print("Temperature Pridiction at 12 km/h is:", predict(w, b, 12))
