import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LinearRegression

df = pandas.read_csv("data.csv")
wind_speed = df[["wind_speed"]]
temperature = df["temperature"]

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(wind_speed, temperature)

# 3. Make a prediction for a new wind speed (e.g., 22 km/h)
new_wind = np.array([[12]])
predicted_temp = model.predict(new_wind)

print(f"Predicted temperature for 12 km/h wind: {predicted_temp[0]:.2f}°C")

# 4. Visualize the results
plt.scatter(wind_speed, temperature, color='blue', label='Actual Data')
plt.plot(wind_speed, model.predict(wind_speed), color='red', label='Regression Line')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
