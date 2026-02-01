import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv("heart_processed.csv")

X = data.drop("HeartDisease", axis=1).to_numpy()
y = data["HeartDisease"].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=43)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

new_data = pd.DataFrame({"Real": y_test, "Prediction": predictions})

print(new_data)
