from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 1. Load data
data = pd.read_csv("data.csv")

data = data.drop(["Name", "Ticket", "Cabin", "PassengerId", "SibSp", "Parch"], axis=1)

data["Sex"] = data["Sex"].map({'male': 0, 'female': 1})
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q":2})

data = data.dropna()

X = data.drop("Survived", axis=1).to_numpy()
y = data["Survived"].to_numpy()

# Logistic Regression works best when features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Instantiate and Fit
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)


print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
