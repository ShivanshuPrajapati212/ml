import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv("heart_processed.csv")

data['Cholesterol'] = data['Cholesterol'].replace(0, np.nan)
data['Cholesterol'] = data['Cholesterol'].fillna(data['Cholesterol'].median())
data['RestingBP'] = data['RestingBP'].replace(0, data['RestingBP'].median())

upper_limit = data['Cholesterol'].quantile(0.95)
data['Cholesterol'] = data['Cholesterol'].clip(upper=upper_limit)

bp_upper = data["RestingBP"].quantile(0.95)
data['RestingBP'] = data['RestingBP'].clip(upper=bp_upper)

X = data.drop("HeartDisease", axis=1).to_numpy()
y = data["HeartDisease"].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=43)

model = RandomForestClassifier(
    n_estimators=1000,       # More trees = more stable predictions
    max_depth=100,            # Prevents the model from 'memorizing' noise
    min_samples_split=5,    # Requires more evidence to make a split
    class_weight='balanced',# Handles any imbalance in your target classes
    random_state=43
    )
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
