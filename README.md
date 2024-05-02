# Predicting_Vehicle_Loan_Defaulters
The aim is to analyze data related to vehicle loans to determine factors affecting the ratio of loan defaulters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing and understanding the data
data = pd.read_csv('vehicle_loan_data.csv')  # Assuming the data file is named vehicle_loan_data.csv
print(data.head())  # Inspecting the first few rows of the dataset
print(data.info())  # Understanding the data types and missing values

# Performing Exploratory Data Analysis (EDA)

# Modeling
# Splitting the data into train and test sets
X = data.drop('loan_status', axis=1)  # Features
y = data['loan_status']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training - Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = logistic_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Model training - Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(conf_matrix_rf)

classification_rep_rf = classification_report(y_test, y_pred_rf)
print("Random Forest Classification Report:")
print(classification_rep_rf)

# Libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn
