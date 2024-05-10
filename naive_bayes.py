import numpy as np
from scipy.stats._mstats_basic import winsorize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("titanic_preprocessed.csv")
df.dropna(subset=["fare"], inplace=True)
# Assuming you have a DataFrame 'df' with features and labels
# Replace 'df' with your actual DataFrame name

# Split data into features and labels
X = df.drop(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'], axis=1)  # Features
y = df['survived']  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123,stratify=y)

# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_test = nb_classifier.predict(X_test)
y_pred_train = nb_classifier.predict(X_train)

# Calculate accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)

print("Test Accuracy:", accuracy_test * 100, "%")
print("Train Accuracy:", accuracy_train * 100, "%")

df_predict = pd.read_csv("titanic_preprocessed_test.csv")
# Add placeholder values for the 'survived' column

# Make predictions using the trained Naive Bayes classifier
predictions = nb_classifier.predict(df_predict)
df_predict['survived'] = predictions
# Print the predictions
print(df_predict.to_string())
