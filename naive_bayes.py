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
X = df.drop(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked_Q', 'embarked_S'], axis=1)  # Features
y = df['survived']  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

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
# TRAIN
df_test = pd.read_csv("test.csv")

df_test.drop(['name', 'cabin', 'ticket'], axis=1, inplace=True)
df_test['sex'] = df_test['sex'].map({'male': 0, 'female': 1})

# Handling missing values
print(df_test.isnull().sum())

# Impute missing values for numerical variables
df_test['age'].fillna(df_test['age'].median(), inplace=True)

# Encode categorical variables
df_test = pd.get_dummies(df_test, columns=['embarked'], drop_first=True)

# Remove outliers using Winsorization
df_test['age'] = winsorize(df_test['age'], limits=[0.05, 0.05])
df_test.dropna(subset=['age'], inplace=True)
scaler = MinMaxScaler()
df_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

# Compute the correlation matrix using numeric columns only
correlation_matrix = df_test.select_dtypes(include=[np.number]).corr()
df_test.to_csv("titanic_preprocessed_test.csv", index=False)

df_predict = pd.read_csv("titanic_preprocessed_test.csv")
# Add placeholder values for the 'survived' column

# Reorder columns to match the order during training
df_predict = df_predict[X.columns]
# Make predictions using the trained Naive Bayes classifier
predictions = nb_classifier.predict(df_predict)

# Print the predictions
print(predictions)
