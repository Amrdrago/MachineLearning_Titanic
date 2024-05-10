import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed dataset
data = pd.read_csv("titanic_preprocessed.csv")

# Splitting the data into features (X) and target variable (y)
X = data.drop('survived', axis=1)
y = data['survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=5 (you can experiment with different values of k)
knn = KNeighborsClassifier(n_neighbors=17)

# Train the KNN model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

########################################################################################################################
########################################################################################################################
print("===============================================================================================================")

# Load the testing dataset
test_data = pd.read_csv("titanic_test_preprocessed.csv")

# Make predictions on the testing set
y_pred_final = knn.predict(test_data)

prediction_csv = pd.read_csv("test.csv")

# Add the predicted 'survived' column to the testing dataset
prediction_csv['survived'] = y_pred_final

# Save the updated testing dataset with predictions to a new CSV file
prediction_csv.to_csv("KNN_Predictions.csv", index=False)

# Display the first few rows of the updated testing dataset
# accuracy_test = accuracy_score(test_data, y_pred_final)
# print("Accuracy on Testing Dataset:", accuracy_test)
print(prediction_csv.to_string())