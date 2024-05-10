import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the preprocessed dataset
data = pd.read_csv("titanic_preprocessed.csv")

# Splitting the data into features (X) and target variable (y)
X = data.drop('survived', axis=1)
y = data['survived']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of k values to test
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17]

# Create an empty list to store accuracy scores for each k value
accuracy_scores = []

# Train and evaluate the models with varying k values
for k in k_values:
    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plotting the accuracy scores for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Convert accuracy_scores to a NumPy array for convenience
accuracy_scores_np = np.array(accuracy_scores)

# Find the index of the highest accuracy score
best_k_index = accuracy_scores_np.argmax()

best_k = k_values[best_k_index]

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Make predictions using the best K value model
y_pred_train = knn_best.predict(X_train)
y_pred_test = knn_best.predict(X_test)

# Calculate accuracy using the best K value model
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Calculate precision
precision_test = precision_score(y_test, y_pred_test)
precision_train = precision_score(y_train, y_pred_train)

# Calculate recall
recall_test = recall_score(y_test, y_pred_test)
recall_train = recall_score(y_train, y_pred_train)
# Calculate F1 score
f1_test = f1_score(y_test, y_pred_test)
f1_train = f1_score(y_train, y_pred_train)

print("---------------------------------------------------------")
print("Best K value based on highest accuracy score:", best_k)
print("---------------------------------------------------------")
print("Accuracy")
print("Test: {:.2f}%".format(accuracy_test * 100))
print("Train: {:.2f}%".format(accuracy_train * 100))
print("---------------------------------------------------------")
print("Precision")
print("Test:", precision_test)
print("Train:", precision_train)
print("---------------------------------------------------------")
print("Recall")
print("Test:", recall_test)
print("Train:", recall_train)
print("---------------------------------------------------------")
print("F1 Score")
print("Test:", f1_test)
print("Train:", f1_train)

########################################################################################################################
########################################################################################################################
print("===============================================================================================================")

# Load the testing dataset
test_data = pd.read_csv("titanic_test_preprocessed.csv")

# Make predictions on the testing set
y_pred_final = knn_best.predict(test_data)

prediction_csv = pd.read_csv("test.csv")

# Add the predicted 'survived' column to the testing dataset
prediction_csv['survived'] = y_pred_final

# Save the updated testing dataset with predictions to a new CSV file
prediction_csv.to_csv("KNN_Predictions.csv", index=False)

########################################################################################################################

# Plotting the predicted survival status for the testing dataset
survival_counts = pd.Series(y_pred_final).value_counts()
labels = ['Deceased', 'Survived']
plt.figure(figsize=(6, 6))
plt.pie(survival_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Predicted Mortality Rate')
plt.show()