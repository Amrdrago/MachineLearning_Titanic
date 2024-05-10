import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the dataset
df = pd.read_csv("titanic_preprocessed.csv")

# Step 4: Split the data into features and target variables
X = df.drop(['survived'], axis=1)
y = df['survived']

# Step 5: Train an SVM classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Step 6: Evaluate the model
train_accuracy = accuracy_score(y_train, svm.predict(X_train))
test_accuracy = accuracy_score(y_test, svm.predict(X_test))
train_f1 = f1_score(y_train, svm.predict(X_train))
test_f1 = f1_score(y_test, svm.predict(X_test))
train_precision = precision_score(y_train, svm.predict(X_train))
test_precision = precision_score(y_test, svm.predict(X_test))
train_recall = recall_score(y_train, svm.predict(X_train))
test_recall = recall_score(y_test, svm.predict(X_test))

print("Training Accuracy:", train_accuracy,'\n')
print("Test Accuracy:", test_accuracy,'\n')
print("Training Precision Score:", train_precision,'\n')
print("Test Precision Score:", test_precision,'\n')
print("Training Recall Score:", train_recall,'\n')
print("Test Recall Score:", test_recall,'\n')
print("Training F1 Score:", train_f1,'\n')
print("Test F1 Score:", test_f1,'\n')

# Define the parameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' and 'poly'
}

# Create the GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params,'\n')
print("Best Score:", best_score,'\n')

# Use the best estimator to make predictions
best_svm = grid_search.best_estimator_
test_accuracy_tuned = accuracy_score(y_test, best_svm.predict(X_test))
test_f1_tuned = f1_score(y_test, best_svm.predict(X_test))
test_precision_tuned = precision_score(y_test, best_svm.predict(X_test))
test_recall_tuned = recall_score(y_test, best_svm.predict(X_test))

print("Test Accuracy (Tuned Model):", test_accuracy_tuned,'\n')
print("Test Precision Score (Tuned Model):", test_precision_tuned,'\n')
print("Test Recall Score (Tuned Model):", test_recall_tuned,'\n')
print("Test F1 Score (Tuned Model):", test_f1_tuned,'\n')

df_test = pd.read_csv("titanic_test_preprocessed.csv")

# Step 4: Use the trained model to make predictions
predictions = best_svm.predict(df_test)

df_output = pd.read_csv("test.csv")
df_output["survived"] = predictions
df_output.to_csv("SVM_Predictions.csv")
print(df_output.to_string())

# Select features "fare" and "pclass" and target variable "survived"
X = df[['fare', 'pclass']]
y = df['survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate the model
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

print("\nTraining Accuracy:", train_accuracy,'\n')
print("Test Accuracy:", test_accuracy,'\n')

# Create a mesh grid for plotting the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X['fare'].min() - 1, X['fare'].max() + 1
y_min, y_max = X['pclass'].min() - 1, X['pclass'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot the decision boundary
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X['fare'], X['pclass'], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Fare')
plt.ylabel('Pclass')
plt.title('SVM Decision Boundary with Fare and Pclass')
plt.show()