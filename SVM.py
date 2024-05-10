import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC

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
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)



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

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best estimator to make predictions
best_svm = grid_search.best_estimator_
test_accuracy_tuned = best_svm.score(X_test, y_test)
print("Test Accuracy (Tuned Model):", test_accuracy_tuned)

df_test = pd.read_csv("titanic_test_preprocessed.csv")

# Step 4: Use the trained model to make predictions
predictions = best_svm.predict(df_test)

df_output = pd.read_csv("test.csv")
df_output["survived"] = predictions
df_output.to_csv("SVM_Predictions.csv")
print(df_output.to_string())