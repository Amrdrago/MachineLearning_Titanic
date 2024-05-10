import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Step 2: Load the dataset
df = pd.read_csv("titanic_preprocessed.csv")
df['fare'].fillna(df['fare'].median(), inplace=True)
print(df.isnull().sum())

# Step 4: Split the data into features and target variables
X = df[['pclass', 'fare']]
y = df['survived']

# Step 5: Train an SVM classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the values for C and gamma to loop through
C_values = [0.1, 1, 10, 100]
gamma_values = [0.1, 0.01, 0.001, 0.0001]

best_accuracy = 0
best_C = None
best_gamma = None

# Loop through each combination of C and gamma
for C in C_values:
    for gamma in gamma_values:
        # Train SVM classifier with current C and gamma
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        svm.fit(X_train, y_train)

        # Evaluate accuracy on test set
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Update best parameters if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_gamma = gamma

# Train SVM classifier with best parameters
best_svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
best_svm.fit(X_train, y_train)

# Step 6: Visualize the hyperplane and support vectors
# For 2D visualization, select two features
feature1 = 'pclass'  # Replace 'pclass' with the actual name of the feature
feature2 = 'fare'  # Replace 'fare' with the actual name of the feature

# Plot decision boundary and support vectors
plt.figure(figsize=(10, 6))

# Plot decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('SVM Decision Boundary')
plt.show()

# Calculate additional metrics
y_pred_test = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print("Best accuracy:", best_accuracy)
print("Best C:", best_C)
print("Best gamma:", best_gamma)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


