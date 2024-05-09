import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Handle missing values (as done previously)

# Encode categorical variables (Sex, Embarked)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Feature selection (choose relevant features for each algorithm)
features_knn = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
features_nb = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
features_svm = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
features_ann = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']

# Define X (features) and y (target)
X_knn = df[features_knn]
X_nb = df[features_nb]
X_svm = df[features_svm]
X_ann = df[features_ann]
y = df['Survived']

# Split data into training and testing sets (80% train, 20% test)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y, test_size=0.2, random_state=42)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y, test_size=0.2, random_state=42)
X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X_ann, y, test_size=0.2, random_state=42)

# Scale numerical features (for KNN and SVM)
scaler = StandardScaler()
X_train_knn_scaled = scaler.fit_transform(X_train_knn)
X_test_knn_scaled = scaler.transform(X_test_knn)

X_train_svm_scaled = scaler.fit_transform(X_train_svm)
X_test_svm_scaled = scaler.transform(X_test_svm)

# For ANN, use Min-Max scaling (values between 0 and 1)
scaler_mm = MinMaxScaler()
X_train_ann_scaled = scaler_mm.fit_transform(X_train_ann)
X_test_ann_scaled = scaler_mm.transform(X_test_ann)

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv("train.csv")

# Handle missing values
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Fare'] = imputer.fit_transform(df[['Fare']])

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to train and evaluate KNN with different k values and distance metrics
def train_eval_knn(X_train, y_train, X_test, y_test, k_values, distance_metrics):
    results = {'k': [], 'distance_metric': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    for k in k_values:
        for metric in distance_metrics:
            # Create KNN classifier with current k and distance metric
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

            # Train the model
            knn.fit(X_train, y_train)

            # Make predictions
            y_pred = knn.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Save results
            results['k'].append(k)
            results['distance_metric'].append(metric)
            results['accuracy'].append(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)

    return results

# Define k values and distance metrics to experiment with
k_values = [3, 5, 7, 9, 11]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

# Train and evaluate KNN using defined function
results = train_eval_knn(X_train_scaled, y_train, X_test_scaled, y_test, k_values, distance_metrics)

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print results for analysis
print(results_df)

# Visualize performance metrics for different k values and distance metrics
plt.figure(figsize=(10, 6))
for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
    plt.figure(figsize=(8, 5))
    for metric_value in distance_metrics:
        plt.plot(results_df[results_df['distance_metric'] == metric_value]['k'],
                 results_df[results_df['distance_metric'] == metric_value][metric],
                 marker='o', label=f'{metric_value}')

    plt.xlabel('k value')
    plt.ylabel(metric.capitalize())
    plt.title(f'KNN {metric.capitalize()} with Different Distance Metrics')
    plt.legend(title='Distance Metric')
    plt.grid(True)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()

# Select the best k value based on the highest F1-score
best_k = results_df.loc[results_df['f1_score'].idxmax(), 'k']
best_metric = results_df.loc[results_df['f1_score'].idxmax(), 'distance_metric']

# Train KNN with the best k value and distance metric on the full training set
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
best_knn.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred_test = best_knn.predict(X_test_scaled)

# Evaluate the model on the testing set
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

# Print evaluation metrics on the testing set
print(f"Best k value: {best_k}")
print(f"Best distance metric: {best_metric}")
print(f"Accuracy on testing set: {accuracy_test:.4f}")
print(f"Precision on testing set: {precision_test:.4f}")
print(f"Recall on testing set: {recall_test:.4f}")
print(f"F1-score on testing set: {f1_test:.4f}")
