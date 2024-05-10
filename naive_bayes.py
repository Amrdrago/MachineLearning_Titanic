import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB


def train(dataframe, classifier):
    # Split data into features and labels
    X = dataframe.drop(['survived'], axis=1)  # Features
    y = dataframe['survived']  # Labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)

    # Initialize Gaussian Naive Bayes classifier
    nb_classifier = classifier

    # Train the classifier
    nb_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred_test = nb_classifier.predict(X_test)
    y_pred_train = nb_classifier.predict(X_train)

    # Calculate accuracy
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

    print("Accuracy")
    print("Test: {:.2f}%".format(accuracy_test * 100))
    print("Train: {:.2f}%".format(accuracy_train * 100), '\n')
    print("Precision")
    print("Test:", precision_test)
    print("Train:", precision_train, '\n')
    print("Recall")
    print("Test:", recall_test)
    print("Train:", recall_train, '\n')
    print("F1 Score")
    print("Test:", f1_test)
    print("Train:", f1_train)
    plot_decision_boundary(nb_classifier, X_train, y_train)


def plot_decision_boundary(classifier, X, y):
    # Define ranges to plot decision boundary
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict class labels for each point in the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(
        xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary for Naive Bayes')
    plt.show()


def prediction(dataframe, classifier):
    nb_classifier = classifier
    predicts = nb_classifier.predict(dataframe)
    return predicts


if __name__ == '__main__':
    # Load the dataframes ------------------------------
    df = pd.read_csv("titanic_preprocessed.csv")
    df_predict = pd.read_csv("titanic_test_preprocessed.csv")
    df_output = pd.read_csv("test.csv")
    # Load the classifier ------------------------------
    naive_bayes_classifier = GaussianNB()
    # Start the training -------------------------------
    train(df, naive_bayes_classifier)
    # Start the Predicting -----------------------------
    predictions = prediction(df_predict, naive_bayes_classifier)
    # Merge the predictions with the original file -----
    df_output['survived'] = predictions
    print("\n", df_output.to_string())
    df_output.to_csv('Naive_Bayes_Predictions.csv')
