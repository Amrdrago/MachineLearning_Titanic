import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import pandas as pd

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
    print("Test: {:.2f}%".format(precision_test * 100))
    print("Train: {:.2f}%".format(precision_train * 100), '\n')
    print("Recall")
    print("Test: {:.2f}%".format(recall_test * 100))
    print("Train: {:.2f}%".format(recall_train * 100), '\n')
    print("F1 Score")
    print("Test: {:.2f}%".format(f1_test * 100))
    print("Train: {:.2f}%".format(f1_train * 100))

    return nb_classifier, X_train, y_train

def plot_decision_boundary(X, y, classifier):
    # Reduce features to 2 dimensions
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the training points
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Decision Boundary of Naive Bayes Classifier')
    plt.show()

if __name__ == '__main__':
    # Load the dataframes ------------------------------
    df = pd.read_csv("titanic_preprocessed.csv")
    df_predict = pd.read_csv("titanic_test_preprocessed.csv")
    df_output = pd.read_csv("test.csv")

    # Load the classifier ------------------------------
    naive_bayes_classifier = GaussianNB()

    # Start the training -------------------------------
    classifier, X_train, y_train = train(df, naive_bayes_classifier)

    # Start the testing --------------------------------
    predictions = classifier.predict(df_predict)

    # Merge the predictions with the original file -----
    df_output['survived'] = predictions
    print("\n", df_output.to_string())
    df_output.to_csv('Naive_Bayes_Predictions.csv')