from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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

    print("Test Accuracy:", accuracy_test * 100, "%")
    print("Train Accuracy:", accuracy_train * 100, "%")


def predict(dataframe, classifier):
    nb_classifier = classifier
    predictions = nb_classifier.predict(dataframe)
    return predictions


if __name__ == '__main__':
    # Load the dataframes
    df = pd.read_csv("titanic_preprocessed.csv")
    df_predict = pd.read_csv("titanic_test_preprocessed.csv")
    df_outputted = pd.read_csv("test.csv")
    # Load the classifier
    naive_bayes_classifier = GaussianNB()
    # Start the training
    train(df, naive_bayes_classifier)
    # Start the testing
    predictions = predict(df_predict, naive_bayes_classifier)
    df_outputted['survived'] = predictions
    df_outputted.to_csv('Naive_Bayes_Predictions.csv')
