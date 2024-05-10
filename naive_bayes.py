from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


def train(dataframe, classifier) :
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


def prediction(dataframe, classifier) :
    nb_classifier = classifier
    predicts = nb_classifier.predict(dataframe)
    return predicts


if __name__ == '__main__' :
    # Load the dataframes ------------------------------
    df = pd.read_csv("titanic_preprocessed.csv")
    df_predict = pd.read_csv("titanic_test_preprocessed.csv")
    df_output = pd.read_csv("test.csv")

    # Load the classifier ------------------------------
    naive_bayes_classifier = GaussianNB()

    # Start the training -------------------------------
    train(df, naive_bayes_classifier)

    # Start the testing --------------------------------
    predictions = prediction(df_predict, naive_bayes_classifier)

    # Merge the predictions with the original file -----
    df_output['survived'] = predictions
    print("\n", df_output.to_string())
    df_output.to_csv('Naive_Bayes_Predictions.csv')
