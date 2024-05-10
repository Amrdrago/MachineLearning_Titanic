import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats.mstats import winsorize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

class ANNModel:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.scaler_mm = MinMaxScaler()
        self.model = None

    def preprocess_data(self):
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        X_train_scaled = self.scaler_mm.fit_transform(X_train)
        X_test_scaled = self.scaler_mm.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=len(self.features), activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, epochs=20, batch_size=32):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def evaluate_model(self, X_test, y_test):
        y_pred_test = (self.model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        return accuracy, precision, recall, f1

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def predict(self, df_predict):
        X_predict = df_predict[self.features]
        X_predict_scaled = self.scaler_mm.transform(X_predict)
        predictions = (self.model.predict(X_predict_scaled) > 0.5).astype("int32")
        return predictions
        
# Load the Titanic dataset
df = pd.read_csv("titanic_preprocessed.csv")
features_ann = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex']
target = 'survived'

# Initialize and use the ANNModel class
ann_model = ANNModel(df, features_ann, target)
X_train_ann_scaled, X_test_ann_scaled, y_train_ann, y_test_ann = ann_model.preprocess_data()
ann_model.build_model()
ann_model.train_model(X_train_ann_scaled, y_train_ann)

# Evaluate the model
accuracy, precision, recall, f1 = ann_model.evaluate_model(X_test_ann_scaled, y_test_ann)
print(f"Accuracy on testing set: {accuracy:.4f}")
print(f"Precision on testing set: {precision:.4f}")
print(f"Recall on testing set: {recall:.4f}")
print(f"F1-score on testing set: {f1:.4f}")

ann_model.plot_metrics()

# Prediction on new data
df_predict = pd.read_csv("titanic_test_preprocessed.csv")
df_output = pd.read_csv("test.csv")
predictions = ann_model.predict(df_predict)
df_output['survived'] = predictions
print("\n", df_output.to_string())
df_output.to_csv('ANN_Predictions.csv', index=False)