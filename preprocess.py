import numpy as np
from scipy.stats._mstats_basic import winsorize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(input, output):
    df = pd.read_csv("titanic.csv")
    df.drop(['name', 'cabin', 'ticket', 'embarked'], axis=1, inplace=True)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    # Handling missing values
    print(df.isnull().sum())

    # Impute missing values for numerical variables
    df['age'].fillna(df['age'].median(), inplace=True)
    df.dropna(subset=['fare'], inplace=True)
    # Encode categorical variables

    # Remove outliers using Winsorization
    df['age'] = winsorize(df['age'], limits=[0.05, 0.05])
    df.dropna(subset=['age'], inplace=True)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Compute the correlation matrix using numeric columns only
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()

    print(df.isnull().sum())

    df.to_csv(output, index=False)


preprocess("titanic.csv", "titanic_preprocessed.csv")
preprocess("test.csv", "titanic_test_preprocessed.csv")
