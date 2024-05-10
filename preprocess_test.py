import numpy as np
from scipy.stats._mstats_basic import winsorize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_test = pd.read_csv("test.csv")

df_test.drop(['name', 'cabin', 'ticket'], axis=1, inplace=True)
df_test['sex'] = df_test['sex'].map({'male': 0, 'female': 1})



# Impute missing values for numerical variables
df_test['age'].fillna(df_test['age'].median(), inplace=True)

# Encode categorical variables
df_test = pd.get_dummies(df_test, columns=['embarked'], drop_first=True)

# Remove outliers using Winsorization
df_test['age'] = winsorize(df_test['age'], limits=[0.05, 0.05])
df_test.dropna(subset=['age'], inplace=True)
scaler = MinMaxScaler()
df_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)

# Handling missing values
print(df_test.isnull().sum())

# Compute the correlation matrix using numeric columns only
correlation_matrix = df_test.select_dtypes(include=[np.number]).corr()
df_test.to_csv("titanic_preprocessed_test.csv", index=False)
