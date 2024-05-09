import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")
df.drop(['name'], axis=1, inplace=True)
df.drop(['cabin'], axis=1, inplace=True)
# Display the first few rows of the dataset
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['ticket'] = df['ticket'].str.extract(r'(\d+)$')
df['ticket'] = pd.to_numeric(df['ticket'], errors='coerce')
print(df.head())

# Summary statistics for numerical variables
print(df.describe())

# Visualize the distribution of numerical variables using histograms
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Visualize the distribution of categorical variables using count plots
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df)
plt.title('Distribution of Sex')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='embarked', data=df)
plt.title('Distribution of Embarked')
plt.show()

# Compute the correlation matrix using numeric columns only
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(df.select_dtypes(include=[np.number]).head().to_string())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Handling missing values
print(df.isnull().sum())

# dump the LINE records
df = df[df['ticket'] != 'LINE']
# Impute missing values for numerical variables
df['age'].fillna(df['age'].median(), inplace=True)
print(df.isnull().sum())

# Encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True, dtype=int)
print(df.describe())

# Detect and handle outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='age', data=df)
plt.title('Box Plot of Age')
plt.show()

# Remove outliers using Winsorization
df['age'] = winsorize(df['age'], limits=[0.05, 0.05])

# Check for outliers after Winsorization
plt.figure(figsize=(10, 6))
sns.boxplot(x='age', data=df)
plt.title('Box Plot of Age (After Winsorization)')
plt.show()

