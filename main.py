import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")
df.drop(['name', 'cabin'], axis=1, inplace=True)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['ticket'] = df['ticket'].str.extract(r'(\d+)$')
df['ticket'] = pd.to_numeric(df['ticket'], errors='coerce')

# Handling missing values
print(df.isnull().sum())

# Impute missing values for numerical variables
df['age'].fillna(df['age'].median(), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Remove outliers using Winsorization
df['age'] = winsorize(df['age'], limits=[0.05, 0.05])

# Compute the correlation matrix using numeric columns only
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

# Visualize the distribution of numerical variables using histograms
def show_histograms():
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

# Visualize the distribution of categorical variables using count plots
def show_sex_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sex', data=df)
    plt.title('Distribution of Sex')
    plt.show()

def show_embarked_distribution():
    embarked_cols = [col for col in df.columns if 'embarked' in col]
    embarked_data = df[embarked_cols].sum().reset_index()
    embarked_data.columns = ['embarked', 'count']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='embarked', y='count', data=embarked_data)
    plt.title('Distribution of Embarked')
    plt.show()

# Visualize the correlation matrix
def show_correlation_matrix():
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

# Detect and handle outliers
def show_age_boxplot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age', data=df)
    plt.title('Box Plot of Age')
    plt.show()

# Check for outliers after Winsorization
def show_age_boxplot_winsorized():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age', data=df)
    plt.title('Box Plot of Age (After Winsorization)')
    plt.show()

# Print DataFrame head
def print_df_head():
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, df.head().to_string())

# Print summary statistics
def print_summary_statistics():
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, df.describe().to_string())

# Print correlation matrix
def print_correlation_matrix():
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, correlation_matrix.to_string())

# Now, the dataset is ready for further analysis and modeling

# Create the main window
root = tk.Tk()
root.title("Machine Learning - Titanic Data Analysis")

# Make the window full screen
root.state('zoomed')

# Add a title label
title_label = ttk.Label(root, text="Titanic Data Analysis", font=("Times", 16))
title_label.pack(pady=10)

# Function to set the width of the buttons
def set_button_width(buttons, width):
    for btn in buttons:
        btn.config(width=width)

# Create and place buttons
buttons = []
btn_histograms = ttk.Button(root, text="Show Histograms", command=show_histograms)
buttons.append(btn_histograms)
btn_histograms.pack(pady=5)

btn_sex_distribution = ttk.Button(root, text="Show Sex Distribution", command=show_sex_distribution)
buttons.append(btn_sex_distribution)
btn_sex_distribution.pack(pady=5)

btn_embarked_distribution = ttk.Button(root, text="Show Embarked Distribution", command=show_embarked_distribution)
buttons.append(btn_embarked_distribution)
btn_embarked_distribution.pack(pady=5)

btn_correlation_matrix = ttk.Button(root, text="Show Correlation Matrix", command=show_correlation_matrix)
buttons.append(btn_correlation_matrix)
btn_correlation_matrix.pack(pady=5)

btn_age_boxplot = ttk.Button(root, text="Show Age Box Plot", command=show_age_boxplot)
buttons.append(btn_age_boxplot)
btn_age_boxplot.pack(pady=5)

btn_age_boxplot_winsorized = ttk.Button(root, text="Show Age Box Plot (Winsorized)", command=show_age_boxplot_winsorized)
buttons.append(btn_age_boxplot_winsorized)
btn_age_boxplot_winsorized.pack(pady=5)

btn_print_df_head = ttk.Button(root, text="Print DataFrame Head", command=print_df_head)
buttons.append(btn_print_df_head)
btn_print_df_head.pack(pady=5)

btn_summary_statistics = ttk.Button(root, text="Print Summary Statistics", command=print_summary_statistics)
buttons.append(btn_summary_statistics)
btn_summary_statistics.pack(pady=5)

btn_print_correlation_matrix = ttk.Button(root, text="Print Correlation Matrix", command=print_correlation_matrix)
buttons.append(btn_print_correlation_matrix)
btn_print_correlation_matrix.pack(pady=5)

# Set the width of all buttons to be the same
set_button_width(buttons, 30)

# Create a Text widget for displaying DataFrame information
output_text = scrolledtext.ScrolledText(root, width=100, height=20, font=("Times", 12))
output_text.pack(pady=10)

# Add a label for team members at the bottom right
team_label_text = """Amr Khaled Mohamed Hassan - 2206159
Loay Salah Abdel Azeem - 2206155
Omar Momen Ahmed - 2206157
Mickel Wassef Riad - 22010449
Abdelrahman Jayasundara Malavi - 2206147
Adham Mohamed Elsayed - 2206132"""

team_label = ttk.Label(root, text=team_label_text, font=("Times", 10), anchor='e', justify='right')
team_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

# Start the GUI event loop
root.mainloop()
