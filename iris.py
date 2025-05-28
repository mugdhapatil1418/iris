# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("iris.csv")

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Summary statistics
print("\nStatistical Summary:")
print(df.describe())

# Data info
print("\nDataset Info:")
print(df.info())

# Visualizations
sns.pairplot(df, hue="Species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='viridis')
plt.title("Feature Correlation Heatmap")
plt.show()

# Box plots by species
plt.figure(figsize=(12, 6))
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Species', y=col, data=df)
    plt.title(f'{col} by Species')
plt.tight_layout()
plt.show()

# Encode the target variable
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])  # Setosa=0, Versicolour=1, Virginica=2

# Split features and labels
X = df.drop('Species', axis=1)
y = df['Species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

# Train, Predict and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

