import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


file_path = "teleCust.csv"
df = pd.read_csv(file_path)

print(df.head())

print("\nMissing values in each column:\n", df.isnull().sum())


df.dropna(inplace=True)

if df.select_dtypes(include=['object']).shape[1] > 0:
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding

# Features (X) and Target (y)
X = df.drop(columns=['custcat'])  # Assuming 'custcat' is the target column
y = df['custcat']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=5)  # Using k=5 neighbors
knn.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn.predict(X_test)

# Compute the confusion matrix
cnf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cnf_matrix_knn, annot=True, cmap="Blues", fmt="g")
plt.title("Confusion Matrix (KNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print model evaluation metrics
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))
print('Accuracy score (KNN):', round(accuracy_score(y_test, y_pred_knn), 2))
print('F1 Score (KNN):', round(f1_score(y_test, y_pred_knn, average='macro'), 2))
