import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    jaccard_score, confusion_matrix, roc_curve, auc
)

# Load the dataset
df = pd.read_csv("samples_cancer.csv")

# Drop unnecessary ID column
df.drop(columns=["ID"], inplace=True)

# Handle missing values (if any)
df = df.replace('?', np.nan).dropna()
df = df.astype(float)

# Define features (X) and target (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# Convert target: 2 (benign) -> 0, 4 (malignant) -> 1
y = y.map({2: 0, 4: 1})

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM kernels
kernels = ["linear", "poly", "rbf", "sigmoid"]
models = {}

# Dictionary to store evaluation metrics
metrics = {}

# Train and evaluate models
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    model = SVC(kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store model
    models[kernel] = model
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred)
    error_rate = 1 - acc
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    metrics[kernel] = {
        "Accuracy": acc, "Recall": recall, "Precision": precision,
        "F1-Score": f1, "Jaccard Score": jaccard, "Error Rate": error_rate,
        "Confusion Matrix": conf_matrix
    }

    # Print results
    print(f"Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}, Error Rate: {error_rate:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

# ROC Curve Comparison
plt.figure(figsize=(10, 7))

for kernel in kernels:
    model = models[kernel]
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{kernel} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of SVM Kernels")
plt.legend()
plt.show()
