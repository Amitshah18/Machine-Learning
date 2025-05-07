import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    confusion_matrix, roc_curve, auc
)

# Load the dataset
df = pd.read_csv("pima-indians-diabetes.data.csv")

# Assign proper column names
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df.columns = column_names 

# Handle missing values (if any)
df = df.replace('?', np.nan).dropna()
df = df.astype(float)

# Define features (X) and target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store evaluation metrics
metrics = {}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    metrics[name] = {
        "Accuracy": acc, "Recall": recall, "Precision": precision,
        "F1-Score": f1, "Confusion Matrix": conf_matrix
    }

    # Print results
    print(f"Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

# Heatmap of Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, (name, metric) in enumerate(metrics.items()):
    ax = axes[i//2, i%2]
    sns.heatmap(metric["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")

plt.tight_layout()
plt.show()

# ROC Curve Comparison
plt.figure(figsize=(10, 7))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Models")
plt.legend()
plt.show()
