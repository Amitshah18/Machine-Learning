import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
file_path = "pima-indians-diabetes.data.csv"
df = pd.read_csv(file_path)

# Assign proper column names
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df.columns = column_names

# Split data into features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naïve Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# Train Bernoulli Naïve Bayes model (Thresholding required for continuous features)
bnb = BernoulliNB()
bnb.fit(X_train > 0, y_train)  # Convert features to binary (greater than zero)
y_pred_bnb = bnb.predict(X_test > 0)

# Calculate accuracy and F1-score for both models
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb)

accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
f1_bnb = f1_score(y_test, y_pred_bnb)

# Print results
print("Gaussian Naïve Bayes:")
print(f"Accuracy: {accuracy_gnb:.4f}")
print(f"F1-Score: {f1_gnb:.4f}\n")

print("Bernoulli Naïve Bayes:")
print(f"Accuracy: {accuracy_bnb:.4f}")
print(f"F1-Score: {f1_bnb:.4f}")
