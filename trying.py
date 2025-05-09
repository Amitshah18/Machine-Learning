import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, 30, 45, 35, 20],
    'Salary': [50000, 60000, 80000, 70000, 40000],
    'Buys_Computer': [0,0,0,0,0]
})
print(data.head())
# Convert categorical data
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])  # Male=1, Female=0
print(data.head())
# Features and label
X = data[['Gender', 'Age', 'Salary']]
y = data['Buys_Computer']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(data.head())
print("Predictions:", y_pred)
print("Accuracy Score:", accuracy)
