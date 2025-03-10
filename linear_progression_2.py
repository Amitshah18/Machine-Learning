import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

house_data = pd.read_csv("C:/Users/as313/OneDrive/Desktop/Machine Learning -College/housing_price_dataset.csv")
print(house_data.head())

house_data = house_data.drop(columns=['Address'])
X = house_data.drop(columns=['Price'])
y = house_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
ridge_reg = Ridge(alpha=1.0)  # Default alpha = 1.0
ridge_reg.fit(X_train_scaled, y_train)
lasso_reg = Lasso(alpha=1.0)  # Default alpha = 1.0
lasso_reg.fit(X_train_scaled, y_train)

y_pred_lin = lin_reg.predict(X_test_scaled)
y_pred_ridge = ridge_reg.predict(X_test_scaled)
y_pred_lasso = lasso_reg.predict(X_test_scaled)

r2_lin = r2_score(y_test, y_pred_lin)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"R² Score for Linear Regression: {r2_lin:.4f}")
print(f"R² Score for Ridge Regression: {r2_ridge:.4f}")
print(f"R² Score for Lasso Regression: {r2_lasso:.4f}")

from sklearn.model_selection import GridSearchCV
alpha_values = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_search = GridSearchCV(Ridge(), param_grid=alpha_values, cv=5)
ridge_search.fit(X_train_scaled, y_train)
lasso_search = GridSearchCV(Lasso(), param_grid=alpha_values, cv=5)
lasso_search.fit(X_train_scaled, y_train)
print(f"Best alpha for Ridge: {ridge_search.best_params_}")
print(f"Best alpha for Lasso: {lasso_search.best_params_}")

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_lin, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dashed')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices (Linear Regression)")
plt.show()

