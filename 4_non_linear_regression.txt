import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load data
china_gdp = pd.read_csv("china_gdp.csv")
print(china_gdp.head())

# Define sigmoid function
def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Prepare data
X = china_gdp["Year"].values
y = china_gdp["Value"].values

# Fit the curve
popt, _ = curve_fit(sigmoid, X, y, p0=[max(y), 0.1, 2000])
L_opt, k_opt, x0_opt = popt
print(f"Optimised Parameters: L = {L_opt:.4f}, k = {k_opt:.4f}, x0 = {x0_opt:.4f}")

# Predict values
x_range = np.linspace(min(X), max(X), 100)
y_pred = sigmoid(x_range, L_opt, k_opt, x0_opt)

# Plot results
plt.scatter(X, y, label="Actual GDP Data", color="blue")
plt.plot(x_range, y_pred, label="Sigmoid Fit", color="red")
plt.xlabel("Year")
plt.ylabel("GDP (Trillion USD)")
plt.title("China GDP Prediction (1960-2014)")
plt.legend()
plt.show()

# Calculate R2 Score
y_pred_final = sigmoid(X, L_opt, k_opt, x0_opt)
r2 = r2_score(y, y_pred_final)
print(f"R2 Score: {r2:.4f}")
