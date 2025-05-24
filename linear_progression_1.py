import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 


fuel_data = pd.read_csv("fuel_consumption_dataset.csv")
cars_data = pd.read_csv("used_cars_dataset.csv")
print("Fuel Consumption Dataset:\n", fuel_data.head(), "\n")
print("Used Cars Dataset:\n", cars_data.head(), "\n")


X_fuel = fuel_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_fuel = fuel_data['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_fuel, test_size=0.2, random_state=42)

cars_data_encoded = pd.get_dummies(cars_data, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

X_cars = cars_data_encoded.drop(columns=['name', 'selling_price'])  
y_cars = cars_data_encoded['selling_price']

X_train_cars, X_test_cars, y_train_cars, y_test_cars = train_test_split(X_cars, y_cars, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
model_cars = LinearRegression()
model_cars.fit(X_train_cars, y_train_cars)

y_pred_fuel = model_fuel.predict(X_test_fuel)

y_pred_cars = model_cars.predict(X_test_cars)

r2_fuel = r2_score(y_test_fuel, y_pred_fuel)
print(f"R² Score for CO2 Emissions Prediction: {r2_fuel:.2f}")

r2_cars = r2_score(y_test_cars, y_pred_cars)
print(f"R² Score for Used Car Price Prediction: {r2_cars:.2f}")

plt.figure(figsize=(10,5))
plt.scatter(y_test_fuel, y_pred_fuel, color='blue', alpha=0.5)
plt.plot([y_test_fuel.min(), y_test_fuel.max()], [y_test_fuel.min(), y_test_fuel.max()], color='red', linestyle='dashed')
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(y_test_cars, y_pred_cars, color='green', alpha=0.5)
plt.plot([y_test_cars.min(), y_test_cars.max()], [y_test_cars.min(), y_test_cars.max()], color='red', linestyle='dashed')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Used Car Selling Price")
plt.show()

