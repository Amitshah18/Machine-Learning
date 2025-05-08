import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# 1. Load Data
data = pd.read_csv(r"Cust_Segmentation.csv")

# 2. Preprocess Data
# Drop Customer Id and Address (non-informative for clustering)
data_clean = data.drop(['Customer Id', 'Address'], axis=1)

# Optional: remove 'Defaulted' if you consider it target
data_clean = data_clean.drop(['Defaulted'], axis=1)

# Scale the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# 3. KMeans with Euclidean distance
k = 4  # or choose based on elbow method
kmeans_euclidean = KMeans(n_clusters=k, random_state=42)
kmeans_euclidean.fit(data_scaled)
labels_euclidean = kmeans_euclidean.labels_

# 4. KMeans with Manhattan distance manually
# Custom KMeans because sklearn uses only Euclidean
# Initialize centers
centers = data_scaled[np.random.choice(range(len(data_scaled)), k, replace=False)]

for i in range(100):  # number of iterations
    distances = cdist(data_scaled, centers, 'cityblock')  # Manhattan (L1)
    labels_manhattan = np.argmin(distances, axis=1)
    
    # Update centers
    new_centers = np.array([data_scaled[labels_manhattan == j].mean(axis=0) for j in range(k)])
    
    if np.all(centers == new_centers):
        break
    centers = new_centers

# 5. Visualization 2D and 3D

# Feature indices for plotting (choose 3 for 3D visualization)
feature1 = 3  # Example: 'Income'
feature2 = 6  # Example: 'DebtIncomeRatio'
feature3 = 2  # Example: 'Age'

# 2D Plotting
plt.figure(figsize=(12, 5))

# Euclidean 2D Plot
plt.subplot(1, 2, 1)
plt.scatter(data_scaled[:, feature1], data_scaled[:, feature2], c=labels_euclidean, cmap='rainbow')
plt.title('KMeans Clustering (Euclidean Distance)')
plt.xlabel('Income')
plt.ylabel('DebtIncomeRatio')

# Manhattan 2D Plot
plt.subplot(1, 2, 2)
plt.scatter(data_scaled[:, feature1], data_scaled[:, feature2], c=labels_manhattan, cmap='rainbow')
plt.title('KMeans Clustering (Manhattan Distance)')
plt.xlabel('Income')
plt.ylabel('DebtIncomeRatio')

plt.tight_layout()
plt.show()

# 3D Plotting
fig = plt.figure(figsize=(12, 6))

# Euclidean 3D Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data_scaled[:, feature1], data_scaled[:, feature2], data_scaled[:, feature3], c=labels_euclidean, cmap='rainbow')
ax1.set_title('KMeans Clustering (Euclidean Distance)')
ax1.set_xlabel('Income')
ax1.set_ylabel('DebtIncomeRatio')
ax1.set_zlabel('Age')

# Manhattan 3D Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(data_scaled[:, feature1], data_scaled[:, feature2], data_scaled[:, feature3], c=labels_manhattan, cmap='rainbow')
ax2.set_title('KMeans Clustering (Manhattan Distance)')
ax2.set_xlabel('Income')
ax2.set_ylabel('DebtIncomeRatio')
ax2.set_zlabel('Age')

plt.tight_layout()
plt.show()
