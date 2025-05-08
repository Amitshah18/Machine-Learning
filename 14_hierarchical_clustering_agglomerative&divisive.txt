import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Step 1: Load data
data = pd.read_csv(r"cars_clus.csv")

# Step 2: Preprocess

# Drop non-numeric columns
data_clean = data.drop(['manufact', 'model', 'type', 'partition'], axis=1)

# Replace '$null$' with np.nan
data_clean.replace('$null$', np.nan, inplace=True)

# Convert all columns to numeric (some may still be object because of '$null$')
data_clean = data_clean.apply(pd.to_numeric)

# Drop rows with missing values
data_clean = data_clean.dropna()

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clean)

# Step 3: Agglomerative Clustering and Dendrograms

linkage_methods = ['single', 'complete', 'average']

plt.figure(figsize=(20, 6))

for i, method in enumerate(linkage_methods):
    plt.subplot(1, 3, i + 1)
    linked = linkage(data_scaled, method=method)
    dendrogram(linked,
               orientation='top',
               distance_sort='ascending',
               show_leaf_counts=True)
    plt.title(f'Agglomerative Clustering\n({method.capitalize()} Linkage)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()
