# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and Load Dataset
2. Select Features for Clustering
3. Train the K-Means Model
4. Predict and Visualize Clusters

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HARISHBALA J
RegisterNumber:  212224223002
*/
```
```
# K-Means Clustering for Customer Segmentation

# Import required libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Display dataset
print("Dataset Preview:")
print(data.head())

# Select features for clustering
# Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Finding the optimal number of clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Predict cluster labels
y_kmeans = kmeans.fit_predict(X)

# Add cluster column to dataset
data['Cluster'] = y_kmeans

# Display clustered data
print("\nClustered Data:")
print(data.head())

# Visualize Clusters
plt.figure(figsize=(8,6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans
)

# Plot cluster centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
```

## Output:
<img width="935" height="664" alt="image" src="https://github.com/user-attachments/assets/6f3f504e-a50c-45c6-986c-36e3d61d0754" />
<img width="992" height="899" alt="image" src="https://github.com/user-attachments/assets/f345e6b6-e2f3-4aa5-9a2b-18b23c163793" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
