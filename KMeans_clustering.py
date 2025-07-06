import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")  

# Drop CustomerID (not useful for clustering)
df = df.drop("CustomerID", axis=1)

# Convert Gender to numeric
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Elbow Method to find optimal K
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia (WCSS)")
plt.grid(True)
plt.show()

# Fit KMeans with chosen K (e.g., K=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(scaled_data)

# Visualize clusters
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='Set1', edgecolor='k')
plt.title("K-Means Clustering (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Evaluate with Silhouette Score
score = silhouette_score(scaled_data, labels)
print("Silhouette Score:", round(score, 3))
