import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv("Mall_Customers.csv")

print("CUSTOMER SEGMENTATION USING K-MEANS")
print("===================================")
print("Total Customers:", len(data))
print("Total Columns:", len(data.columns))

print("\nFIRST 5 RECORDS")
print("================")
print(data.head())

X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

print("\nCLUSTER EVALUATION")
print("==================")

for k in range(2, 11):
    print("K =", k, 
          " WCSS =", round(inertia[k-2], 2),
          " Silhouette =", round(silhouette[k-2], 3))

best_k = range(2, 11)[silhouette.index(max(silhouette))]

print("\nBEST NUMBER OF CLUSTERS:", best_k)
print("BEST SILHOUETTE SCORE:", round(max(silhouette), 3))

model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data["Cluster"] = model.fit_predict(X_scaled)

print("\nCUSTOMERS IN EACH CLUSTER")
print("=========================")

counts = data["Cluster"].value_counts().sort_index()

for cluster, count in counts.items():
    print("Cluster", cluster, ":", count, "customers")

print("\nCLUSTER CHARACTERISTICS")
print("=======================")

summary = data.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

print(summary.round(2))

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

data["PC1"] = pca_data[:, 0]
data["PC2"] = pca_data[:, 1]

print("\nPCA EXPLAINED VARIANCE")
print("======================")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")
print("Total:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 6))

for cluster in sorted(data["Cluster"].unique()):
    group = data[data["Cluster"] == cluster]
    plt.scatter(
        group["PC1"],
        group["PC2"],
        s=60,
        label="Cluster " + str(cluster)
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means and PCA")
plt.legend()
plt.grid(True)
plt.show()
