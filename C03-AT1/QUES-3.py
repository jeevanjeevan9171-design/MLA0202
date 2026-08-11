import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

digits = load_digits()

X = digits.data
y = digits.target

print("DIGITS CLUSTERING USING GMM AND K-MEANS")
print("========================================")

print("Total Samples:", X.shape[0])
print("Total Features:", X.shape[1])
print("Number of Classes:", len(np.unique(y)))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA EXPLAINED VARIANCE")
print("======================")
print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")
print("Total:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(
    n_components=10,
    covariance_type="full",
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

kmeans_ari = adjusted_rand_score(y, kmeans_labels)
gmm_ari = adjusted_rand_score(y, gmm_labels)

print("\nCLUSTERING PERFORMANCE")
print("======================")

print("\nK-MEANS")
print("Silhouette Score:", round(kmeans_silhouette, 4))
print("Adjusted Rand Index:", round(kmeans_ari, 4))

print("\nGAUSSIAN MIXTURE MODEL")
print("Silhouette Score:", round(gmm_silhouette, 4))
print("Adjusted Rand Index:", round(gmm_ari, 4))

print("\nCLUSTER SIZES - K-MEANS")
print("=======================")

for i in range(10):
    print("Cluster", i, ":", np.sum(kmeans_labels == i), "samples")

print("\nCLUSTER SIZES - GMM")
print("===================")

for i in range(10):
    print("Cluster", i, ":", np.sum(gmm_labels == i), "samples")

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="tab10",
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using K-Means and PCA")
plt.colorbar(label="K-Means Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gmm_labels,
    cmap="tab10",
    s=15
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Digits Clustering using GMM and PCA")
plt.colorbar(label="GMM Cluster")
plt.grid(True)
plt.show()

models = ["K-Means", "GMM"]
silhouette_values = [kmeans_silhouette, gmm_silhouette]

plt.figure(figsize=(7, 5))
plt.bar(models, silhouette_values)
plt.ylabel("Silhouette Score")
plt.title("K-Means vs GMM")
plt.ylim(0, max(silhouette_values) + 0.1)
plt.grid(axis="y")
plt.show()
