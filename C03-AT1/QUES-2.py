import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

wine = load_wine()

X = wine.data
y = wine.target

print("DIMENSIONALITY REDUCTION")
print("========================")

print("Total Samples:", X.shape[0])
print("Original Features:", X.shape[1])
print("Number of Classes:", len(np.unique(y)))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

print("\nPCA ANALYSIS")
print("============")
print("PC1 Variance:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2 Variance:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")
print("Total Variance:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

print("\nFA OUTPUT")
print("========")
print("Original Dimensions:", X_scaled.shape[1])
print("Reduced Dimensions:", X_fa.shape[1])

print("\nICA OUTPUT")
print("=========")
print("Original Dimensions:", X_scaled.shape[1])
print("Reduced Dimensions:", X_ica.shape[1])

print("\nCOMPONENT SHAPES")
print("================")
print("PCA:", X_pca.shape)
print("FA :", X_fa.shape)
print("ICA:", X_ica.shape)

plt.figure(figsize=(8, 6))

for c in np.unique(y):
    plt.scatter(
        X_pca[y == c, 0],
        X_pca[y == c, 1],
        s=50,
        label="Class " + str(c)
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset - PCA")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

for c in np.unique(y):
    plt.scatter(
        X_fa[y == c, 0],
        X_fa[y == c, 1],
        s=50,
        label="Class " + str(c)
    )

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Wine Dataset - Factor Analysis")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

for c in np.unique(y):
    plt.scatter(
        X_ica[y == c, 0],
        X_ica[y == c, 1],
        s=50,
        label="Class " + str(c)
    )

plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.title("Wine Dataset - ICA")
plt.legend()
plt.grid(True)
plt.show()

methods = ["PCA", "FA", "ICA"]
dimensions = [X_pca.shape[1], X_fa.shape[1], X_ica.shape[1]]

plt.figure(figsize=(7, 5))
plt.bar(methods, dimensions)
plt.xlabel("Dimensionality Reduction Method")
plt.ylabel("Number of Components")
plt.title("Comparison of Dimensionality Reduction Methods")
plt.grid(axis="y")
plt.show()
