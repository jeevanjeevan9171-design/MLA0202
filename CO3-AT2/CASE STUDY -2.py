import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture

data = {
    "Sample": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Alcohol": [14.23, 13.20, 13.16, 14.37, 13.24, 14.20, 14.39, 14.06, 14.83, 13.86],
    "MalicAcid": [1.71, 1.78, 2.36, 1.95, 2.59, 1.76, 1.87, 2.15, 1.64, 1.35],
    "Ash": [2.43, 2.14, 2.67, 2.50, 2.87, 2.45, 2.45, 2.61, 2.17, 2.27],
    "Alcalinity": [15.6, 11.2, 18.6, 16.8, 21.0, 15.2, 14.6, 17.6, 14.0, 16.0],
    "Magnesium": [127, 100, 101, 113, 118, 112, 96, 121, 97, 98],
    "Phenols": [2.80, 2.65, 2.80, 3.85, 2.80, 3.27, 2.50, 2.60, 2.80, 2.98]
}

df = pd.DataFrame(data)

df.to_csv("wine_samples.csv", index=False)

X = df[
    [
        "Alcohol",
        "MalicAcid",
        "Ash",
        "Alcalinity",
        "Magnesium",
        "Phenols"
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

factor = FactorAnalysis(n_components=2, random_state=42)
X_factor = factor.fit_transform(X_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

print("\nPCA Transformed Data:")
print(pd.DataFrame(X_pca, columns=["PCA1", "PCA2"]))

print("\nFactor Analysis Transformed Data:")
print(pd.DataFrame(X_factor, columns=["Factor1", "Factor2"]))

print("\nICA Transformed Data:")
print(pd.DataFrame(X_ica, columns=["ICA1", "ICA2"]))

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nPCA Total Explained Variance:")
print(round(sum(pca.explained_variance_ratio_), 4))

gmm_pca = GaussianMixture(n_components=2, random_state=42)
gmm_factor = GaussianMixture(n_components=2, random_state=42)
gmm_ica = GaussianMixture(n_components=2, random_state=42)

cluster_pca = gmm_pca.fit_predict(X_pca)
cluster_factor = gmm_factor.fit_predict(X_factor)
cluster_ica = gmm_ica.fit_predict(X_ica)

df["PCA_Cluster"] = cluster_pca
df["Factor_Cluster"] = cluster_factor
df["ICA_Cluster"] = cluster_ica

print("\nGMM Clustering Results:")
print(
    df[
        [
            "Sample",
            "PCA_Cluster",
            "Factor_Cluster",
            "ICA_Cluster"
        ]
    ]
)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_pca,
    cmap="viridis",
    s=100
)

for i in range(len(df)):
    plt.annotate(
        df["Sample"][i],
        (X_pca[i, 0], X_pca[i, 1]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("PCA + Gaussian Mixture Model")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_factor[:, 0],
    X_factor[:, 1],
    c=cluster_factor,
    cmap="viridis",
    s=100
)

for i in range(len(df)):
    plt.annotate(
        df["Sample"][i],
        (X_factor[i, 0], X_factor[i, 1]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("Factor Analysis + Gaussian Mixture Model")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    X_ica[:, 0],
    X_ica[:, 1],
    c=cluster_ica,
    cmap="viridis",
    s=100
)

for i in range(len(df)):
    plt.annotate(
        df["Sample"][i],
        (X_ica[i, 0], X_ica[i, 1]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("ICA + Gaussian Mixture Model")
plt.xlabel("Independent Component 1")
plt.ylabel("Independent Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

comparison = pd.DataFrame({
    "Sample": df["Sample"],
    "PCA_Cluster": cluster_pca,
    "Factor_Cluster": cluster_factor,
    "ICA_Cluster": cluster_ica
})

print("\nFinal Comparison:")
print(comparison)
