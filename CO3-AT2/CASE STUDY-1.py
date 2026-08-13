import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    "AnnualIncome": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
    "SpendingScore": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}

df = pd.DataFrame(data)
df.to_csv("customer_segmentation.csv", index=False)
print("Dataset:")
print(df)

X = df[["Age", "AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 7), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

silhouette_scores = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print("K =", k, "Silhouette Score =", round(score, 3))

best_k = range(2, 7)[silhouette_scores.index(max(silhouette_scores))]

print("\nOptimal number of clusters =", best_k)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)
print("\nCustomer Cluster Assignments:")
print(df)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))

plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

for i in range(len(df)):
    plt.annotate(
        df["CustomerID"][i],
        (df["PCA1"][i], df["PCA2"][i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("Customer Segmentation using K-Means and PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

print("\nFinal Customer Segmentation:")
print(
    df[
        [
            "CustomerID",
            "Age",
            "AnnualIncome",
            "SpendingScore",
            "Cluster"
        ]
    ]
)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
