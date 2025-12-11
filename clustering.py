# MODEL 3: CLUSTERING - KMeans Campaign Segmentation


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("marketing_campaign.csv")

# Clean numeric column
df["Acquisition_Cost"] = df["Acquisition_Cost"].replace(r'[\$,]', '', regex=True).astype(float)

# Feature Engineering
df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]

# Select clustering features
cluster_features = df[["CTR", "CPC", "ROI", "Engagement_Score"]]

# Scale data
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_features)

# Use elbow method to check best clusters
wcss = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled)
    wcss.append(km.inertia_)

# Plot elbow
plt.plot(range(2, 10), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Build final model (choose K=3 or 4 based on elbow)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled)

print(df[["CTR", "CPC", "ROI", "Engagement_Score", "Cluster"]].head())








plt.scatter(df["CTR"], df["ROI"], c=df["Cluster"], cmap="viridis")
plt.xlabel("CTR")
plt.ylabel("ROI")
plt.title("K-Means Campaign Clusters")
plt.show()
