import numpy as np
from sklearn.cluster import DBSCAN

# Example: create or load your 3D points as a NumPy array of shape [N, 3]
# For demonstration, here's random data (replace with your actual points)
np.random.seed(0)
points = np.random.rand(10000, 3)  # 10000 points in 3D

# Run DBSCAN clustering
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: minimum number of points required to form a dense region
dbscan = DBSCAN(eps=0.05, min_samples=10)
labels = dbscan.fit_predict(points)

# Count points in each cluster (excluding noise points if needed)
unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
print("Cluster sizes:", dict(zip(unique_labels, counts)))



# Define a threshold for what you consider a "large cluster"
max_cluster_size = 100  # Adjust as needed



# Build a mask to keep only points that are:
# - Marked as noise (label == -1), OR
# - Belonging to clusters smaller than max_cluster_size.
keep_mask = np.zeros_like(labels, dtype=bool)
for label, count in zip(unique_labels, counts):
    if count < max_cluster_size:
        keep_mask[labels == label] = True



# Optionally, include the noise points as well
keep_mask[labels == -1] = True

# Filter the points
filtered_points = points[keep_mask]

print("Original number of points:", points.shape[0])
print("Filtered number of points:", filtered_points.shape[0])
