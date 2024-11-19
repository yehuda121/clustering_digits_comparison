import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

import os
import warnings

# Set the environment variable to avoid memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '3'
# Ignore specific warning related to KMeans memory leak on Windows with MKL
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Select three random digit classes
np.random.seed(42)  # Set seed for reproducibility
chosen_digits = np.random.choice(np.unique(y), 3, replace=False)

# Extract the chosen digits
mask = np.isin(y, chosen_digits)
X = X[mask]
y = y[mask]

# Hide the true labels (for unsupervised learning)
labels_hidden = np.full_like(y, -1)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Reduce dimensionality to two using PCA
pca = PCA(n_components=2)
data_reduced_pca = pca.fit_transform(data_scaled)

# Run K-means with different initial conditions and check the cost function
k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_reduced_pca)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow method to find the optimal k
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose k=3 (according to the graph) and run K-means with different initial conditions
kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans1.fit(data_reduced_pca)

kmeans2 = KMeans(n_clusters=3, random_state=43, n_init=10)
kmeans2.fit(data_reduced_pca)

# Compare the cost function
print("K-means inertia (init 1):", kmeans1.inertia_)
print("K-means inertia (init 2):", kmeans2.inertia_)

# k_values = range(1, 11)
# inertia_values_1 = []
# inertia_values_2 = []

# for k in k_values:
#     kmeans1 = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans1.fit(data_reduced_pca)
#     inertia_values_1.append(kmeans1.inertia_)
    
#     kmeans2 = KMeans(n_clusters=k, random_state=43, n_init=10)
#     kmeans2.fit(data_reduced_pca)
#     inertia_values_2.append(kmeans2.inertia_)

# plt.plot(k_values, inertia_values_1, marker='o', label='Random state 42')
# plt.plot(k_values, inertia_values_2, marker='x', label='Random state 43')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.legend()
# plt.show()

# print("K-means inertia (k=3, init 1):", inertia_values_1[2])  
# print("K-means inertia (k=3, init 2):", inertia_values_2[2])


# Compute the variance of each feature after PCA (two features)
variances = np.var(data_reduced_pca, axis=0)

# Initialize the covariance matrices (Σ) for each component in GMM
initial_covariances = np.array([(variances) for _ in range(3)])

# Run GMM (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42, init_params='kmeans')

# Manually set the covariance matrices to the initial ones we calculated
gmm.covariances_init_ = initial_covariances

# Fit the GMM model and check the log-likelihood after each iteration
log_likelihood = []
for i in range(10):
    gmm.fit(data_reduced_pca)
    log_likelihood.append(gmm.score(data_reduced_pca))

# Plot the log-likelihood over iterations
plt.plot(range(1, 11), log_likelihood, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood over iterations (GMM)')
plt.show()

# Compare results to the true labels using silhouette score and adjusted Rand index
kmeans_labels = kmeans1.labels_
gmm_labels = gmm.predict(data_reduced_pca)

# Calculate and print evaluation metrics for K-means clustering
print("Silhouette Score (K-means):", silhouette_score(data_reduced_pca, kmeans_labels))
print("Adjusted Rand Index (K-means):", adjusted_rand_score(y, kmeans_labels))

# Calculate and print evaluation metrics for GMM clustering
print("Silhouette Score (GMM):", silhouette_score(data_reduced_pca, gmm_labels))
print("Adjusted Rand Index (GMM):", adjusted_rand_score(y, gmm_labels))

# Plot the clustering results for visual comparison between K-means and GMM
plt.figure(figsize=(14, 6))

# Plot the K-means clustering results
plt.subplot(1, 2, 1)
plt.scatter(data_reduced_pca[:, 0], data_reduced_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-means Clustering')

# Plot the GMM clustering results
plt.subplot(1, 2, 2)
plt.scatter(data_reduced_pca[:, 0], data_reduced_pca[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title('GMM Clustering')

plt.show()
