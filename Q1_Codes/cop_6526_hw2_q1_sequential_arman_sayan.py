# Created: September 20, 2025
# Last Edit Date: September 21, 2025

# This solution is created by Arman Sayan
# as part of the Homework 2 for the COP6526 course.

# How to run:
# Please type the following command to run the script:
# source .venv/bin/activate (if using a virtual environment)
# pip install numpy (if not already installed)
# pip install sklearn (if not already installed)
# pip install time (if not already installed)
# python3 cop6526_hw2_q1_sequential_arman_sayan.py

# Question 1.1 : Sequential Implementation of k-Means Clustering

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import time


# Load MNIST digits via fetch_openml
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("MNIST dataset loaded.")

X, y = mnist.data, mnist.target.astype(int)

# Subset to first 10,000 samples
print("\nSubsetting to first 10,000 samples...")
X, y = X[:10000], y[:10000]
print("Subsetting complete.")

print("\nDataset shape:", X.shape)  # (10000, 784)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Optionally standardize which helps k-Means performance
X = StandardScaler().fit_transform(X)

# Select number of clusters
k = 10
np.random.seed(42)

# Random initialization of centroids
random_indices = np.random.choice(X.shape[0], k, replace=False)
centroids = X[random_indices]

def EuclideanDistance(a, b):
    return np.linalg.norm(a - b)

def kMeans(X, centroids, max_iters=100):
    print("\nStarting k-Means clustering...")
    start_time = time.time()
    for it in range(max_iters):
        print(f"\nIteration {it+1}/{max_iters}")

        # Step 1: Assign clusters
        print("E-step: Assigning clusters...")
        labels = []
        for x in X:
            distances = [EuclideanDistance(x, c) for c in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        # Step 2: Update centroids
        print("M-step: Updating centroids...")
        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])  # keep old if empty
        new_centroids = np.array(new_centroids)

        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"\nConverged after {it+1} iterations.")
            break

        centroids = new_centroids

    end_time = time.time()
    print(f"\nk-Means completed in {end_time - start_time:.2f} seconds.")

    return labels, centroids

labels, centroids = kMeans(X, centroids)

print("\nFinal cluster distribution:", np.unique(labels, return_counts=True))