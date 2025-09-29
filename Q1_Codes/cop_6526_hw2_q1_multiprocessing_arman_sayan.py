# Created: September 21, 2025
# Last Edit Date: September 29, 2025

# This solution is created by Arman Sayan
# as part of the Homework 2 for the COP6526 course.

# How to run:
# Please type the following command to run the script:
# source ./venv/bin/activate (if using a virtual environment)
# pip install numpy (if not already installed)
# pip install sklearn (if not already installed)
# pip install time (if not already installed)
# pip install multiprocessing (if not already installed)
# python3 cop_6526_hw2_q1_multiprocessing_arman_sayan.py -n <num_processes> (<num_processes> should be <= number of CPU cores)

# Question 1.2 : Parallelized Implementation of k-Means Clustering using Python Multiprocessing

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import time
from multiprocessing import Pool, cpu_count, current_process
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import argparse

def clustering_accuracy(y_true, y_pred, n_clusters=10, n_classes=10):
    """
    Compute clustering accuracy by finding the best cluster-to-label mapping
    using the Hungarian algorithm.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Cluster assignments from k-means.
    n_clusters : int, optional (default=10)
        Number of clusters used in k-means.
    n_classes : int, optional (default=10)
        Number of unique ground-truth classes.

    Returns
    -------
    acc : float
        Clustering accuracy in [0, 1].
    mapping : dict
        Dictionary mapping cluster_id -> assigned class label.
    """

    # Build confusion matrix (clusters × classes)
    conf_matrix = np.zeros((n_clusters, n_classes), dtype=int)
    for cluster_id, true_label in zip(y_pred, y_true):
        conf_matrix[cluster_id, true_label] += 1

    # Hungarian algorithm (maximize matching)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # Map clusters to labels
    cluster_to_label = dict(zip(row_ind, col_ind))

    # Translate predictions
    mapped_preds = np.array([cluster_to_label[cluster] for cluster in y_pred])

    # Accuracy
    acc = accuracy_score(y_true, mapped_preds)

    return acc, cluster_to_label

# Helper functions for parallel k-Means
def EuclideanDistance(a, b):
    return np.linalg.norm(a - b)

def AssignPointsToCluster(chunk, centroids):
    """
    E-step: Assign each point in the chunk to the nearest centroid.
    """
    proc = current_process()
    print(f"[E-step] Worker {proc.name} (PID={proc.pid}) is processing {len(chunk)} samples")

    labels = []
    for x in chunk:
        distances = [EuclideanDistance(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def ComputePartialSums(args):
    """
    M-step: Compute partial sums and counts for each cluster in a chunk.
    """
    chunk, labels, k = args

    proc = current_process()
    print(f"[M-step] Worker {proc.name} (PID={proc.pid}) is processing a chunk of size {chunk.shape[0]}")

    sums = np.zeros((k, chunk.shape[1]))
    counts = np.zeros(k, dtype=int)
    for i, x in enumerate(chunk):
        cluster_id = labels[i]
        sums[cluster_id] += x
        counts[cluster_id] += 1
    return sums, counts


def kMeans_Multiprocessing(X, num_processes, k=10, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = X.shape

    # Random initialization of centroids
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    num_workers = num_processes
    pool = Pool(processes=num_workers)

    print("\nStarting parallelized k-Means clustering with multiprocessing...")
    start_time = time.time()
    for it in range(max_iters):
        print(f"\nIteration {it+1}/{max_iters}")

        # Step 1: Assign clusters
        print("E-step: Assigning clusters...")
        chunks = np.array_split(X, num_workers)
        labels_chunks = pool.starmap(AssignPointsToCluster, [(chunk, centroids) for chunk in chunks])
        labels = np.concatenate(labels_chunks)

        # Step 2: Update centroids
        print("M-step: Updating centroids...")
        partial_results = pool.map(ComputePartialSums, [(chunks[i], labels_chunks[i], k) for i in range(num_workers)])

        # Aggregate sums and counts across processes
        total_sums = np.zeros((k, n_features))
        total_counts = np.zeros(k, dtype=int)

        for sums, counts in partial_results:
            total_sums += sums
            total_counts += counts

        # Compute new centroids
        new_centroids = np.array([
            total_sums[i] / total_counts[i] if total_counts[i] > 0 else centroids[i]
            for i in range(k)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"\nConverged after {it+1} iterations.")
            break

        centroids = new_centroids

    pool.close()
    pool.join()
    end_time = time.time()
    print(f"\nk-Means completed in {end_time - start_time:.2f} seconds.")

    return labels, centroids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num-processes",
        type=int,
        default= cpu_count(),
        help="Number of processes to run (default: cpu_count())"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    print(f"Using {num_processes} processes for parallel computation.")


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

    labels, centroids = kMeans_Multiprocessing(X, num_processes, k)

    print("\nFinal cluster distribution:", np.unique(labels, return_counts=True))

    # Evaluate clustering accuracy
    acc, _ = clustering_accuracy(y, labels, n_clusters=10, n_classes=10)
    print(f"Clustering accuracy: {acc:.4f}")