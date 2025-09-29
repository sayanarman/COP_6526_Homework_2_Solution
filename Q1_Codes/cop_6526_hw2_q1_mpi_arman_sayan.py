# Created: September 21, 2025
# Last Edit Date: September 29, 2025

# This solution is created by Arman Sayan
# as part of the Homework 2 for the COP6526 course.

# How to run:
# Please type the following command to run the script:
# source .venv/bin/activate (if using a virtual environment)
# pip install numpy (if not already installed)
# pip install scikit-learn (if not already installed)
# pip install time (if not already installed)
# pip install mpi4py (if not already installed)
# pip install pandas (if not already installed)
# pip install scipy (if not already installed)
# mpiexec -n <num_processes> python cop_6526_hw2_q1_mpi_arman_sayan.py (<num_processes> should be <= number of CPU cores)

# Question 1.3 : Parallelized Implementation of k-Means Clustering using MPI

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import time
from mpi4py import MPI
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

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

def AssignPointsToCluster(chunk, centroids, rank):
    """
    E-step: Assign each point in the chunk to the nearest centroid.
    """
    print(f"[E-step] Rank {rank} is processing {len(chunk)} samples")

    labels = []
    for x in chunk:
        distances = [EuclideanDistance(x, c) for c in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def ComputePartialSums(chunk, labels, k, n_features, rank):
    """
    M-step: Compute partial sums and counts for each cluster in a chunk.
    """
    print(f"[M-step] Rank {rank} is processing a chunk of size {chunk.shape[0]}")

    sums = np.zeros((k, n_features))
    counts = np.zeros(k, dtype=int)
    for i, x in enumerate(chunk):
        cluster_id = labels[i]
        sums[cluster_id] += x
        counts[cluster_id] += 1
    return sums, counts


def kMeans_Multiprocessing(X_local, comm, rank, size, centroids, k=10, max_iters=100):
    n_local, n_features = X_local.shape

    if rank == 0:
        print("\nStarting parallelized k-Means clustering with multiprocessing...")
        start_time = time.time()

    for it in range(max_iters):
        if rank == 0:
            print(f"\nIteration {it+1}/{max_iters}")

        # Step 1: Assign clusters
        if rank == 0:
            print("E-step: Assigning clusters...")
        local_labels = AssignPointsToCluster(X_local, centroids, rank)

        # Step 2: Update centroids
        if rank == 0:
            print("M-step: Updating centroids...")
        local_sums, local_counts = ComputePartialSums(X_local, local_labels, k, n_features, rank)

        # Reduce across all processes
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)

        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

        # Compute new centroids
        new_centroids = np.array([
            global_sums[i] / global_counts[i] if global_counts[i] > 0 else centroids[i]
            for i in range(k)
        ])

        # Check convergence
        converged = np.allclose(centroids, new_centroids)
        converged = comm.bcast(converged, root=0)

        if converged:
            if rank == 0:
                print(f"Converged after {it+1} iterations.")
            break

        centroids = new_centroids

    if rank == 0:
        end_time = time.time()
        print(f"\nk-Means completed in {end_time - start_time:.2f} seconds.")

    return local_labels, centroids

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load MNIST digits via fetch_openml
    if rank == 0:
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

        # Pick initial centroids as random samples from dataset
        random_indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[random_indices]

        # Split dataset across ranks
        X_chunks = np.array_split(X, size)
    else:
        X_chunks = None
        y = None
        k = None
        centroids = None

    # Broadcast number of clusters and centroids to all ranks
    k = comm.bcast(k, root=0)
    centroids = comm.bcast(centroids, root=0)

    # Scatter data to all ranks
    X_local = comm.scatter(X_chunks, root=0)

    # Run parallel k-Means on local data
    local_labels, local_centroids = kMeans_Multiprocessing(X_local, comm, rank, size, centroids, k)

    # Gather all labels and centroids at root
    all_labels = comm.gather(local_labels, root=0)
    all_centroids = comm.gather(local_centroids, root=0)

    if rank == 0:
        # Concatenate results from all ranks
        labels = np.concatenate(all_labels)
        centroids = np.concatenate(all_centroids)

        print("\nFinal cluster distribution:", np.unique(labels, return_counts=True))

        # Evaluate clustering accuracy
        acc, _ = clustering_accuracy(y, labels, n_clusters=10, n_classes=10)
        print(f"Clustering accuracy: {acc:.4f}")