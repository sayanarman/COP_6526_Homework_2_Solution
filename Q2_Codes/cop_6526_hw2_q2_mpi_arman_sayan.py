# Created: September 22, 2025
# Last Edit Date: September 22, 2025

# This solution is created by Arman Sayan
# as part of the Homework 2 for the COP6526 course.

# How to run:
# Please type the following command to run the script:
# source .venv/bin/activate (if using a virtual environment)
# pip install numpy (if not already installed)
# pip install pandas (if not already installed)
# pip install matplotlib (if not already installed)
# pip install time (if not already installed)
# pip install mpi4py (if not already installed)
# mpiexec -n <num_processes> python cop_6526_hw2_q2_mpi_arman_sayan.py --debug (<num_processes> should be <= number of CPU cores) (to see detailed logs of each process)
# mpiexec -n <num_processes> python cop_6526_hw2_q2_mpi_arman_sayan.py (<num_processes> should be <= number of CPU cores) (to run normally without debug logs)

# Question 2.2 : Parallelized Implementation of Nonlinear Regression using MPI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import logging
import argparse

logger = logging.getLogger("train")   # name your logger (e.g. "train")
handler = logging.StreamHandler()     # log to stdout
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Helper functions for parallel Nonlinear Regression
def PrepareCountsDispls(n_total, size):
    """
    Create element counts and displacements for Scatterv.
    """
    base = n_total // size
    rem = n_total % size
    counts = np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32)
    displs = np.zeros(size, dtype=np.int32)
    if size > 1:
        displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Set logging level only for *your* logger
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the model parameters
    L = 0.0001  # The learning Rate
    epochs = 10000  # The number of iterations to perform gradient descent

    # Preprocessing Input data
    if rank == 0:
        data = pd.read_csv('20K_Datapoints.csv')
        X_full = data.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
        Y_full = data.iloc[:, 1].to_numpy(dtype=np.float64, copy=False)
        n_total = X_full.shape[0]
        plt.scatter(X_full, Y_full)
        plt.show()
    else:
        X_full = None
        Y_full = None
        n_total = None

    # Broadcast total length so all ranks can prepare local buffers
    n_total = comm.bcast(n_total, root=0)

    # Prepare counts/displacements on root, 
    # then broadcast counts so each rank knows its recv size
    if rank == 0:
        print(f"Using {size} processes for parallel computation.")
        counts, displs = PrepareCountsDispls(n_total, size)
        print(f"\nData has been split into {size} chunks.")
    else:
        counts = np.empty(size, dtype=np.int32)
        displs = np.empty(size, dtype=np.int32)

    comm.Bcast(counts, root=0)
    comm.Bcast(displs, root=0)
    n_local = int(counts[rank])

    # Allocate local buffers
    X_local = np.empty(n_local, dtype=np.float64)
    Y_local = np.empty(n_local, dtype=np.float64)

    # Scatter the data to all ranks
    comm.Scatterv([X_full, counts, displs, MPI.DOUBLE], X_local, root=0)
    comm.Scatterv([Y_full, counts, displs, MPI.DOUBLE], Y_local, root=0)

    # Set initial model parameters
    a, b, c = 0.0, 0.0, 0.0
    params = np.array([a, b, c], dtype=np.float64)

    # Start training
    if rank == 0:
        print("\nStarting parallel Nonlinear Regression...")
        start_time = time.time()
    for epoch in range(epochs):
        # Broadcast current model parameters to all ranks
        comm.Bcast(params, root=0)
        a, b, c = params

        # Each rank computes local predictions and residuals
        logger.debug(f"[Grad Comp] Epoch {epoch} Rank {rank} is processing {n_local} samples")
        Y_pred_local = a * X_local * X_local + b * X_local + c
        res_local = (Y_local - Y_pred_local)  # residuals

        # Each rank computes local gradients (un-normalized sums)
        S_a_local = np.sum(X_local * X_local * res_local)
        S_b_local = np.sum(X_local * res_local)
        S_c_local = np.sum(res_local)
        #n_local_f  = float(n_local)

        # Reduce local gradients to root for sum across all ranks
        S_totals = np.array([S_a_local, S_b_local, S_c_local, len(X_local)], dtype=np.float64)
        S_global = np.zeros_like(S_totals)
        comm.Reduce(S_totals, S_global, op=MPI.SUM, root=0)

        # Update model parameters at root
        if rank == 0:
            S_a_tot, S_b_tot, S_c_tot, N = S_global
            logger.debug(f"Epoch {epoch}: Collected gradients from all processes.")

            # Average gradients to calculate overall derivative
            D_a = (-2.0 / N) * S_a_tot
            D_b = (-2.0 / N) * S_b_tot
            D_c = (-2.0 / N) * S_c_tot

            # Update model parameters
            params[0] -= L * D_a
            params[1] -= L * D_b
            params[2] -= L * D_c

            logger.debug(f"Epoch {epoch}: Updated model parameters\n")

            a, b, c = params

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: a={a}, b={b}, c={c}\n")

    if rank == 0:
        end_time = time.time()
        print(f"Nonlinear Regression training completed in {end_time - start_time:.2f} seconds.")

        print(f'Final Parameters: a: {a}, b: {b}, c: {c}')

        # Make predictions with the trained model
        Y_pred = a * X_full * X_full + b * X_full + c

        # Visualize the results
        plt.scatter(X_full, Y_full)
        plt.scatter(X_full, Y_pred , color='red') # predicted
        plt.show()