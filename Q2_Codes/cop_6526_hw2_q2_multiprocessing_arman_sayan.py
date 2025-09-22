# Created: September 21, 2025
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
# pip install multiprocessing (if not already installed)
# python3 cop_6526_hw2_q2_multiprocessing_arman_sayan.py --debug (to see detailed logs of each process)
# python3 cop_6526_hw2_q2_multiprocessing_arman_sayan.py (to run normally without debug logs)

# Question 2.1 : Parallelized Implementation of Nonlinear Regression using Python Multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count, current_process
import time
import logging
import argparse

logger = logging.getLogger("train")   # name your logger (e.g. "train")
handler = logging.StreamHandler()     # log to stdout
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Helper functions for parallel Nonlinear Regression
def ComputeGradients(X_chunk, Y_chunk, grad_queue, param_queue, log_level):
    """
    Compute gradients for a chunk of data.
    """
    logger = logging.getLogger("train")
    logger.setLevel(log_level)

    while True:
        # Learn the latest model parameters from the main process
        params = param_queue.get()
        if params is None:
            break  # Exit signal to finish the process
        a, b, c, epoch = params

        proc = current_process()
        logger.debug(f"[Grad Comp] Epoch {epoch} Worker {proc.name} (PID={proc.pid}) is processing {len(X_chunk)} samples")

        # Make predictions and compute gradients
        Y_pred_chunk = a * X_chunk * X_chunk + b * X_chunk + c
        res = (Y_chunk - Y_pred_chunk) # residuals

        # Compute un-normalized sums and send to the main process
        grad_a_total = np.sum(X_chunk * X_chunk * res)
        grad_b_total = np.sum(X_chunk * res)
        grad_c_total = np.sum(res)
        grad_queue.put((grad_a_total, grad_b_total, grad_c_total, len(X_chunk)))


def ParallelNonlinearRegression(X, Y, L=0.0001, epochs=10000):
    """
    Perform parallel nonlinear regression using multiprocessing.
    """
    # Initialize model parameters
    a, b, c = 0.0, 0.0, 0.0

    # Split data into chunks for each process
    num_processes = cpu_count()
    print(f"Using {num_processes} processes for parallel computation.")
    X_chunks = np.array_split(X, num_processes)
    Y_chunks = np.array_split(Y, num_processes)
    print(f"\nData has been split into {num_processes} chunks.")

    # Create a queue to collect results from each process
    # and another queue to send updated model parameters
    grad_queue = Queue()
    param_queue = Queue()

    # Start worker processes
    print("\nStarting parallel Nonlinear Regression...")
    start_time = time.time()
    workers = []
    for i in range(num_processes):
        worker_args = (X_chunks[i], Y_chunks[i], grad_queue, param_queue, logger.level)
        p = Process(target=ComputeGradients, args=worker_args)
        p.start()
        workers.append(p)

    # Wait for all processes to finish with each epoch
    for epoch in range(epochs):

        # Send current model parameters to all processes
        model_params = (a, b, c, epoch)
        for _ in range(num_processes):
            param_queue.put(model_params)

        # Collect partial gradients from all processes
        grads = [grad_queue.get() for _ in range(num_processes)]
        grad_a_parts, grad_b_parts, grad_c_parts, chunk_sizes = zip(*grads)
        logger.debug(f"Epoch {epoch}: Collected gradients from all processes.")

        # Aggregate over the whole dataset
        total_chunk_size = np.sum(chunk_sizes)
        grad_a_total = np.sum(grad_a_parts)
        grad_b_total = np.sum(grad_b_parts)
        grad_c_total = np.sum(grad_c_parts)

        # Average gradients to calculate overall derivative
        D_a = (-2.0 / total_chunk_size) * grad_a_total
        D_b = (-2.0 / total_chunk_size) * grad_b_total
        D_c = (-2.0 / total_chunk_size) * grad_c_total

        # Update model parameters
        a -= L * D_a
        b -= L * D_b
        c -= L * D_c
        logger.debug(f"Epoch {epoch}: Updated model parameters\n")

        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: a={a}, b={b}, c={c}\n")

    # Ensure all processes have finished
    for _ in range(num_processes):
        param_queue.put(None)  # Signal processes to exit
    for p in workers:
        p.join()

    end_time = time.time()
    print(f"Nonlinear Regression training completed in {end_time - start_time:.2f} seconds.")

    return a, b, c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Set logging level only for *your* logger
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Preprocessing Input data
    data = pd.read_csv('20K_Datapoints.csv')
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    plt.scatter(X, Y)
    plt.show()

    # Set the model parameters
    L = 0.0001  # The learning Rate
    epochs = 10000  # The number of iterations to perform gradient descent

    # Build and train the model in parallel
    a, b, c = ParallelNonlinearRegression(X, Y, L, epochs)
    print(f'Final Parameters: a: {a}, b: {b}, c: {c}')

    # Make predictions with the trained model
    Y_pred = a*X*X + b*X + c

    # Visualize the results
    plt.scatter(X, Y)
    plt.scatter(X, Y_pred , color='red') # predicted
    plt.show()