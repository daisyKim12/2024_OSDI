# Scheme 1
# Computaton Jump
# - fetch graph
# - pybind to custom search algorithm

import os
import sys
import logging
import pybind11
import pickle
import argparse
import numpy as np
import torch
from dataset_loader import fetch_dataset

# pybind
import cagra

logging.basicConfig(
    level=logging.DEBUG,
)

def get_torch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.uint32:
        return torch.uint32
    else:
        raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")
def convert_to_torch(tensor):
    torch_dtype = get_torch_dtype(tensor.dtype)
    return torch.from_numpy(tensor).to(torch_dtype)

def load_graph(base_path, algorithm, dataset):
    with open(f'{base_path}/{algorithm}/{dataset}.pkl', 'rb') as f:
        graph = pickle.load(f)
    return graph


def main():

    parser = argparse.ArgumentParser(description='graph name')
    parser.add_argument('--dataset', type=str, default='sift-128-euclidean')
    parser.add_argument('--algorithm', type=str, default='cagra')
    args = parser.parse_args()

    graph = load_graph('../_graph', args.algorithm, args.dataset)
    dataset, queries, ground_truth, _ = fetch_dataset('../_datasets', args.dataset)
    logging.debug(f"Graph loaded: {graph.shape}")    
    logging.debug(f"Queries loaded: {queries.shape}")
    logging.debug(f"Ground truth loaded: {ground_truth.shape}")
    
    graph = convert_to_torch(graph)
    dataset = convert_to_torch(dataset)
    queries = convert_to_torch(queries)

    # Search the graph
    topk = cagra.search(graph, dataset, queries)
    logging.debug(f"Topk: {topk.shape}")

    # # Fetch ground truth
    # alg_ground_truth = fetch_algorithm_ground_truth(args.algorithm)
    # ground_truth = fetch_ground_truth(args.algorithm, args.dataset)

if __name__ == "__main__":
    main()