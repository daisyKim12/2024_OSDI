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
    _, queries, ground_truth, _ = fetch_dataset('../_datasets', args.dataset)
    logging.debug(f"Graph loaded: {graph.shape}")    
    logging.debug(f"Queries loaded: {queries.shape}")
    logging.debug(f"Ground truth loaded: {ground_truth.shape}")

    # Search the graph
    topk = cagra.search(torch.from_numpy(graph).to(torch.int32), torch.from_numpy(queries).to(torch.int32))
    logging.debug(f"Topk: {topk.shape}")

    # # Fetch ground truth
    # alg_ground_truth = fetch_algorithm_ground_truth(args.algorithm)
    # ground_truth = fetch_ground_truth(args.algorithm, args.dataset)

if __name__ == "__main__":
    main()