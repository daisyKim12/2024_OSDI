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

logging.basicConfig(
    level=logging.DEBUG
)

def load_graph(algorithm, dataset):
    with open(f'./_graph/{algorithm}/{dataset}.pkl', 'rb') as f:
        graph = pickle.load(f)
    return graph

def main():

    parser = argparse.ArgumentParser(description='graph name')
    parser.add_argument('--dataset', type=str, default='sift-128-euclidean')
    parser.add_argument('--algorithm', type=str, default='cagra')
    args = parser.parse_args()

    graph = load_graph(args.algorithm, args.dataset)
    logging.debug(f"Graph loaded: row {graph.shape[0]}, col {graph.shape[1]}")    

    # # Search the graph
    # search(graph)

    # # Fetch ground truth
    # ground_truth = fetch_ground_truth(args.algorithm, args.dataset)

if __name__ == "__main__":
    main()