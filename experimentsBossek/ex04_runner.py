#!/usr/bin/env python
from Graph.Graph import Graph
from EA.Operators import USGMutator, SGSSimplifiedMutator, USGMutatorWithPresorting, InsertionFirstGlobalMutator, InsertionFirstLocalMutator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Algorithms import NSGA2

import pprint
import random
import csv
import itertools
import os
import math
import time

from joblib import Parallel, delayed # pip install joblib
import multiprocessing
import sys

from Experiments.utils import read_setup

'''
FINAL EXPERIMENTS
===

We want to vary:
* All 40 ECJ instances with n = 100 nodes (each 10 of classes C1 to C4)
* 5 Operators:
*   - USGS-F with edge filtering (and log^2(n) sigma)
*.  - USGS-PRE with pre-sorting (and log^2(n) sigma)
*.  - SGS simplified (and log^2(n) sigma)
*.  - IF-G (and linear sigma)
*.  - IF-L (and log^2(n) node sigma and linear edge sigma)
* L = n (= 100) equidistant weights for the pre-sorting
* Termination criterion: 100n (the 10% limit of the ECJ paper)
* Independent runs: 30

SETUP
===

I guess there is no need for highly parallel experimentation.
However, we should go for local multi-core parallelization
(subprocess to the rescue!)
To this end, instead of nested loops, we should lazily iterate
over a collection of tuples/lists (n, r, R, L, type, ...) and
call a run_experiment function which also deals with output.
'''

def setup_initializer(graph):
    return lambda: RandomSpanningTreeGenerator.sample(graph, method = "broder")

def setup_mutator(graph, mutator, n):
    sigma = 2 * n if mutator in ["IF-G"] else math.ceil(math.log(n) * math.log(n))
    mutator_object = None
    if mutator == "USGS-F":
      mutator_object = USGMutator(graph, sigma = sigma, filter_edges = True)
    elif mutator == "USGS-PRE":
      mutator_object = USGMutatorWithPresorting(graph, sigma = sigma, L = n)
    elif mutator == "SGS":
      mutator_object = SGSSimplifiedMutator(graph, sigma = sigma)
    elif mutator == "IF-G":
      mutator_object = InsertionFirstGlobalMutator(graph, sigma = sigma)
    elif mutator == "IF-L":
      mutator_object = InsertionFirstLocalMutator(graph, sigma_nodes = math.ceil(math.log(n) * math.log(n)), sigma_edges = sigma)

    return lambda x: mutator_object.create(x)


def objective_fun(tree):
    return tree.get_sum_of_edge_costs()


def run_experiment(setup_path, jobid, out_path):
    # read experimental setup
    setup = read_setup(setup_path)
    #print(setup)
    path_to_instance = setup['instance']

    random.seed(setup['seed'])

    # import instance
    start_time = time.time()
    graph = Graph.import_grapherator_format(path_to_instance)
    n = graph.get_n()

    # define algorithm parameters
    MU = 100
    LAMBDAA = 10
    sigma = math.ceil(math.log(n) * math.log(n))
    max_evals = 100 * n
    log_every = 10

    # instantiate algorithm
    algorithm = NSGA2(
      mu = MU,
      lambdaa = LAMBDAA,
      initializer = setup_initializer(graph),
      mutator = setup_mutator(graph, setup["mutator"], n),
      objective_fun = objective_fun,
      max_evals = max_evals,
      log_every = 10)

    # run and produce output
    res = algorithm.run()
    total_time = time.time() - start_time
    algorithm.write_result(out_path, total_time)

# 1: setup_path, 2: jobid, 3: out_path
run_experiment(sys.argv[1], sys.argv[2], sys.argv[3])
