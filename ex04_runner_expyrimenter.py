#!/usr/bin/env python
import math
import os
import random
import sys
import time

import pandas as pd

# expyrimenter is a git submodule under external/expyrimenter/ (see
# .gitmodules) -- it is its own top-level "Experiments" package, so it
# needs its own sys.path entry rather than living at the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "expyrimenter"))
from Experiments.Registry import Registry
from Experiments.Backend import JoblibRunnerBackend

from Graph.Graph import Graph
from EA.Operators import USGMutator, SGSSimplifiedMutator, USGMutatorWithPresorting, InsertionFirstGlobalMutator, InsertionFirstLocalMutator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Algorithms import NSGA2



REGISTRY_PATH = "results/approximations/ex04"


def setup_initializer(graph):
    return lambda: RandomSpanningTreeGenerator.sample(graph, method="broder")


def setup_mutator(graph, mutator, n):
    sigma = 2 * n if mutator in ["IF-G"] else math.ceil(math.log(n) * math.log(n))
    mutator_object = None
    if mutator == "USGS-F":
        mutator_object = USGMutator(graph, sigma=sigma, filter_edges=True)
    elif mutator == "USGS-PRE":
        mutator_object = USGMutatorWithPresorting(graph, sigma=sigma, L=n)
    elif mutator == "SGS":
        mutator_object = SGSSimplifiedMutator(graph, sigma=sigma)
    elif mutator == "IF-G":
        mutator_object = InsertionFirstGlobalMutator(graph, sigma=sigma)
    elif mutator == "IF-L":
        mutator_object = InsertionFirstLocalMutator(graph, sigma_nodes=math.ceil(math.log(n) * math.log(n)), sigma_edges=sigma)

    return lambda x: mutator_object.create(x)


def objective_fun(tree):
    return tree.get_sum_of_edge_costs()


def make_runner(registry_path, max_evals_factor=100):
    """max_evals_factor mirrors the "100n (10% limit of the ECJ paper)" budget."""

    def run_experiment(jobid, params):
        repl = int(params["repl"])

        # common random numbers: same seed for repl=k across all mutators
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(params["instance"])
        n = graph.get_n()

        MU = 100
        LAMBDAA = 10
        max_evals = max_evals_factor * n

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            mutator=setup_mutator(graph, params["mutator"], n),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10)

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)

        return {"runtime": total_time}

    return run_experiment


def build_registry(registry_path, instances, mutators, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or JoblibRunnerBackend())
    reg.add_jobs(instance=instances, mutator=mutators, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/50"
    instances = [os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)]
    mutators = ["USGS-F", "USGS-PRE", "SGS", "IF-L", "IF-G"]
    repls = list(range(1, 11))

    reg = build_registry(REGISTRY_PATH, instances, mutators, repls,
                          backend=JoblibRunnerBackend(ncores=os.cpu_count() - 1))
    print(reg)

    runner = make_runner(REGISTRY_PATH)
    reg.run(runner, batchsize=3 * (os.cpu_count() - 1))

    print(reg)
    print("failed:", reg.get_failed())
    if reg.get_failed():
        reg.show_logs(reg.get_failed())

    res = reg.get_results(jobids=reg.get_done(), simplify=True)
    print(pd.DataFrame(res).to_string())
