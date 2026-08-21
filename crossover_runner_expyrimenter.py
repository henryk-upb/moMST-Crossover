#!/usr/bin/env python
import os
import random
import sys

import pandas as pd

# expyrimenter is a git submodule under external/expyrimenter/ (see
# .gitmodules) -- it is its own top-level "Experiments" package, so it
# needs its own sys.path entry rather than living at the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "expyrimenter"))
from Experiments.Registry import Registry
from Experiments.Backend import JoblibRunnerBackend, SequentialRunnerBackend

from Graph.Graph import Graph
from EA.Operators import UnionCrossover, IntersectionCrossover, XORCrossover
from EA.utils.RandomWeightSampler import WeightSampler, RandomWeightSampler, EquidistantWeightSampler
from EA.utils.EdgeSampler import EdgeSampler, EdgeSamplerUniform
from EA.utils.EdgeOrderer import EdgeOrderer, EdgeOrdererExact
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Algorithms import NSGA2


REGISTRY_PATH = "results/approximations/ex05"


def setup_initializer(graph):
    return lambda: RandomSpanningTreeGenerator.sample(graph, method="broder")


def setup_crossover(graph, crossover, n):
    sigma = n // 2
    crossover_object = None
    edge_orderer = EdgeOrdererExact()
    edge_sampler = EdgeSamplerUniform(graph.get_edges())
    weight_sampler = RandomWeightSampler()
    if crossover == "UNION":
        crossover_object = UnionCrossover(graph, edge_orderer, weight_sampler)
    elif crossover == "INTER":
        crossover_object = IntersectionCrossover(graph, edge_sampler, edge_orderer, weight_sampler, sigma=sigma)
    elif crossover == "XOR":
        crossover_object = XORCrossover(graph, edge_sampler, edge_orderer, weight_sampler, sigma=sigma)

    return lambda x, y: crossover_object.create(x, y)


def objective_fun(tree):
    return tree.get_sum_of_edge_costs()


def make_runner(registry_path, max_gens=None):
    """
    Builds the runner function expyrimenter calls as runner(jobid, params).
    max_gens overrides the generation budget for quick smoke tests; leave
    None in production to use the paper's 100*n / (mu + lambda*gens) budget.
    """

    def run_experiment(jobid, params):
        path_to_instance = params["instance"]
        crossover = params["crossover"]
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all crossovers ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        MU = 100
        LAMBDAA = 10
        gens = max_gens if max_gens is not None else 100
        max_evals = MU + LAMBDAA * gens

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=setup_crossover(graph, crossover, n),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10)

        import time
        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)

        # small summary dict -> pickled by expyrimenter into the job's result file
        return {
            "n_final": len(algorithm.final_population),
            "runtime": total_time,
        }

    return run_experiment


def build_registry(registry_path, instances, crossovers, repls, predicate=None, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, crossover=crossovers, repl=repls,
                 predicate=predicate or (lambda params: True))
    return reg


if __name__ == "__main__":
    # ---- full-scale design (mirrors the old ex05_parameters.py grid) ----
    path_to_instances = "instances/100"
    instances = [os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)]
    crossovers = ["UNION", "INTER", "XOR"]
    repls = list(range(1, 11))  # 10 independent reps, as in ex05_parameters.py

    # example use of the predicate feature Jakob just added: exclude a
    # parameter combination instead of hand-editing the factorial grid.
    # Replace with whatever exclusion you actually need, or drop the
    # argument entirely for the plain full factorial design.
    def keep(params):
        return True

    reg = build_registry(REGISTRY_PATH, instances, crossovers, repls,
                          predicate=keep,
                          backend=JoblibRunnerBackend(ncores=os.cpu_count() - 1))
    print(reg)

    runner = make_runner(REGISTRY_PATH)  # max_gens=None -> full 100-generation budget
    reg.run(runner, batchsize=3 * (os.cpu_count() - 1))

    print(reg)
    print("failed:", reg.get_failed())
    if reg.get_failed():
        reg.show_logs(reg.get_failed())

    # quick pandas overview, e.g. for a sanity check right after the run
    res = reg.get_results(jobids=reg.get_done(), simplify=True)
    print(pd.DataFrame(res).drop(columns=["n_final"]).to_string())
