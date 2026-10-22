#!/usr/bin/env python
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
from Experiments.Backend import JoblibRunnerBackend, SequentialRunnerBackend

from Graph.Graph import Graph
from EA.Algorithms import NSGA2
from crossover_runner_expyrimenter import setup_initializer, setup_crossover, objective_fun
from operator_comparison_runner_expyrimenter import setup_mutator

'''
PURE CROSSOVER vs. USGS-PRE (first comparison of the thesis' evaluation chapter)

The three pure crossover operators in their exact formalisation (UNION, INTER,
XOR -- i.e. crossover_runner_expyrimenter.py's setup_crossover(), NOT the
Sampled variants that operator_comparison_runner_expyrimenter.py registers
under the same names) against the pure-mutation baseline USGS-PRE, as ONE
"operator" factor of a single registry (same instances, same common-random-
number seeds per repl), on all four instance classes C1-C4 with n=100.
'''

REGISTRY_PATH = "results/approximations/ex10_pure_cx_vs_usgs_pre"

CROSSOVER_OPERATORS = ["UNION", "INTER", "XOR"]
MUTATION_OPERATORS = ["USGS-PRE"]

MU = 100
LAMBDAA = 100
GENS = 100


def make_runner(registry_path, max_gens=None):
    """
    Builds the runner function expyrimenter calls as runner(jobid, params).
    params["operator"] is one of CROSSOVER_OPERATORS (pure crossover run) or
    MUTATION_OPERATORS (pure mutation run).
    max_gens overrides the generation budget for quick smoke tests.
    """

    def run_experiment(jobid, params):
        path_to_instance = params["instance"]
        operator = params["operator"]
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all operators ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        gens = max_gens if max_gens is not None else GENS
        max_evals = MU + LAMBDAA * gens

        is_crossover = operator in CROSSOVER_OPERATORS
        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=setup_crossover(graph, operator, n) if is_crossover else None,
            mutator=setup_mutator(graph, operator, n) if not is_crossover else None,
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


def build_registry(registry_path, instances, operators, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/100"
    # ordered by instance number first, then class: the registry runs jobs in this order, so
    # after every 4 instances one more complete instance PER CLASS is available for analysis
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    operators = CROSSOVER_OPERATORS + MUTATION_OPERATORS
    repls = list(range(1, 11))  # 10 independent reps, as in the other ex0X designs

    reg = build_registry(REGISTRY_PATH, instances, operators, repls,
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
