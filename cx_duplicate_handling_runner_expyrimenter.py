#!/usr/bin/env python
import csv
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

'''
DUPLICATE HANDLING FOR PURE CROSSOVER

The three pure crossover operators in their exact formalisation (UNION, INTER,
XOR, as in ex10) with three ways of handling identical trees ("variant"):

  * baseline:          original NSGA2 sampling (identical to ex10, same seeds)
  * distinct:          the second parent is redrawn until its tree differs from
                       the first parent's tree (NSGA2 distinct_parents)
  * distinct_noclone:  distinct parents, and an offspring identical to one of its
                       parents is discarded without evaluation and does not count
                       towards the lambda offspring (NSGA2 discard_clones)

All three variants run interleaved in one registry, so their runtimes are
measured under the same machine conditions. Each job additionally writes
repro_stats.csv with the duplicate-handling counters of the run.
'''

REGISTRY_PATH = "results/approximations/ex13_cx_duplicate_handling"
OPERATORS = ["UNION", "INTER", "XOR"]
VARIANTS = ["baseline", "distinct", "distinct_noclone"]
VARIANT_KWARGS = {
    "baseline": dict(),
    "distinct": dict(distinct_parents=True),
    "distinct_noclone": dict(distinct_parents=True, discard_clones=True),
}
MU = 100
LAMBDAA = 100
GENS = 100


def make_runner(registry_path, max_gens=None):
    """
    Builds the runner function expyrimenter calls as runner(jobid, params).
    max_gens overrides the generation budget for quick smoke tests.
    """

    def run_experiment(jobid, params):
        path_to_instance = params["instance"]
        operator = params["operator"]
        variant = params["variant"]
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all operators/variants (as ex10) ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        gens = max_gens if max_gens is not None else GENS
        max_evals = MU + LAMBDAA * gens

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=setup_crossover(graph, operator, n),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            **VARIANT_KWARGS[variant])

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)
        with open(os.path.join(out_path, "repro_stats.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(algorithm.repro_stats))
            w.writeheader()
            w.writerow(algorithm.repro_stats)

        return {"runtime": total_time, **algorithm.repro_stats}

    return run_experiment


def build_registry(registry_path, instances, operators, variants, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, variant=variants, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/100"
    # ordered by instance number first, then class (as ex10)
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    repls = list(range(1, 11))

    reg = build_registry(REGISTRY_PATH, instances, OPERATORS, VARIANTS, repls,
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
