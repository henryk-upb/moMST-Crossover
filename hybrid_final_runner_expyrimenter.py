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
from pc_sweep_exact_xor_runner_expyrimenter import setup_initializer, setup_crossover, setup_mutator, objective_fun
from edge_diversity_runner_expyrimenter import EdgeDiversityTracker
from survival_variants_runner_expyrimenter import CloneCounter

'''
HYBRID VS. PURE OPERATORS ON ALL INSTANCES

Final comparison of the hybrid chapter: exact XOR + USGS-PRE in Mode 1
("either/or") with the crossover probabilities that were best in the ex12
sweep (p_crossover = 0.2: C1, 0.4: C3; C2/C4 flat), against pure USGS-PRE
(p_crossover = 0.0) and pure XOR (p_crossover = 1.0), on all 40 instances
(C1-C4, 10 each) with 10 repetitions, same seeds as ex10/ex12.

Every job additionally writes, via observers/wrappers that consume no random
numbers (runs stay identical):
  * edge_diversity.csv  (as ex11: distinct edges, new edges, ... per generation)
  * clone_stats.csv     (as ex14: offspring identical to a parent per generation)
'''

REGISTRY_PATH = "results/approximations/ex15_hybrid_final"
PC_VALUES = [0.0, 0.2, 0.4, 1.0]
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
        pc = float(params["p_crossover"])
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all pc values/instances (as ex10/ex12) ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        gens = max_gens if max_gens is not None else GENS
        max_evals = MU + LAMBDAA * gens

        counter = CloneCounter()
        crossover = counter.wrap_crossover(setup_crossover(graph, n))
        mutator = counter.wrap_mutator(setup_mutator(graph, n))
        if pc <= 0.0:
            algo_kwargs = dict(crossover=None, mutator=mutator)
        elif pc >= 1.0:
            algo_kwargs = dict(crossover=crossover, mutator=None)
        else:
            algo_kwargs = dict(crossover=crossover, mutator=mutator, p_crossover=pc)

        tracker = EdgeDiversityTracker()

        def observer(gen, parents, offspring, survivors):
            tracker(gen, parents, offspring, survivors)
            counter.observer(gen, parents, offspring, survivors)

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            on_generation=observer,
            **algo_kwargs)

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)
        tracker.write(out_path)
        counter.write(os.path.join(out_path, "clone_stats.csv"))

        return {"runtime": total_time}

    return run_experiment


def build_registry(registry_path, instances, pc_values, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, p_crossover=pc_values, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/100"
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    repls = list(range(1, 11))

    reg = build_registry(REGISTRY_PATH, instances, PC_VALUES, repls,
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
