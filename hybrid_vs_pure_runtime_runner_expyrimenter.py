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
import pc_sweep_exact_xor_runner_expyrimenter as hybrid_setup

'''
HYBRID VS. USGS-PRE VS. PURE CROSSOVER: PERFORMANCE AND RUNTIME

All configurations in ONE registry, run interleaved, so that their runtimes are
measured under the same machine conditions (ex10 ran in Low Power Mode and ex17
at a different time, so their runtimes are not comparable). On all 40 instances
(C1-C4, 10 each) with 10 repetitions and the same seeds as ex10/ex17:
  * UNION, INTER, XOR: pure exact crossover (identical to ex10)
  * USGS-PRE:          pure mutation (identical to ex10)
  * HYBRID-0.4:        exact XOR + USGS-PRE, Mode 1 with p_crossover = 0.4
                       (identical to ex17)

Besides the wall-clock runtime (runtime.csv), every job writes the CPU time of
the optimisation (cpu_time.csv, time.process_time()), which is much less
affected by other load on the machine and does not include system sleep.
'''

REGISTRY_PATH = "results/approximations/ex19_hybrid_vs_pure_runtime"
CROSSOVER_OPERATORS = ["UNION", "INTER", "XOR"]
OPERATORS = CROSSOVER_OPERATORS + ["USGS-PRE", "HYBRID-0.4"]
HYBRID_PC = 0.4
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
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all operators (as ex10/ex17) ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        gens = max_gens if max_gens is not None else GENS
        max_evals = MU + LAMBDAA * gens

        if operator in CROSSOVER_OPERATORS:
            algo_kwargs = dict(crossover=setup_crossover(graph, operator, n), mutator=None)
        elif operator == "USGS-PRE":
            algo_kwargs = dict(crossover=None, mutator=setup_mutator(graph, operator, n))
        elif operator == "HYBRID-0.4":
            algo_kwargs = dict(crossover=hybrid_setup.setup_crossover(graph, n),
                               mutator=hybrid_setup.setup_mutator(graph, n), p_crossover=HYBRID_PC)
        else:
            raise ValueError(f"unknown operator {operator}")

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            **algo_kwargs)

        start_time = time.time()
        start_cpu = time.process_time()
        algorithm.run()
        cpu_time = time.process_time() - start_cpu
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)
        with open(os.path.join(out_path, "cpu_time.csv"), "w") as f:
            f.write(f"{cpu_time}\n")

        return {"runtime": total_time, "cpu_time": cpu_time}

    return run_experiment


def build_registry(registry_path, instances, operators, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/100"
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    repls = list(range(1, 11))

    reg = build_registry(REGISTRY_PATH, instances, OPERATORS, repls,
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
