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
from Experiments.Backend import JoblibRunnerBackend, SequentialRunnerBackend

from Graph.Graph import Graph
from EA.Operators import USGMutatorWithPresorting, XORCrossover
from EA.utils.RandomWeightSampler import RandomWeightSampler
from EA.utils.EdgeOrderer import EdgeOrdererExact
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Algorithms import NSGA2

'''
HYBRID MODES BY INSTANCE CLASS, EXACT XOR

Same design as hybrid_modes_by_class_runner_expyrimenter.py (ex08: same
instances, P_MIX, repls and seeds), but with the EXACT XOR crossover
(XORCrossover, as in ex10/ex12) instead of XORCrossoverSampled.

Compares the three mutually exclusive ways of combining a crossover and a
mutation operator per offspring (see EA.Algorithms.NSGA2's docstring),
fixing the operator pair to XOR + USGS-PRE (the strongest hybrid pairing
found in the earlier crossover/mutation comparison), across one
representative instance per class C1-C4 (the "_6" instance of each class,
same as the single-instance long-run comparison this generalizes).

Modes (design.csv column "mode"):
  * mode1: p_crossover = P_MIX  ("either/or" -- NSGA2's p_crossover)
  * mode2: p_c = P_MIX, p_m = P_MIX  ("independent Bernoulli trials")
  * mode3: neither given -- crossover always, then mutation always ("sequential")
'''

REGISTRY_PATH = "results/approximations/ex16_hybrid_modes_exact_xor"

CLASS_INSTANCES = [
    "instances/100/CLASS1_100_4950_0_2_UNG_CEG_RWG-RWG_6.graph",
    "instances/100/CLASS2_100_4950_0_2_UNG_CEG_CONCWG_6.graph",
    "instances/100/CLASS3_100_4950_0_2_UNG_CEG_-0.95--CORWG_6.graph",
    "instances/100/CLASS4_100_4950_0_2_UNG_CEG_0.95--CORWG_6.graph",
]

MODES = ["mode1", "mode2", "mode3"]
P_MIX = 0.5


def setup_initializer(graph):
    return lambda: RandomSpanningTreeGenerator.sample(graph, method="broder")


def setup_crossover(graph, n):
    # exact XOR, identical to ex10's / ex12's "XOR" (no randomly sampled outside edges)
    crossover_object = XORCrossover(graph, EdgeOrdererExact(), RandomWeightSampler())
    return lambda x, y: crossover_object.create(x, y)


def setup_mutator(graph, n):
    sigma = math.ceil(math.log(n) * math.log(n))
    mutator_object = USGMutatorWithPresorting(graph, sigma=sigma, L=n)
    return lambda x: mutator_object.create(x)


def objective_fun(tree):
    return tree.get_sum_of_edge_costs()


def make_runner(registry_path, max_gens=None):
    """
    Builds the runner function expyrimenter calls as runner(jobid, params).
    params["mode"] selects which of NSGA2's three reproduction modes is used;
    max_gens overrides the generation budget for quick smoke tests; leave
    None in production to use the 100-generation default shared with the
    other ex0X designs.
    """

    def run_experiment(jobid, params):
        path_to_instance = params["instance"]
        mode = params["mode"]
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all modes/instances ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        MU = 100
        LAMBDAA = 100
        gens = max_gens if max_gens is not None else 100
        max_evals = MU + LAMBDAA * gens

        mode_kwargs = {
            "mode1": dict(p_crossover=P_MIX),
            "mode2": dict(p_c=P_MIX, p_m=P_MIX),
            "mode3": dict(),
        }[mode]

        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=setup_crossover(graph, n),
            mutator=setup_mutator(graph, n),
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            **mode_kwargs)

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)

        return {"runtime": total_time}

    return run_experiment


def build_registry(registry_path, instances, modes, repls, predicate=None, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, mode=modes, repl=repls,
                 predicate=predicate or (lambda params: True))
    return reg


if __name__ == "__main__":
    repls = list(range(1, 11))  # 10 independent reps, as in the other ex0X designs

    reg = build_registry(REGISTRY_PATH, CLASS_INSTANCES, MODES, repls,
                          backend=JoblibRunnerBackend(ncores=os.cpu_count() - 1))
    print(reg)

    runner = make_runner(REGISTRY_PATH)  # max_gens=None -> full 100-generation budget
    reg.run(runner, batchsize=3 * (os.cpu_count() - 1))

    print(reg)
    print("failed:", reg.get_failed())
    if reg.get_failed():
        reg.show_logs(reg.get_failed())

    res = reg.get_results(jobids=reg.get_done(), simplify=True)
    print(pd.DataFrame(res).to_string())
