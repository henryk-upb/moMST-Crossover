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
from operator_comparison_runner_expyrimenter import setup_mutator

'''
SURVIVAL SELECTION VARIANTS AND OFFSPRING CLONES

The pure crossover operators (UNION, INTER, XOR, exact) and the pure mutation
operator USGS-PRE, as in ex10, under three survival selection variants:

  * baseline:  original NSGA2 (identical to ex10, same seeds)
  * cd_fix:    standard crowding distance d_1 + d_2 instead of the original
               implementation's 2 * d_1 + d_2 (NSGA2 fixed_crowding_distance)
  * dedup:     duplicate trees are removed from R = P + Q before survival
               selection (NSGA2 eliminate_duplicates), original crowding distance

Additionally every job counts, per generation, how often an offspring is
identical to one of its parents (clone_stats.csv). The counting wrappers and
the generation observer consume no random numbers, so the runs stay identical
to runs without them. For crossover each parent is checked separately, so
the per-parent clone rate is comparable to the single parent of mutation.
'''

REGISTRY_PATH = "results/approximations/ex14_survival_variants"
OPERATORS = ["UNION", "INTER", "XOR", "USGS-PRE"]
CROSSOVER_OPERATORS = ["UNION", "INTER", "XOR"]
VARIANTS = ["baseline", "cd_fix", "dedup"]
VARIANT_KWARGS = {
    "baseline": dict(),
    "cd_fix": dict(fixed_crowding_distance=True),
    "dedup": dict(eliminate_duplicates=True),
}
INSTANCE_NUMBERS = range(1, 6)  # instances _1 .. _5 of every class
MU = 100
LAMBDAA = 100
GENS = 100

CLONE_FIELDS = ["gen", "cx_calls", "cx_parents_identical", "cx_eq_p1", "cx_eq_p2", "cx_eq_any",
                "cx_eq_any_distinct_parents", "mut_calls", "mut_eq_parent"]


def tree_key(tree):
    return frozenset((e.v, e.w) if e.v < e.w else (e.w, e.v) for e in tree.get_edges())


class CloneCounter:
    '''Wraps the variation operators and counts offspring identical to a parent, per generation.
    Uses no randomness.'''

    def __init__(self):
        self.gen = 1
        self.rows = {}

    def row(self):
        return self.rows.setdefault(self.gen, {f: 0 for f in CLONE_FIELDS if f != "gen"})

    def wrap_crossover(self, cx):
        def wrapped(x, y):
            child = cx(x, y)
            r = self.row()
            kx, ky, kc = tree_key(x), tree_key(y), tree_key(child)
            r["cx_calls"] += 1
            r["cx_parents_identical"] += kx == ky
            r["cx_eq_p1"] += kc == kx
            r["cx_eq_p2"] += kc == ky
            r["cx_eq_any"] += kc == kx or kc == ky
            r["cx_eq_any_distinct_parents"] += kx != ky and (kc == kx or kc == ky)
            return child
        return wrapped

    def wrap_mutator(self, mut):
        def wrapped(x):
            child = mut(x)
            r = self.row()
            r["mut_calls"] += 1
            r["mut_eq_parent"] += tree_key(child) == tree_key(x)
            return child
        return wrapped

    def observer(self, gen, parents, offspring, survivors):
        self.gen = gen + 1

    def write(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CLONE_FIELDS)
            w.writeheader()
            for g in sorted(self.rows):
                w.writerow({"gen": g, **self.rows[g]})


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

        counter = CloneCounter()
        is_crossover = operator in CROSSOVER_OPERATORS
        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=counter.wrap_crossover(setup_crossover(graph, operator, n)) if is_crossover else None,
            mutator=counter.wrap_mutator(setup_mutator(graph, operator, n)) if not is_crossover else None,
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            on_generation=counter.observer,
            **VARIANT_KWARGS[variant])

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)
        counter.write(os.path.join(out_path, "clone_stats.csv"))

        return {"runtime": total_time, "duplicates_removed": algorithm.duplicates_removed}

    return run_experiment


def build_registry(registry_path, instances, operators, variants, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, variant=variants, repl=repls)
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/100"
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    instances = [p for p in instances if int(p.rsplit("_", 1)[1].split(".")[0]) in INSTANCE_NUMBERS]
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
