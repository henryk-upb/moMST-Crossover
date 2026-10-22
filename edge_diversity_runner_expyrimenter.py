#!/usr/bin/env python
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter

import pandas as pd

# expyrimenter is a git submodule under external/expyrimenter/ (see .gitmodules)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "expyrimenter"))
from Experiments.Registry import Registry
from Experiments.Backend import JoblibRunnerBackend, SequentialRunnerBackend

from Graph.Graph import Graph
from EA.Algorithms import NSGA2
from crossover_runner_expyrimenter import setup_initializer, setup_crossover, objective_fun
from operator_comparison_runner_expyrimenter import setup_mutator

'''
EDGE DIVERSITY OF THE POPULATION UNDER PURE CROSSOVER (ex11)

Same design as ex10 (pure UNION / INTER / XOR in their exact formalisation, n=100,
40 instances C1-C4, 10 repls, mu = lambda = 100, 100 generations, same common-random-
number seeds), but every generation the edge content of the population is measured.

The exact crossover operators only ever build the child from E(T1) u E(T2), so no
edge that is absent from the population can enter it again. The number of distinct
edges in the population can therefore only stay equal or shrink, and every loss is
caused by survival selection discarding the last tree(s) carrying an edge.

Per generation t a row of edge_diversity.csv holds (E(X) = union of the edge sets of
all trees in X, an edge being the unordered node pair {v, w}):

  distinct_edges     |E(P_t)|: number of different edges in the population
  singleton_edges    edges contained in exactly ONE tree of P_t
  distinct_trees     number of different trees (edge sets) in P_t
  new_edges          |E(Q_t) \\ E(P_{t-1})|: edges the offspring bring in that no parent
                     had (always 0 for the exact CX operators; sanity check)
  pool_edges         |E(P_{t-1} u Q_t)|: distinct edges in the selection pool R_t
  lost_edges         pool_edges - distinct_edges: edges removed by survival selection
  carriers           trees in R_t with >= 1 edge no other tree of R_t has
  carriers_survived  how many of those carriers survive selection
  others_survived    survivors among the remaining (non-carrier) trees of R_t
  pool_size          |R_t| = mu + lambda

The gen-0 row describes the random initial population (transition columns empty).

Usage:
  python edge_diversity_runner_expyrimenter.py                 # full ex11 design
  python edge_diversity_runner_expyrimenter.py --smoke         # 1 instance/class, 1 repl, 10 gens
  python edge_diversity_runner_expyrimenter.py --operators USGS-PRE \
      --registry results/approximations/ex11_edge_diversity_usgs_pre   # mutation reference
'''

REGISTRY_PATH = "results/approximations/ex11_edge_diversity_pure_cx"
# pure-mutation reference (USGS-PRE) with the same design, kept in its own registry
USGS_PRE_REGISTRY_PATH = "results/approximations/ex11_edge_diversity_usgs_pre"

CROSSOVER_OPERATORS = ["UNION", "INTER", "XOR"]
MUTATION_OPERATORS = ["USGS-PRE"]

MU = 100
LAMBDAA = 100
GENS = 100

FIELDS = ["gen", "distinct_edges", "singleton_edges", "distinct_trees", "new_edges", "pool_edges",
          "lost_edges", "carriers", "carriers_survived", "others_survived", "pool_size"]


def edge_keys(tree):
    '''Edge set of a spanning tree as unordered node pairs (v < w).'''
    return frozenset((e.v, e.w) if e.v < e.w else (e.w, e.v) for e in tree.get_edges())


class EdgeDiversityTracker:
    '''Observer for NSGA2(on_generation = ...). Uses no randomness, so the run is
    identical to the same run without the tracker.'''

    def __init__(self):
        self.rows = []

    def __call__(self, gen, parents, offspring, survivors):
        surv_sets = [edge_keys(ind.x) for ind in survivors]
        surv_count = Counter(e for s in surv_sets for e in s)
        row = {"gen": gen,
               "distinct_edges": len(surv_count),
               "singleton_edges": sum(1 for c in surv_count.values() if c == 1),
               "distinct_trees": len(set(surv_sets))}

        if parents is not None:
            par_sets = [edge_keys(ind.x) for ind in parents]
            off_sets = [edge_keys(ind.x) for ind in offspring]
            par_edges = set().union(*par_sets)
            off_edges = set().union(*off_sets)
            pool = parents + offspring
            pool_sets = par_sets + off_sets
            pool_count = Counter(e for s in pool_sets for e in s)

            # a tree is a "carrier" if it holds at least one edge that no other tree in the
            # selection pool has -- discarding it removes that edge from the population for good
            survivor_ids = {id(ind) for ind in survivors}
            is_carrier = [any(pool_count[e] == 1 for e in s) for s in pool_sets]
            survived = [id(ind) in survivor_ids for ind in pool]

            row.update({
                "new_edges": len(off_edges - par_edges),
                "pool_edges": len(pool_count),
                "lost_edges": len(pool_count) - len(surv_count),
                "carriers": sum(is_carrier),
                "carriers_survived": sum(c and s for c, s in zip(is_carrier, survived)),
                "others_survived": sum((not c) and s for c, s in zip(is_carrier, survived)),
                "pool_size": len(pool)})
        self.rows.append(row)

    def write(self, out_path):
        with open(os.path.join(out_path, "edge_diversity.csv"), "w", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def make_runner(registry_path, max_gens=None):

    def run_experiment(jobid, params):
        path_to_instance = params["instance"]
        operator = params["operator"]
        repl = int(params["repl"])

        # --- common random numbers: same seed for repl=k across all operators (as in ex10) ---
        random.seed(repl)
        seed = random.randint(1, 1_000_000)
        random.seed(seed)

        graph = Graph.import_grapherator_format(path_to_instance)
        n = graph.get_n()

        gens = max_gens if max_gens is not None else GENS
        max_evals = MU + LAMBDAA * gens

        tracker = EdgeDiversityTracker()
        is_crossover = operator in CROSSOVER_OPERATORS
        algorithm = NSGA2(
            mu=MU,
            lambdaa=LAMBDAA,
            initializer=setup_initializer(graph),
            crossover=setup_crossover(graph, operator, n) if is_crossover else None,
            mutator=setup_mutator(graph, operator, n) if not is_crossover else None,
            objective_fun=objective_fun,
            max_evals=max_evals,
            log_every=10,
            on_generation=tracker)

        start_time = time.time()
        algorithm.run()
        total_time = time.time() - start_time

        out_path = os.path.join(registry_path, "experiments", str(jobid))
        os.makedirs(out_path, exist_ok=True)
        algorithm.write_result(out_path, total_time)
        tracker.write(out_path)

        last = tracker.rows[-1]
        return {"runtime": total_time, "distinct_edges_final": last["distinct_edges"]}

    return run_experiment


def build_registry(registry_path, instances, operators, repls, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, repl=repls)
    return reg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="quick test: first instance per class, repl 1, 10 generations")
    parser.add_argument("--operators", default=",".join(CROSSOVER_OPERATORS),
                        help="comma-separated, e.g. UNION,INTER,XOR,USGS-PRE to add the mutation reference")
    parser.add_argument("--registry", default=REGISTRY_PATH,
                        help="registry path; the registry is OVERWRITTEN, so use a new path for additional "
                             "operators (e.g. %s)" % USGS_PRE_REGISTRY_PATH)
    args = parser.parse_args()

    path_to_instances = "instances/100"
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    operators = args.operators.split(",")
    repls = list(range(1, 11))
    registry_path, max_gens = args.registry, None
    if args.smoke:
        instances, repls, max_gens = instances[:4], [1], 10
        registry_path = registry_path + "_smoke"

    reg = build_registry(registry_path, instances, operators, repls,
                         backend=JoblibRunnerBackend(ncores=os.cpu_count() - 1))
    print(reg)

    runner = make_runner(registry_path, max_gens=max_gens)
    reg.run(runner, batchsize=3 * (os.cpu_count() - 1))

    print(reg)
    print("failed:", reg.get_failed())
    if reg.get_failed():
        reg.show_logs(reg.get_failed())

    res = reg.get_results(jobids=reg.get_done(), simplify=True)
    print(pd.DataFrame(res).to_string())
