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
from EA.Operators import USGMutator, SGSSimplifiedMutator, USGMutatorWithPresorting, InsertionFirstGlobalMutator, InsertionFirstLocalMutator
from EA.Operators import (
    UnionCrossover,
    IntersectionCrossoverSampled,
    XORCrossoverSampled,
    XORCrossoverAsymmetric,
    XORCrossoverPrioritized,
    UnionCrossoverInterleaved,
    IntersectionCrossoverInterleaved,
    XORCrossoverPrioritizedInterleaved,
    XORCrossoverAsymmetricInterleaved,
)
from EA.utils.RandomWeightSampler import WeightSampler, RandomWeightSampler, EquidistantWeightSampler
from EA.utils.EdgeSampler import EdgeSampler, EdgeSamplerUniform
from EA.utils.EdgeOrderer import EdgeOrderer, EdgeOrdererExact
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Algorithms import NSGA2

'''
OPERATOR COMPARISON (pure crossover, pure mutation, AND hybrid, as ONE factor)

Unlike crossover_runner_expyrimenter.py (only crossover operators) and
ex04_runner_expyrimenter.py (only mutation operators), which live in two
separate registries with different design.csv schemas ("crossover" vs.
"mutator" column), this runner puts all operators into a single "operator"
column of ONE registry, directly comparable side by side (same instances,
same common-random-number seeds per repl) via a single
    analysis/analyze_hypervolume.py results/approximations/ex07 --algo-col operator
call across all of them at once. Three modes:
  * CROSSOVER_OPERATORS: pure crossover-only run (no mutator), like ex05.
  * MUTATION_OPERATORS: pure mutation-only run (no crossover), like ex04.
  * HYBRID_OPERATORS: crossover AND mutator together, governed by NSGA2's
    p_crossover (see EA/Algorithms.py), like crossover_mutation_runner_
    expyrimenter.py -- one hybrid per crossover type x mutator (e.g.
    "UNION+USGS-PRE").

Mutation operators are restricted to USGS-PRE (both for MUTATION_OPERATORS
and as hybrid partner): ex06's cross-instance analysis found USGS-F/USGS-PRE
to consistently outperform SGS/IF-G/IF-L, so the older operators are no
longer run; a follow-up comparison (2026-08-22 full report, ex07 n=50) found
USGS-PRE to be consistently faster than USGS-F (pre-sorted edge lists avoid
the O(log m) per-call re-sort) and at least as good on HV/Epsilon when paired
with a crossover, so USGS-F was dropped too.

XORCrossoverAsymmetric/XORCrossoverPrioritized (XOR-A/XOR-P) are excluded
from CROSSOVER_OPERATORS: diagnosed root cause was that their inherited
parent edges bypass lambda-scalarisation entirely (raw insertion order),
so they consistently ranked worst among all crossover operators (HV
boxplot: ~35-40% relative HV vs. ~90-100% for the others). Superseded by
XORCrossoverAsymmetricInterleaved/XORCrossoverPrioritizedInterleaved
(XOR-A-I/XOR-P-I), which apply the same lambda-ordered-interleaving fix
used by Union/IntersectionCrossoverInterleaved and close most (not all) of
that gap. XOR-A/XOR-P remain defined in EA/Operators.py and reachable via
setup_crossover() if ever needed again, just not run by default.
'''

REGISTRY_PATH = "results/approximations/ex07"

CROSSOVER_OPERATORS = ["UNION", "INTER", "XOR", "UNION-I", "INTER-I", "XOR-P-I", "XOR-A-I"]
MUTATION_OPERATORS = ["USGS-PRE"]
HYBRID_MUTATORS = ["USGS-PRE"]
HYBRID_P_CROSSOVER = 0.5
HYBRID_OPERATORS = [f"{c}+{m}" for c in CROSSOVER_OPERATORS for m in HYBRID_MUTATORS]


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
        crossover_object = IntersectionCrossoverSampled(graph, edge_sampler, edge_orderer, weight_sampler, sigma=sigma)
    elif crossover == "XOR":
        crossover_object = XORCrossoverSampled(graph, edge_sampler, edge_orderer, weight_sampler, sigma=sigma)
    elif crossover == "XOR-A":
        crossover_object = XORCrossoverAsymmetric(graph, edge_orderer, weight_sampler)
    elif crossover == "XOR-P":
        crossover_object = XORCrossoverPrioritized(graph, edge_orderer, weight_sampler)
    elif crossover == "UNION-I":
        crossover_object = UnionCrossoverInterleaved(graph, edge_orderer, weight_sampler)
    elif crossover == "INTER-I":
        crossover_object = IntersectionCrossoverInterleaved(graph, edge_orderer, weight_sampler)
    elif crossover == "XOR-P-I":
        crossover_object = XORCrossoverPrioritizedInterleaved(graph, edge_orderer, weight_sampler)
    elif crossover == "XOR-A-I":
        crossover_object = XORCrossoverAsymmetricInterleaved(graph, edge_orderer, weight_sampler)

    return lambda x, y: crossover_object.create(x, y)


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


def make_runner(registry_path, max_gens=None):
    """
    Builds the runner function expyrimenter calls as runner(jobid, params).
    params["operator"] is one of CROSSOVER_OPERATORS (pure crossover run),
    MUTATION_OPERATORS (pure mutation run), or HYBRID_OPERATORS (crossover
    AND mutator together, e.g. "UNION+USGS-PRE").
    max_gens overrides the generation budget for quick smoke tests; leave
    None in production to use the paper's 100*n / (mu + lambda*gens) budget.
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

        MU = 100
        LAMBDAA = 100
        gens = max_gens if max_gens is not None else 100
        max_evals = MU + LAMBDAA * gens

        if operator in HYBRID_OPERATORS:
            crossover, mutator = operator.split("+")
            algorithm = NSGA2(
                mu=MU,
                lambdaa=LAMBDAA,
                initializer=setup_initializer(graph),
                crossover=setup_crossover(graph, crossover, n),
                mutator=setup_mutator(graph, mutator, n),
                p_crossover=HYBRID_P_CROSSOVER,
                objective_fun=objective_fun,
                max_evals=max_evals,
                log_every=10)
        else:
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


def build_registry(registry_path, instances, operators, repls, predicate=None, backend=None):
    reg = Registry(path=registry_path, overwrite=True, backend=backend or SequentialRunnerBackend())
    reg.add_jobs(instance=instances, operator=operators, repl=repls,
                 predicate=predicate or (lambda params: True))
    return reg


if __name__ == "__main__":
    path_to_instances = "instances/50"
    instances = [os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)]
    operators = CROSSOVER_OPERATORS + MUTATION_OPERATORS + HYBRID_OPERATORS
    repls = list(range(1, 11))  # 10 independent reps, as in the other ex0X designs

    reg = build_registry(REGISTRY_PATH, instances, operators, repls,
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
