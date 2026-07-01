from EA.Operators import USGMutator, USGMutatorWithPresorting
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.utils.EdgeOrderer import EdgeOrdererRandom, EdgeOrdererExact
import math
import timeit
import random

# run
# python -OO ex01_usgs.py > results/ex01_usgs.csv

def half(n):
    """Set threshold to n/2."""
    return n / 2

def logsquared(n):
    return math.ceil(math.log2(n) * math.log2(n))

# Number of nodes
ns = [100 * i for i in range(1, 11)]#range(2, 21, 2)]

# Probabilities for ER-graphs
ps = [1.0]

funs = {'log': math.log, 'sqrt': math.sqrt, 'logsquared': logsquared, 'half': half}
R = 30

# Number of pre-sorted values
L = 15

# OBSERVATIONS:
# * ...

print('n p sigma.fun sigma repl USGS USGS-F USGS-PRE')

for n in ns:
    for p in ps:
        g = GraphGenerator.generate_kc_random(n, p)
        for fun_name, fun in funs.items():
            sigma = math.ceil(fun(n))
            sigma_sgs = sigma
            edge_orderer = EdgeOrdererExact()
            usgs = USGMutator(g, sigma = sigma, fixed = True, filter_edges = False, edge_orderer = edge_orderer)
            usgs_filter = USGMutator(g, sigma = sigma, fixed = True, filter_edges = True, edge_orderer = edge_orderer)
            usgs_presorting = USGMutatorWithPresorting(g, sigma = sigma, fixed = True, L = L, edge_orderer = edge_orderer)

            for repl in range(1, R + 1):
                # print(f'Iteration: {repl}')
                parent = RandomSpanningTreeGenerator.sample(g)

                random.seed(repl)
                time_usgs = timeit.timeit('usgs.create(parent)', number = 1, globals = globals())

                random.seed(repl)
                time_usgs_filter = timeit.timeit('usgs_filter.create(parent)', number = 1, globals = globals())

                random.seed(repl)
                time_usgs_presorting = timeit.timeit('usgs_presorting.create(parent)', number = 1, globals = globals())

                print(f'{n} {p} "{fun_name}" {sigma} {repl} {time_usgs} {time_usgs_filter} {time_usgs_presorting}')
