from EA.Operators import SGSECJMutator, SGSImprovedMutator, SGSSimplifiedMutator, USGMutator
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.utils.EdgeOrderer import EdgeOrdererRandom, EdgeOrdererExact
import math
import random
import timeit

# run
# python -OO ex00_sgs.py > results/ex00_sgs.csv

def half(n):
    """Set threshold to n/2."""
    return n / 2

def logsquared(n):
    return math.ceil(math.log2(n) * math.log2(n))

ns = [100 * i for i in range(1, 11)]

gs = [GraphGenerator.generate_kc_random(n) for n in ns]
funs = {'log': math.log, 'sqrt': math.sqrt, 'logsquared': logsquared, 'half': half}
R = 30

# OBSERVATION:
# * USGS for n=1000 is by a factor of 4.5 slower than R plus C++ :(
# * SGS is twice as fast as USGS on n=1000 instances (factor ~2 times worse than R plus C++)
# * PyPy3 does not work for our code currently since we are using Python3.11
#   -> make Python3.9 branch, get rid of TypeHints and run with PyPy3
# * SGSSimplified is twice as fast on average as SGS
# * With edge_orderer = EdgeOrdererRandom() USGS and SGSSimplified are faster, but SGSMutator is slower!!!
# -> SGSMutator does not use KruskalRelink, it ignores the edge_orderer
# * can we sort edges in-place instead of making copies?


print('n sigma.fun sigma repl SGSECJ SGSI SGS')

for g in gs:
    n = g.get_n()
    for fun_name, fun in funs.items():
        sigma = math.ceil(fun(n))
        edge_orderer = EdgeOrdererExact()
        sgsecj = SGSECJMutator(g, sigma = sigma, fixed = True, edge_orderer = edge_orderer)
        sgsi = SGSImprovedMutator(g, sigma = sigma, fixed = True, edge_orderer = edge_orderer)
        sgss = SGSSimplifiedMutator(g, sigma = sigma, fixed = True, edge_orderer = edge_orderer)
        # usgs = USGMutator(g, sigma = sigma, fixed = True, filter_edges = False, edge_orderer = edge_orderer)

        for repl in range(1, R + 1):
            # print(f'Iteration: {repl}')
            parent = RandomSpanningTreeGenerator.sample(g)

            random.seed(repl)
            time_sgsecj = timeit.timeit('sgsecj.create(parent)', number = 1, globals = globals())

            random.seed(repl)
            time_sgsi = timeit.timeit('sgsi.create(parent)', number = 1, globals = globals())

            random.seed(repl)
            time_sgss = timeit.timeit('sgss.create(parent)', number = 1, globals = globals())

            # random.seed(repl)
            # time_usgs = timeit.timeit('usgs.create(parent)', number = 1, globals = globals())

            print(f'{n} "{fun_name}" {sigma} {repl} {time_sgsecj} {time_sgsi} {time_sgss}')
