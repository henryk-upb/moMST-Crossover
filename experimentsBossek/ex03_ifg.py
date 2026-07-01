from EA.Operators import InsertionFirstGlobalMutator
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
import math
import timeit
import random

# run
# python -OO ex03_ifg.py > results/ex03_ifg.csv

# Number of nodes
ns = [100, 200, 300]

# fractions of m
ks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Runs
R = 30

print('n sigmafrac sigma repl runtime')


for n in ns:
    # generate a random complete graph
    g = GraphGenerator.generate_kc_random(n, 1)
    m = g.get_m()
    max_sigma = m - (n - 1)

    # now calculate the sigma values

    for k in ks:
        sigma = math.ceil(k * m)
        ifg = InsertionFirstGlobalMutator(g, sigma = sigma, fixed = True)

        for repl in range(1, R + 1):
            parent = RandomSpanningTreeGenerator.sample(g)

            random.seed(repl)
            time_ifg = timeit.timeit('ifg.create(parent)', number = 1, globals = globals())

            print(f'{n} {k} {sigma} {repl} {time_ifg}')


# for sigma in range(100, max_sigma, 200):
#     ifg = InsertionFirstGlobalMutator(g, sigma = sigma, fixed = True)
#     for repl in range(1, R + 1):
#         parent = RandomSpanningTreeGenerator.sample(g)

#         random.seed(repl)
#         time_ifg = timeit.timeit('ifg.create(parent)', number = 1, globals = globals())

#         print(f'{n} {sigma} {repl} {time_ifg}')
