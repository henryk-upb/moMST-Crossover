from Graph.Graph import Graph
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.Operators import SGSImprovedMutator, SGSSimplifiedMutator, USGMutator
import math
import glob

graphs = sorted(glob.glob('instances/CLASS*'))

# Graphs (same as used in ECJ2023 paper)
gs = {'c' + str(i + 1): Graph.import_grapherator_format(g) for i, g in enumerate(graphs)}
#print(gs)

repls = 5
n = 100
sigmas = [math.ceil(math.log(n)), 7, 10, 25, 50]
max_iter = 100

print('p1 p2 c1 c2 dominates iter repl n max.sigma instance mut')

for gname, g in gs.items():
    for sigma in sigmas:
        mutators = {
            'usgs': USGMutator(g, sigma = sigma, fixed = True),
            'sgsi': SGSImprovedMutator(g, sigma = sigma, fixed = False),
            'sgss': SGSSimplifiedMutator(g, sigma = sigma, fixed = False)
        }
        for mutname, mut in mutators.items():
            for repl in range(1, repls + 1):
                x = RandomSpanningTreeGenerator.sample(g)
                fitness_x = x.get_sum_of_edge_costs()

                for it in range(max_iter):
                    y = mut.create(x)
                    fitness_y = y.get_sum_of_edge_costs()
                    print(f'{fitness_x[0]} {fitness_x[1]} {fitness_y[0]} {fitness_y[1]} TRUE {it} {repl} {n} {sigma} {gname} {mutname}')
                    x = y
                    fitness_x = fitness_y
