from Graph.Graph import Graph
from Graph.algorithms.CC import CC
from Graph.generators.GraphGenerator import GraphGenerator
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
import math
import timeit

# run
# python -OO ex02_usgs_bounds.py > results/ex02_usgs_bounds.csv

def half(n):
    """Set threshold to n/2."""
    return n / 2

def logsquared(n):
    return math.ceil(math.log2(n) * math.log2(n))

def get_bounds_on_relevant_edges_on_complete_graphs(n: int, k: int):
    """Calculate lower and upper bounds for the number of relevant edges for USGS variants."""
    LB = (k - 1) * (n - k + 1)
    UB = (k - 1) * n * n / (2 * k)
    return LB, UB

def get_number_of_relevant_edges(counts: list[int]) -> int:
    """Calculate actual number of relevant edges for a given partition of the nodes."""
    n = sum(counts)
    m_complete = n * (n - 1) / 2
    m_ccs = [ni * (ni - 1) / 2 for ni in counts]
    return m_complete - sum(m_ccs)

# Number of nodes
ns = [100 * i for i in range(1, 11)]#range(2, 21, 2)]

funs = {'log': math.log, 'sqrt': math.sqrt, 'logsquared': logsquared, 'half': half}
R = 30

print('n sigma.fun sigma repl LB UB value')

for n in ns:
    g = GraphGenerator.generate_kc_random(n, 1)
    for fun_name, fun in funs.items():
        sigma = math.ceil(fun(n))

        for repl in range(1, R + 1):
            # random spanning tree
            tree = RandomSpanningTreeGenerator.sample(g)

            # drop exactly sigma edges randomly
            forest = Graph.get_subgraph_by_deleting_random_edges(tree, sigma)

            # get number of connected components of forest
            forest_ccs = CC(forest)
            k = forest_ccs.count()

            # calculate theoretical bounds for the number of relevant edges
            LB, UB = get_bounds_on_relevant_edges_on_complete_graphs(n, k)

            # calculate actual number of relevant edges
            no_relevant_edges = get_number_of_relevant_edges(forest_ccs.get_counts())

            print(f'{n} "{fun_name}" {sigma} {repl} {LB} {UB} {no_relevant_edges}')

