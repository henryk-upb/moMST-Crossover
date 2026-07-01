from Graph.Graph import Graph
from Graph.algorithms.MST import Kruskal, Prim
from Graph.algorithms.Scalarisation import MOMSTWeightedSum
from EA.utils.EdgeSampler import do_fast_nondominated_sorting
import os
import sys
import csv

input_folder = "instances/100/"
output_folder = "results/approximations/wsum/"

path_to_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
print(path_to_files)

n_lambdas = 1000
lambdas = [(k / n_lambdas, 1 - k / n_lambdas) for k in range(n_lambdas + 1)]

for path_to_file in path_to_files:
    bn = os.path.basename(path_to_file)
    graph = Graph.import_grapherator_format(path_to_file)
    result = MOMSTWeightedSum(graph, lambdas = lambdas, solver = Kruskal(graph)).solve()
    pf = result.get_approximation_front()

    print(f"Processing file {bn} ...")

    output_file = os.path.join(output_folder, bn.replace(".graph", ".csv"))
    # print(output_file)
    # sys.exit(1)
    with open(output_file, 'w', newline = '') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',')
        csv_writer.writerow(["c1", "c2"])
        for x, y in pf:
            csv_writer.writerow([x, y])

