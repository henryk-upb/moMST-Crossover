import random
import csv
import itertools
import os
import math
import sys

'''
FINAL EXPERIMENTS
===

We want to vary:
* All 40 ECJ instances with n = 100 nodes (each 10 of classes C1 to C4)
* 5 Operators:
*   - USGS with edge filtering (and log^2(n) sigma)
*.  - USGS with pre-sorting (and log^2(n) sigma)
*.  - SGS simplified with pre-sorting (and log^2(n) sigma)
*.  - IF-L with pre-sorting (and linear sigma)
*.  - IF-G with pre-sorting (and linear sigma)
* L = n (= 100) equidistant weights for the pre-sorting
* Termination criterion: 100n (the 10% limit of the ECJ paper)
* Independent runs: 30

SETUP
===

I guess there is no need for highly parallel experimentation.
However, we should go for local multi-core parallelization
(subprocess to the rescue!)
To this end, instead of nested loops, we should lazily iterate
over a collection of tuples/lists (n, r, R, L, type, ...) and
call a run_experiment function which also deals with output.
'''

setup_file = "parameters/ex04.csv"

if __name__ == "__main__":
  # reproducibility
  random.seed(1)

  exp_param_names = ["jobid", "instance", "mutator", "L", "repl", "seed"]

  # instances
  path_to_instances = "instances/50"
  instances = [os.path.join(path_to_instances, file) for file in os.listdir(path_to_instances)]

  # Mutation operators
  mutators = ["USGS-F", "USGS-PRE", "SGS", "IF-L", "IF-G"]

  # number of weights for pre-sorting
  # TODO: run experiments for different L for USGS-Pre to check how it affects the running time
  L = 100 # equals the no. of nodes

  # independent reps
  repls = list(range(1, 11))

  exp_design = itertools.product(instances, mutators, [L], repls)

  with open(setup_file, "w", newline = '') as file:
    writer = csv.writer(file, delimiter = ",")
    writer.writerow(exp_param_names)
    for exp_id, exp_params in enumerate(exp_design):
      random.seed(int(exp_params[3]))
      exp_seed = random.randint(1, 1000000)
      writer.writerow([exp_id + 1, exp_params[0], exp_params[1], exp_params[2], exp_params[3], exp_seed])
