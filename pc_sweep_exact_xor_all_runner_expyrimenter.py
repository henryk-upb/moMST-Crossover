#!/usr/bin/env python
import os
import sys

import pandas as pd

# expyrimenter is a git submodule under external/expyrimenter/ (see
# .gitmodules) -- it is its own top-level "Experiments" package, so it
# needs its own sys.path entry rather than living at the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "expyrimenter"))
from Experiments.Backend import JoblibRunnerBackend

from pc_sweep_exact_xor_runner_expyrimenter import make_runner, build_registry, PC_VALUES

'''
P_CROSSOVER SWEEP, EXACT XOR, ALL INSTANCES

Same runner as ex12 (pc_sweep_exact_xor_runner_expyrimenter.py: exact XOR +
USGS-PRE, Mode 1, p_crossover in {0.0, 0.1, ..., 1.0}, same seeds), but on all
40 instances (C1-C4, 10 each) instead of only the "_6" instance per class, so the
sweep can be reported as a class mean over all instances. The "_6" jobs reproduce
ex12 exactly.
'''

REGISTRY_PATH = "results/approximations/ex17_pc_sweep_exact_xor_all"

if __name__ == "__main__":
    path_to_instances = "instances/100"
    instances = sorted((os.path.join(path_to_instances, f) for f in os.listdir(path_to_instances)),
                       key=lambda p: (int(p.rsplit("_", 1)[1].split(".")[0]), p))
    repls = list(range(1, 11))

    reg = build_registry(REGISTRY_PATH, instances, PC_VALUES, repls,
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
