#!/usr/bin/env python
'''
Resume an interrupted expyrimenter registry without overwriting finished jobs.

The *_runner_expyrimenter.py scripts build their registry with overwrite=True,
so running them again would delete all finished jobs. This script instead loads
the existing registry and runs only the jobs that have no runtime.csv yet (a
job writes its outputs only at the very end, so a missing runtime.csv means the
job did not finish).

Usage:
  python resume_registry.py <runner module> <registry path>
  e.g. python resume_registry.py survival_variants_runner_expyrimenter results/approximations/ex14_survival_variants
'''

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "expyrimenter"))
from Experiments.Registry import Registry
from Experiments.Backend import JoblibRunnerBackend

if __name__ == "__main__":
    module_name, registry_path = sys.argv[1], sys.argv[2]
    module = importlib.import_module(module_name)

    reg = Registry.load(registry_path, readonly=False)
    reg.set_backend(JoblibRunnerBackend(ncores=os.cpu_count() - 1))
    missing = [j for j in reg._get_all_jobids()
               if not os.path.exists(os.path.join(registry_path, "experiments", str(j), "runtime.csv"))]
    print(f"{registry_path}: {len(missing)} of {reg.size()} jobs missing", flush=True)

    if missing:
        reg.run(module.make_runner(registry_path), jobids=missing, batchsize=3 * (os.cpu_count() - 1))

    still_missing = [j for j in reg._get_all_jobids()
                     if not os.path.exists(os.path.join(registry_path, "experiments", str(j), "runtime.csv"))]
    print(f"{registry_path}: {len(still_missing)} jobs still missing", flush=True)
