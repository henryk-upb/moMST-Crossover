#!/usr/bin/env python
'''
Extend a finished expyrimenter registry by further values of one factor and run the new jobs.

New rows (the full factorial of the new values with all existing values of the other factors)
are appended to design.csv with consecutive job IDs; existing rows and jobs are left untouched.
The new jobs are then run with the registry's runner (make_runner of the given module) in a
process pool. Rows that already exist are not added again, and jobs that already have a
runtime.csv are not run again, so the script can be restarted safely.

The registry's own Registry.add_jobs() is not used, since it rewrites design.csv from the
loaded job collection, which is incomplete if job pickles are missing.

Usage:
  python extend_registry.py <runner module> <registry path> <factor> <value> [<value> ...]
  e.g. python extend_registry.py hybrid_final_runner_expyrimenter results/approximations/ex15_hybrid_final p_crossover 0.6 0.8
'''

import csv
import importlib
import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUNNER = None


def run_job(job):
    jobid, params = job
    RUNNER(jobid, params)
    return jobid


def init_worker(module_name, registry_path):
    global RUNNER
    RUNNER = importlib.import_module(module_name).make_runner(registry_path)


if __name__ == "__main__":
    module_name, registry_path, factor, *values = sys.argv[1:]
    design_path = os.path.join(registry_path, "design.csv")
    with open(design_path, newline="") as f:
        rows = list(csv.DictReader(f))
    columns = list(rows[0].keys())
    others = [c for c in columns if c not in ("jobid", factor)]

    existing = {tuple(r[c] for c in columns[1:]) for r in rows}
    levels = {c: list(dict.fromkeys(r[c] for r in rows)) for c in others}
    next_id = max(int(r["jobid"]) for r in rows) + 1
    new_rows = []
    for combo in itertools.product(values, *(levels[c] for c in others)):
        params = dict(zip([factor] + others, combo))
        row = {c: params[c] for c in columns[1:]}
        if tuple(row.values()) in existing:
            continue
        new_rows.append({"jobid": str(next_id), **row})
        next_id += 1

    if new_rows:
        with open(design_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=columns).writerows(new_rows)
    print(f"{design_path}: {len(new_rows)} rows appended", flush=True)

    with open(design_path, newline="") as f:
        rows = list(csv.DictReader(f))
    todo = [(int(r["jobid"]), {c: r[c] for c in columns[1:]}) for r in rows
            if r[factor] in values and not os.path.exists(
                os.path.join(registry_path, "experiments", r["jobid"], "runtime.csv"))]
    print(f"{len(todo)} jobs to run", flush=True)

    with Pool(os.cpu_count() - 1, initializer=init_worker, initargs=(module_name, registry_path)) as pool:
        for k, jobid in enumerate(pool.imap_unordered(run_job, todo), 1):
            if k % 50 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} done (last job {jobid})", flush=True)
