#!/usr/bin/env python
'''
Runtime ECDF ("how fast are HV targets reached"), as in the COCO/BBOB benchmarking platform,
for any registry whose runs logged their population (trajectory.csv, log_every = 10).

Per instance, one reference point is shared by all runs and logged generations (final fronts +
trajectory snapshots, as determine_reference_point() in analyze_hypervolume.py). The best HV
of the instance is the highest HV any run reached at any logged generation. For every run and
every target tau (fraction of that best HV), the first-hitting generation is the first logged
generation at which the run's HV is >= tau * best HV (infinite if it never gets there).

The ECDF of an algorithm in a class at generation t is the fraction of all (run, target) pairs
of that class whose target was reached by generation t. It can only rise; a curve that rises
earlier reaches the targets faster, a curve that stays below 1 never reaches some targets.

Outputs (in --out, default figures/thesis/<registry name>/):
  * <prefix>-runtime-ecdf.pdf/png: one panel per class (compact thesis layout)
  * <prefix>-runtime-ecdf.csv:     the plotted ECDF values
  * <prefix>-first-hits.csv:       first-hitting generation per run and target
  * <prefix>-runtime-ecdf-summary.csv: per class and algorithm, the area under the ECDF
    (mean fraction of targets reached over all logged generations > 0) and the final fraction

Usage:
  python analysis/analyze_runtime_ecdf.py results/approximations/ex19_hybrid_vs_pure_runtime \\
      --algo-col operator --prefix hybrid-vs-pure --thesis-dir ../moMST-Crossover-LaTeX/figures --note "..."
'''

import argparse
import math
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator

from analyze_hypervolume import (read_params, collect_fronts, collect_trajectories, determine_reference_point,
                                 hypervolume_2d, infer_instance_class, archive_generic_run)
from analyze_edge_diversity import set_gen_axis
from make_thesis_figures_operators import new_grid, finish, styles

TARGETS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def first_hits(registry, algo_col, targets):
    rows = read_params(os.path.join(registry, 'design.csv'))
    records = []
    for instance in sorted({r['instance'] for r in rows}):
        fronts = collect_fronts(rows, registry, instance, [algo_col])
        traj = collect_trajectories(rows, registry, instance, [algo_col])
        ref = determine_reference_point(fronts, traj)
        cls = infer_instance_class(instance)
        curves = {(algo, repl): {g: hypervolume_2d(pts, ref) for g, pts in gens.items()}
                  for algo, repls in traj.items() for repl, gens in repls}
        best = max(hv for curve in curves.values() for hv in curve.values())
        for (algo, repl), curve in curves.items():
            gens = sorted(curve)
            for tau in targets:
                hit = next((g for g in gens if curve[g] >= tau * best), math.inf)
                records.append({'class': cls, 'instance': instance, 'algorithm': algo, 'repl': int(repl),
                                'target': tau, 'first_hit': hit})
        print(f'{cls} {os.path.basename(instance)}: {len(curves)} runs', flush = True)
    return pd.DataFrame(records)


def ecdf(hits, gens):
    rows = []
    for (cls, algo), sub in hits.groupby(['class', 'algorithm']):
        for g in gens:
            rows.append({'class': cls, 'algorithm': algo, 'gen': g, 'fraction': (sub['first_hit'] <= g).mean()})
    return pd.DataFrame(rows)


def plot(curves, out_base):
    classes = sorted(curves['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(curves['algorithm'].unique())
    handles = {}
    for ax, cls in zip(axes.flat, classes):
        sub = curves[curves['class'] == cls]
        for algo in order:
            color, marker = style[algo]
            line = sub[sub['algorithm'] == algo].sort_values('gen')
            handles[algo], = ax.step(line['gen'], line['fraction'], where = 'post', color = color,
                                     linewidth = 1, label = algo)
            ax.plot(line['gen'], line['fraction'], linestyle = 'none', marker = marker, markersize = 2,
                    color = color)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, int(sub['gen'].max()))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Targets reached', labelpad = 1)
    finish(fig, axes, [handles[a] for a in order], order, out_base)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path')
    parser.add_argument('--algo-col', default = 'operator')
    parser.add_argument('--prefix', default = None, help = 'file name prefix (default: registry name)')
    parser.add_argument('--targets', default = ','.join(str(t) for t in TARGETS),
                        help = 'comma-separated HV targets as fractions of the best HV of the instance')
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<registry name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy the PDF into this directory')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.registry_path))
    prefix = args.prefix or label
    out_dir = args.out or os.path.join('figures', 'thesis', label)
    os.makedirs(out_dir, exist_ok = True)
    targets = [float(t) for t in args.targets.split(',')]

    hits = first_hits(args.registry_path, args.algo_col, targets)
    gens = sorted(g for g in set(hits['first_hit']) if g != math.inf) or [0]
    gens = list(range(0, int(max(gens)) + 1, 10)) if gens else [0]
    curves = ecdf(hits, gens)
    summary = curves[curves['gen'] > 0].groupby(['class', 'algorithm'])['fraction'].agg(
        area = 'mean', final = 'last').reset_index()

    base = os.path.join(out_dir, f'{prefix}-runtime-ecdf')
    paths = [f'{base}.csv', os.path.join(out_dir, f'{prefix}-first-hits.csv'), f'{base}-summary.csv']
    curves.to_csv(paths[0], index = False)
    hits.to_csv(paths[1], index = False)
    summary.to_csv(paths[2], index = False)
    plot(curves, base)
    figures = [f'{base}.pdf', f'{base}.png']
    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index = False))

    if args.thesis_dir:
        shutil.copy(f'{base}.pdf', args.thesis_dir)
        print(f'copied PDF to {args.thesis_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'runtime_ecdf',
            title = 'Runtime ECDF (HV targets reached over the generations)', note = args.note,
            copy_paths = paths + figures,
            changelog_header = '| Datum | Commit | Klasse | Algorithmus | Fläche unter ECDF | Anteil am Ende | Notiz |\n'
                               '|---|---|---|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {algorithm} '
                           '| {area:.3f} | {final:.3f} | {note} |',
            changelog_rows = summary.to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
