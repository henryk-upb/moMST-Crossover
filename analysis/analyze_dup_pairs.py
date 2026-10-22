#!/usr/bin/env python
'''
Identical parents in crossover over the generations (dup_pairs.csv written by
analysis_log/dup_pairs_pure_cx/<entry>/replay_dup_pairs.py, one row per run and generation).

Figures (compact thesis layout of make_thesis_figures_operators.py, one panel per class):

  * dup-pairs:      share of crossover calls whose two parents are the same tree, pooled over
                    all runs of a class and binned into windows of --bin generations; dotted
                    line at 1/mu, the share expected from drawing the same population slot twice
  * distinct-trees: mean number of distinct trees (edge sets) in the parent population
                    (band: quartiles), dotted line at mu

Usage:
  python analysis/analyze_dup_pairs.py analysis_log/dup_pairs_pure_cx/<entry>/dup_pairs.csv \\
      --note "..."
'''

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator

from analyze_hypervolume import archive_generic_run
from analyze_edge_diversity import set_gen_axis
from make_thesis_figures_operators import new_grid, finish, styles, MUTED

MU = 100


def load(path):
    df = pd.read_csv(path)
    df = df[df['gen'] > 0].copy()
    df['class'] = 'C' + df['instance'].str.extract(r'CLASS(\d)')[0]
    df['algorithm'] = df['setting']
    df['identical'] = df['same_index'] + df['same_tree']
    return df


def plot_dup_pairs(df, bin_size, out_base):
    classes = sorted(df['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(df['algorithm'].unique())
    handles = {}
    # window k covers generations (k-1)*bin+1 .. k*bin and is drawn at its last generation
    df = df.assign(window = ((df['gen'] - 1) // bin_size + 1) * bin_size)
    pooled = df.groupby(['class', 'algorithm', 'window'])[['identical', 'cx_calls']].sum().reset_index()
    pooled['share'] = 100 * pooled['identical'] / pooled['cx_calls']
    for ax, cls in zip(axes.flat, classes):
        sub = pooled[pooled['class'] == cls]
        for op in order:
            color, marker = style[op]
            line = sub[sub['algorithm'] == op].sort_values('window')
            handles[op], = ax.plot(line['window'], line['share'], color = color, marker = marker,
                                   markersize = 2, linewidth = 0.9, label = op)
        ax.axhline(100 / MU, color = MUTED, linestyle = ':', linewidth = 0.8)
        ax.set_ylim(0, None)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, int(df['gen'].max()))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Identical parents (%)', labelpad = 1)
    finish(fig, axes, [handles[op] for op in order], order, out_base)


def plot_distinct_trees(df, out_base):
    classes = sorted(df['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(df['algorithm'].unique())
    handles = {}
    agg = df.groupby(['class', 'algorithm', 'gen'])['distinct_parents'].agg(
        mean = 'mean', q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75)).reset_index()
    for ax, cls in zip(axes.flat, classes):
        sub = agg[agg['class'] == cls]
        for op in order:
            color, marker = style[op]
            line = sub[sub['algorithm'] == op].sort_values('gen')
            ax.fill_between(line['gen'], line['q25'], line['q75'], color = color, alpha = 0.15, linewidth = 0)
            handles[op], = ax.plot(line['gen'], line['mean'], color = color, linewidth = 0.9, label = op)
        ax.axhline(MU, color = MUTED, linestyle = ':', linewidth = 0.8)
        ax.set_ylim(0, MU + 5)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, int(df['gen'].max()))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Distinct trees', labelpad = 1)
    finish(fig, axes, [handles[op] for op in order], order, out_base)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', help = 'dup_pairs.csv from replay_dup_pairs.py')
    parser.add_argument('--label', default = 'dup_pairs_pure_cx')
    parser.add_argument('--bin', type = int, default = 5, help = 'generations per window in the dup-pairs figure')
    parser.add_argument('--out-dir', default = 'figures/thesis')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    df = load(args.csv)
    out_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(out_dir, exist_ok = True)
    base = os.path.join(out_dir, args.label)
    plot_dup_pairs(df, args.bin, f'{base}-dup-pairs')
    plot_distinct_trees(df, f'{base}-distinct-trees')
    outputs = [f'{base}-{name}.{ext}' for name in ('dup-pairs', 'distinct-trees') for ext in ('pdf', 'png')]
    print('\n'.join(outputs))

    entry_dir = archive_generic_run(
        args.archive_dir, args.label, 'dup_pairs_figures',
        title = 'Identische Eltern im Crossover', note = args.note, copy_paths = outputs,
        changelog_header = "| Datum | Commit | Basis | Notiz |\n|---|---|---|---|\n",
        row_template = "| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {basis} | {note} |",
        changelog_rows = [{'basis': os.path.relpath(args.csv, args.archive_dir)}])
    print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
