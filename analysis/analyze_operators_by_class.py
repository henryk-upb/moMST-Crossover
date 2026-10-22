#!/usr/bin/env python
'''
Operator comparison per instance class, pooled over ALL instances of the class.

For every instance the hypervolume of each run is computed with the instance's own
reference point (compute_cross_instance_table() in analyze_hypervolume.py). Since raw
HV is not comparable between instances, each (instance, repl) block is reduced to
  * rank          (1 = highest HV among the operators of that block)
  * relative_pct  (100 * hv / best hv of that block)
and these are aggregated per instance class C1-C4. Friedman and pairwise Wilcoxon
signed-rank tests (Holm-corrected) run per class, blocked over (instance, repl).

Only instances whose jobs have ALL finished are used, so the script can also be run
on a registry that is still being filled.

Usage:
  python analysis/analyze_operators_by_class.py results/approximations/ex10_pure_cx_vs_usgs_pre
'''

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analyze_hypervolume import (
    read_params, compute_cross_instance_table, infer_instance_class, run_statistics, archive_generic_run,
)

# fixed operator order and colours (colour follows the operator, not its rank)
OPERATOR_ORDER = ['UNION', 'INTER', 'XOR', 'USGS-PRE']
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def complete_instances(params_rows, registry_path):
    '''Instances for which every job has written its pf.csv.'''
    done = {}
    for row in params_rows:
        ok = os.path.exists(os.path.join(registry_path, 'experiments', row['jobid'], 'pf.csv'))
        done[row['instance']] = done.get(row['instance'], True) and ok
    return sorted(i for i, ok in done.items() if ok)


def plot_relative_hv(df, operators, out_path):
    classes = sorted(df['class'].unique())
    if len(operators) <= len(PALETTE):
        colors = dict(zip(operators, PALETTE))
    else:
        # more groups than fixed colours (e.g. ordered p_crossover values): one-hue light-to-dark ramp
        ramp = matplotlib.colormaps['Blues']
        colors = {op: ramp(0.35 + 0.6 * i / (len(operators) - 1)) for i, op in enumerate(operators)}
    fig, axes = plt.subplots(1, len(classes), figsize = (3.1 * len(classes), 3.9), facecolor = SURFACE)
    for ax, cls in zip(axes, classes):
        sub = df[df['class'] == cls]
        data = [sub.loc[sub['algorithm'] == op, 'relative_pct'] for op in operators]
        bp = ax.boxplot(data, widths = 0.55, patch_artist = True, showfliers = True,
                         medianprops = {'color': INK, 'linewidth': 1.4},
                         whiskerprops = {'color': MUTED, 'linewidth': 1},
                         capprops = {'color': MUTED, 'linewidth': 1},
                         flierprops = {'marker': 'o', 'markersize': 2.5, 'markerfacecolor': MUTED,
                                       'markeredgecolor': 'none', 'alpha': 0.5})
        for patch, op in zip(bp['boxes'], operators):
            patch.set_facecolor(colors[op])
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1)
        n_inst = sub['instance'].nunique()
        ax.set_title(f'{cls}  ({n_inst} instances)', fontsize = 10, color = INK, loc = 'left')
        ax.set_xticks(range(1, len(operators) + 1), operators, fontsize = 8, color = INK)
        ax.tick_params(axis = 'y', labelsize = 8, colors = MUTED, length = 0)
        ax.tick_params(axis = 'x', length = 0)
        ax.yaxis.grid(True, color = GRID, linewidth = 0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    axes[0].set_ylabel('HV relative to best run of the block (%)', fontsize = 9, color = MUTED)
    fig.suptitle('Final hypervolume per operator and instance class (y-axis differs per panel)',
                  fontsize = 11, color = INK, x = 0.01, ha = 'left')
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(out_path, dpi = 200, facecolor = SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry')
    parser.add_argument('--algo-col', default = 'operator')
    parser.add_argument('--out', default = None, help = 'default: figures/operators_by_class/<registry name>')
    parser.add_argument('--archive-dir', default = 'analysis_log',
                         help = 'base directory of the permanent run archive (default: analysis_log)')
    parser.add_argument('--no-archive', action = 'store_true', help = 'do not archive this run (e.g. test runs)')
    parser.add_argument('--note', default = None, help = 'short note, ends up in summary.md and CHANGELOG.md')
    args = parser.parse_args()

    experiment_label = os.path.basename(os.path.normpath(args.registry))
    out_dir = args.out or os.path.join('figures', 'operators_by_class', experiment_label)
    os.makedirs(out_dir, exist_ok = True)

    params_rows = read_params(os.path.join(args.registry, 'design.csv'))
    instances = complete_instances(params_rows, args.registry)
    rows = [r for r in params_rows if r['instance'] in instances]
    print(f'{len(instances)} of {len(set(r["instance"] for r in params_rows))} instances complete')

    df = compute_cross_instance_table(rows, args.registry, args.algo_col.split(','))
    df['class'] = df['instance'].map(infer_instance_class)
    operators = [op for op in OPERATOR_ORDER if op in set(df['algorithm'])] + \
                sorted(set(df['algorithm']) - set(OPERATOR_ORDER))

    summary = df.groupby(['class', 'algorithm']).agg(
        mean_rank = ('rank', 'mean'), mean_rel_hv = ('relative_pct', 'mean'),
        median_rel_hv = ('relative_pct', 'median'), std_rel_hv = ('relative_pct', 'std'),
        wins = ('rank', lambda r: int((r == 1).sum())), n_blocks = ('rank', 'size')).reset_index()
    summary['algorithm'] = pd.Categorical(summary['algorithm'], operators, ordered = True)
    summary = summary.sort_values(['class', 'algorithm'])

    lines = ['=== Summary per class (rank 1 = best; relative HV in % of the best run per block) ===',
             summary.round(3).to_string(index = False)]
    for cls in sorted(df['class'].unique()):
        sub = df[df['class'] == cls]
        report, _ = run_statistics(sub, block_cols = ['instance', 'repl'], value_col = 'hv',
                                    block_label = 'Bloecke (Instanz, Repl)')
        lines.append(f'\n################ {cls} ################\n{report}')
    text = '\n'.join(lines)
    print(text)

    df.to_csv(os.path.join(out_dir, 'hv_by_block.csv'), index = False)
    summary.to_csv(os.path.join(out_dir, 'summary_by_class.csv'), index = False)
    with open(os.path.join(out_dir, 'statistics.txt'), 'w') as f:
        f.write(text + '\n')
    plot_relative_hv(df, operators, os.path.join(out_dir, 'relative_hv_by_class.png'))
    print(f'\nwritten to {out_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, 'operators_by_class',
            title = 'Operator comparison per instance class', note = args.note,
            copy_paths = [os.path.join(out_dir, f) for f in
                          ('relative_hv_by_class.png', 'summary_by_class.csv', 'statistics.txt', 'hv_by_block.csv')],
            changelog_header = "| Datum | Commit | Klasse | Operator | Mean Rank | Mean rel. HV (%) | Wins | n | Notiz |\n"
                               "|---|---|---|---|---|---|---|---|---|\n",
            row_template = "| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {algorithm} | "
                           "{mean_rank:.2f} | {mean_rel_hv:.2f} | {wins} | {n_blocks} | {note} |",
            changelog_rows = summary.to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
