#!/usr/bin/env python
'''
Edge diversity of the population over the generations (registry written by
edge_diversity_runner_expyrimenter.py, one edge_diversity.csv per job).

Figures (compact thesis layout of make_thesis_figures_operators.py, one panel per class,
over ALL instances x repls of the class):

  * edge-diversity:   mean number of distinct edges in the population (band: quartiles),
                      linear y-axis with evenly spaced ticks, one point per generation up
                      to --max-gen; dotted line at n - 1 = the value of a population
                      consisting of copies of a single tree; each panel lists the mean
                      edge count per operator at the end of the full run
  * carrier-survival: share of trees in the selection pool R_t = P_{t-1} u Q_t that survive
                      survival selection, split into trees holding at least one edge no
                      other tree of R_t has ("carriers", solid) and all other trees (dashed);
                      pooled over all runs, shown while >= --min-carriers carriers exist,
                      up to --max-gen

Plus edge-diversity-summary.csv / .md with the numbers per class and operator.

Usage:
  python analysis/analyze_edge_diversity.py results/approximations/ex11_edge_diversity_pure_cx \\
      --note "..." [--thesis-dir ../moMST-Crossover-LaTeX/figures]

  # joint diagram: pure CX + USGS-PRE reference
  python analysis/analyze_edge_diversity.py results/approximations/ex11_edge_diversity_pure_cx \\
      results/approximations/ex11_edge_diversity_usgs_pre --label ex11_edge_diversity_cx_vs_usgs_pre
'''

import argparse
import os
import shutil

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, ScalarFormatter

from analyze_hypervolume import read_params, infer_instance_class, archive_generic_run
from make_thesis_figures_operators import new_grid, finish, styles, MUTED, FONT

SNAPSHOTS = [0, 10, 20, 50, 100]


def load(registry, algo_col):
    frames = []
    for row in read_params(os.path.join(registry, 'design.csv')):
        path = os.path.join(registry, 'experiments', row['jobid'], 'edge_diversity.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['algorithm'] = row[algo_col]
        df['class'] = infer_instance_class(row['instance'])
        df['instance'] = row['instance']
        df['repl'] = int(row['repl'])
        frames.append(df)
    if not frames:
        raise SystemExit(f'no edge_diversity.csv found in {registry}')
    return pd.concat(frames, ignore_index = True)


def aggregate(df):
    g = df.groupby(['class', 'algorithm', 'gen'])
    agg = g['distinct_edges'].agg(mean = 'mean', q25 = lambda s: s.quantile(0.25),
                                  q75 = lambda s: s.quantile(0.75)).reset_index()
    pooled = g[['carriers', 'carriers_survived', 'others_survived', 'pool_size']].sum().reset_index()
    pooled['others'] = pooled['pool_size'] - pooled['carriers']
    pooled['carrier_survival'] = 100 * pooled['carriers_survived'] / pooled['carriers'].where(pooled['carriers'] > 0)
    pooled['other_survival'] = 100 * pooled['others_survived'] / pooled['others'].where(pooled['others'] > 0)
    return agg, pooled


def set_gen_axis(ax, max_gen):
    '''Generation axis from 0; tick labels every 20 generations for the full run, every
    5 (minor tick every generation) for a zoom on the first generations, where a data
    point is shown for every generation.'''
    step = 20 if max_gen > 40 else 5
    ax.set_xticks(range(0, max_gen + 1, step))
    if step == 5:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, max_gen + 0.15 * step)


def plot_edge_diversity(agg, n_minus_1, max_gen, out_base):
    classes = sorted(agg['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(agg['algorithm'].unique())
    handles = {}
    y_max = 500 * -(-agg['q75'].max() // 500)   # round up to the next multiple of 500
    for ax, cls in zip(axes.flat, classes):
        sub = agg[agg['class'] == cls]
        last_gen = int(sub['gen'].max())
        shown = sub[sub['gen'] <= max_gen]
        for op in order:
            color, marker = style[op]
            line = shown[shown['algorithm'] == op].sort_values('gen')
            ax.fill_between(line['gen'], line['q25'], line['q75'], color = color, alpha = 0.15, linewidth = 0)
            handles[op], = ax.plot(line['gen'], line['mean'], color = color, marker = marker,
                                   markersize = 2, linewidth = 0.9, label = op)
        ax.axhline(n_minus_1, color = MUTED, linestyle = ':', linewidth = 0.8)

        # mean edge count at the end of the full run, one line per operator
        box = dict(boxstyle = 'square,pad=0.25', facecolor = 'white', edgecolor = 'none')
        ax.text(0.97, 0.96, f'Distinct edges after\n{last_gen} generations (mean)', transform = ax.transAxes,
                ha = 'right', va = 'top', fontsize = FONT - 2, color = MUTED, linespacing = 1.1, bbox = box)
        for i, op in enumerate(order):
            final = sub[(sub['algorithm'] == op) & (sub['gen'] == last_gen)]['mean'].iloc[0]
            ax.text(0.97, 0.78 - 0.09 * i, f'{op}: {final:.0f}', transform = ax.transAxes,
                    ha = 'right', va = 'top', fontsize = FONT - 2, color = style[op][0], bbox = box)

        # linear axis with evenly spaced, fully written-out ticks (no 1e3 offset)
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset = False))
        ax.yaxis.get_major_formatter().set_scientific(False)
        set_gen_axis(ax, min(max_gen, last_gen))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Distinct edges', labelpad = 1)
    finish(fig, axes, [handles[op] for op in order], order, out_base)


def plot_carrier_survival(pooled, min_carriers, max_gen, out_base):
    classes = sorted(pooled['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(pooled['algorithm'].unique())
    for ax, cls in zip(axes.flat, classes):
        sub = pooled[(pooled['class'] == cls) & (pooled['gen'] > 0) & (pooled['gen'] <= max_gen)]
        for op in order:
            color, _ = style[op]
            line = sub[sub['algorithm'] == op].sort_values('gen')
            ax.plot(line['gen'], line['other_survival'], color = color, linestyle = '--', linewidth = 0.9)
            carr = line[line['carriers'] >= min_carriers]
            ax.plot(carr['gen'], carr['carrier_survival'], color = color, linewidth = 1, marker = 'o', markersize = 1.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        set_gen_axis(ax, int(sub['gen'].max()))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Survival rate (%)', labelpad = 1)
    # two legend rows: operators (colour), then carriers / others (line style)
    kw = dict(loc = 'lower center', frameon = False, handlelength = 1.8, columnspacing = 1.0, borderaxespad = 0.1)
    fig.legend([Line2D([], [], color = style[op][0], linewidth = 1.2) for op in order], order,
               ncol = len(order), bbox_to_anchor = (0.5, 0.045), **kw)
    fig.legend([Line2D([], [], color = MUTED, linewidth = 1), Line2D([], [], color = MUTED, linestyle = '--', linewidth = 0.9)],
               ['carriers', 'others'], ncol = 2, bbox_to_anchor = (0.5, 0.0), **kw)
    fig.tight_layout(rect = (0, 0.1, 1, 1), h_pad = 0.8, w_pad = 0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def summarise(df, pooled):
    rows = []
    for (cls, op), sub in df.groupby(['class', 'algorithm']):
        means = sub.groupby('gen')['distinct_edges'].mean()
        p = pooled[(pooled['class'] == cls) & (pooled['algorithm'] == op) & (pooled['gen'] > 0)]
        last = sub[sub['gen'] == sub['gen'].max()]
        row = {'class': cls, 'operator': op, 'runs': sub.groupby(['instance', 'repl']).ngroups}
        row.update({f'edges_g{g}': round(means[g], 1) for g in SNAPSHOTS if g in means.index})
        row.update({
            'retained_pct': round(100 * means.iloc[-1] / means.iloc[0], 2),
            'max_new_edges': int(sub['new_edges'].max()),
            'mean_new_edges': round(sub['new_edges'].mean(), 2),
            # edges brought in by the offspring over the whole run (per run, mean); an edge that is
            # lost and reintroduced later counts again, so this counts introductions, not unique edges
            'introduced_total': round(sub.groupby(['instance', 'repl'])['new_edges'].sum().mean(), 1),
            'trees_final': round(last['distinct_trees'].mean(), 1),
            'carrier_survival_pct': round(100 * p['carriers_survived'].sum() / p['carriers'].sum(), 1),
            'other_survival_pct': round(100 * p['others_survived'].sum() / p['others'].sum(), 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry', nargs = '+',
                        help = 'one or more registries; several are combined into joint diagrams '
                               '(e.g. the pure-CX registry plus the USGS-PRE reference)')
    parser.add_argument('--label', default = None,
                        help = 'experiment label for output folder and archive (default: first registry name)')
    parser.add_argument('--algo-col', default = 'operator')
    parser.add_argument('--n', type = int, default = 100, help = 'number of nodes (reference line at n - 1)')
    parser.add_argument('--min-carriers', type = int, default = 20,
                        help = 'carrier survival is drawn only for generations with at least this many '
                               'carriers pooled over all runs of the panel (default: 20)')
    parser.add_argument('--max-gen', type = int, default = 20,
                        help = 'last generation shown in the diagrams (default: 20); the summary '
                               'and the CSVs always cover the full run')
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<registry name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy the PDFs into this directory')
    parser.add_argument('--prefix', default = None, help = 'file name prefix (default: registry name)')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = None)
    args = parser.parse_args()

    experiment_label = args.label or os.path.basename(os.path.normpath(args.registry[0]))
    prefix = args.prefix or experiment_label
    out_dir = args.out or os.path.join('figures', 'thesis', experiment_label)
    os.makedirs(out_dir, exist_ok = True)

    df = pd.concat([load(r, args.algo_col) for r in args.registry], ignore_index = True)
    agg, pooled = aggregate(df)

    div_base = os.path.join(out_dir, f'{prefix}-edge-diversity')
    plot_edge_diversity(agg, args.n - 1, args.max_gen, div_base)
    agg.to_csv(f'{div_base}.csv', index = False)

    surv_base = os.path.join(out_dir, f'{prefix}-carrier-survival')
    plot_carrier_survival(pooled, args.min_carriers, args.max_gen, surv_base)
    pooled.to_csv(f'{surv_base}.csv', index = False)

    summary = summarise(df, pooled)
    sum_base = os.path.join(out_dir, f'{prefix}-edge-diversity-summary')
    summary.to_csv(f'{sum_base}.csv', index = False)
    with open(f'{sum_base}.md', 'w') as f:
        cols = list(summary.columns)
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for rec in summary.itertuples(index = False):
            f.write('| ' + ' | '.join(str(v) for v in rec) + ' |\n')
    print(summary.to_string(index = False))

    outputs = [f'{b}.{e}' for b in (div_base, surv_base) for e in ('pdf', 'png', 'csv')] \
              + [f'{sum_base}.csv', f'{sum_base}.md']
    print('written:', *outputs, sep = '\n  ')
    if args.thesis_dir:
        for b in (div_base, surv_base):
            shutil.copy(f'{b}.pdf', args.thesis_dir)
        print(f'PDFs copied to {args.thesis_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, 'edge_diversity',
            title = 'Edge diversity of the population', note = args.note, copy_paths = outputs,
            changelog_header = "| Datum | Commit | Klasse | Operator | Edges g0 | Edges g100 | Retained (%) "
                               "| Carrier surv. (%) | Other surv. (%) | Runs | Notiz |\n"
                               "|---|---|---|---|---|---|---|---|---|---|---|\n",
            row_template = "| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {operator} "
                           "| {edges_g0} | {edges_g100} | {retained_pct} | {carrier_survival_pct} "
                           "| {other_survival_pct} | {runs} | {note} |",
            changelog_rows = summary.to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
