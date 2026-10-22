#!/usr/bin/env python
'''
Compact thesis versions of two diagrams, drawn for display SIDE BY SIDE at half the
text width (instead of shrinking the full-page diagrams of analyze_pf_and_convergence.py
and analyze_indicator_progress.py, whose text becomes unreadable at that size):

  * pf_approximation: final front of one repetition per operator, one panel per class
  * hv_progress:      mean HV over the generations (band: quartiles), one panel per class
  * mean-ranks:       mean rank of all configurations (operator after 10 generations and
                      at the end of the run) per class, over ALL instances

The first two figures share the same size, panel grid, colours and a single legend below the
panels, so they line up when placed next to each other. One instance per class is
used (chosen like analyze_all.py --per-class --seed, or given via --instances).

Usage:
  python analysis/make_thesis_figures_operators.py results/approximations/ex10_pure_cx_vs_usgs_pre \\
      --seed 1 --thesis-dir ../moMST-Crossover-LaTeX/figures --prefix pure-cx-vs-usgs-pre
'''

import argparse
import math
import os
import random
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from analyze_hypervolume import read_params, collect_fronts, infer_instance_class, archive_generic_run
from analyze_mean_rankings import compute_indicator_table
from analyze_indicator_progress import compute_value_progress_by_class, add_generation_column

# fixed operator order, colour and marker (colour follows the operator; the marker is a
# second encoding so the operators stay distinguishable in greyscale print)
OPERATOR_ORDER = ['UNION', 'INTER', 'XOR', 'USGS-PRE']
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#eda100']  # 5th slot (yellow) validated with the dataviz palette checker
MARKERS = ['o', 's', '^', 'D', 'v']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

FIGSIZE = (3.5, 3.55)   # inches; shown at 0.49 * textwidth, so fonts shrink only slightly
FONT = 8


def choose_instances(params_rows, seed):
    '''One instance per class, same rule as analyze_all.py --per-class --seed.'''
    by_class = {}
    for row in params_rows:
        by_class.setdefault(infer_instance_class(row['instance']), set()).add(row['instance'])
    rng = random.Random(seed)
    return {cls: rng.choice(sorted(instances)) for cls, instances in sorted(by_class.items())}


def styles(operators):
    """Fixed colour and marker per group. UNION, INTER, XOR and USGS-PRE always get their own slot;
    a sampled variant (e.g. UNION-S) gets the slot of its exact operator unless that one is shown
    too; all other groups (e.g. a hybrid) take the following slots in sorted order."""
    operators = set(operators)

    def base(op):
        b = op[:-2] if op.endswith('-S') else op
        return b if b in OPERATOR_ORDER and (b == op or b not in operators) else None

    known = sorted((op for op in operators if base(op)), key = lambda o: OPERATOR_ORDER.index(base(o)))
    other = sorted(operators - set(known))
    if not known and len(other) > len(PALETTE):
        # more groups than fixed colours and no operator names (e.g. ordered p_crossover values):
        # one-hue light-to-dark ramp, so the order of the values is visible
        ramp = matplotlib.colormaps['Blues']
        return other, {op: (ramp(0.35 + 0.6 * i / (len(other) - 1)), MARKERS[i % len(MARKERS)])
                       for i, op in enumerate(other)}
    slot = {op: OPERATOR_ORDER.index(base(op)) for op in known}
    first_free = len(OPERATOR_ORDER) if known else 0
    slot.update({op: first_free + j for j, op in enumerate(other)})
    order = known + other
    return order, {op: (PALETTE[slot[op] % len(PALETTE)], MARKERS[slot[op] % len(MARKERS)]) for op in order}


def new_grid(classes, rows = 2, cols = 2, figsize = FIGSIZE):
    plt.rcParams.update({'font.size': FONT, 'axes.titlesize': FONT, 'axes.labelsize': FONT,
                         'xtick.labelsize': FONT - 1, 'ytick.labelsize': FONT - 1,
                         'legend.fontsize': FONT, 'axes.linewidth': 0.6})
    fig, axes = plt.subplots(rows, cols, figsize = figsize, squeeze = False)
    for ax, cls in zip(axes.flat, classes):
        ax.set_title(cls, loc = 'left', pad = 3, color = INK, fontweight = 'bold')
        ax.grid(True, color = GRID, linewidth = 0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length = 2, width = 0.5, colors = MUTED, pad = 2)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(MUTED)
        # short scientific offset ("1e6") instead of long tick labels
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (-2, 3), useMathText = True)
        ax.yaxis.get_offset_text().set_fontsize(FONT - 2)
        ax.yaxis.get_offset_text().set_color(MUTED)
    return fig, axes


def finish(fig, axes, handles, labels, out_base, ncol = None):
    # more than four entries do not fit into one row at half the text width: two rows
    # (a wide figure can pass ncol = len(labels) to keep a single row)
    if ncol is None:
        ncol = len(labels) if len(labels) <= 4 else math.ceil(len(labels) / 2)
    fig.legend(handles, labels, loc = 'lower center', ncol = ncol, frameon = False,
               handletextpad = 0.3, columnspacing = 1.0, borderaxespad = 0.1, markerscale = 1.8)
    rows = math.ceil(len(labels) / ncol)
    legend_height = 0.2 * rows / fig.get_size_inches()[1]   # ~0.2 inch per legend row
    fig.tight_layout(rect = (0, legend_height, 1, 1), h_pad = 0.8, w_pad = 0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def plot_pf_approximation(rows, registry, algo_cols, chosen, out_base):
    classes = sorted(chosen)
    fig, axes = new_grid(classes)
    handles = {}
    for ax, cls in zip(axes.flat, classes):
        by_algo = collect_fronts(rows, registry, chosen[cls], algo_cols)
        order, style = styles(by_algo.keys())
        for op in order:
            color, marker = style[op]
            repl, points = sorted(by_algo[op], key = lambda rp: int(rp[0]))[0]
            xs, ys = zip(*points)
            handles[op] = ax.scatter(xs, ys, s = 5, color = color, marker = marker, linewidths = 0, label = op)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[-1]:
        ax.set_xlabel('$c_1$', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('$c_2$', labelpad = 1)
    finish(fig, axes, [handles[op] for op in order], order, out_base)


def plot_hv_progress(agg, out_base):
    classes = sorted(agg['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(agg['algorithm'].unique())
    handles = {}
    for ax, cls in zip(axes.flat, classes):
        sub = agg[agg['class'] == cls]
        for op in order:
            color, marker = style[op]
            line = sub[sub['algorithm'] == op].sort_values('gen')
            ax.fill_between(line['gen'], line['q25'], line['q75'], color = color, alpha = 0.15, linewidth = 0)
            handles[op], = ax.plot(line['gen'], line['mean'], color = color, marker = marker,
                                   markersize = 2.5, linewidth = 1, label = op)
        # generation axis from 0, ticks every 20 (data points every 10)
        ax.set_xticks(range(0, int(sub['gen'].max()) + 1, 20))
        ax.set_xlim(0, sub['gen'].max() + 3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean HV', labelpad = 1)
    finish(fig, axes, [handles[op] for op in order], order, out_base)


def compute_mean_ranks(df):
    '''As analyze_mean_rankings.py: within each (instance, repl) block ALL configurations
    (operator x generation) are ranked against each other (1 = highest HV), then the
    ranks are averaged per (class, gen, algorithm).'''
    df = df.copy()
    df['rank'] = df.groupby(['instance', 'repl'])['value'].rank(ascending = False, method = 'average')
    return df.groupby(['class', 'gen', 'algorithm']).agg(
        mean_rank = ('rank', 'mean'), n_blocks = ('rank', 'size')).reset_index()


def plot_mean_ranks(ranks, out_base):
    '''One panel as in analyze_mean_rankings.py: classes on the x-axis, one line per
    configuration -- colour/marker = operator, solid = end of the run, dashed = early.'''
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': FONT, 'axes.titlesize': FONT, 'axes.labelsize': FONT,
                         'xtick.labelsize': FONT, 'ytick.labelsize': FONT - 1,
                         'legend.fontsize': FONT, 'axes.linewidth': 0.6})
    gens = sorted(ranks['gen'].unique(), reverse = True)
    linestyles = ['-', '--', ':', '-.']
    classes = sorted(ranks['class'].unique())
    order, style = styles(ranks['algorithm'].unique())
    n_configs = len(order) * len(gens)

    fig, ax = plt.subplots(figsize = (5.4, 2.7))
    for gi, gen in enumerate(gens):
        sub = ranks[ranks['gen'] == gen]
        for op in order:
            color, marker = style[op]
            line = sub[sub['algorithm'] == op].set_index('class').reindex(classes)
            ax.plot(range(len(classes)), line['mean_rank'], linestyle = linestyles[gi % len(linestyles)],
                    linewidth = 1.2, marker = marker, markersize = 4.5, color = color,
                    markeredgecolor = 'white', markeredgewidth = 0.5)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_xlim(-0.25, len(classes) - 0.75)
    ax.set_yticks(range(1, n_configs + 1))
    ax.set_ylim(n_configs + 0.4, 0.6)   # rank 1 at the top
    ax.set_xlabel('Instance class', labelpad = 2)
    ax.set_ylabel('Mean rank', labelpad = 2)
    ax.yaxis.grid(True, color = GRID, linewidth = 0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length = 2, width = 0.5, colors = MUTED, pad = 2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)

    handles = [Line2D([], [], color = style[op][0], marker = style[op][1], linestyle = '-', linewidth = 1.2,
                      markersize = 4.5, markeredgecolor = 'white', markeredgewidth = 0.5, label = op)
               for op in order]
    handles += [Line2D([], [], linestyle = 'none', label = '')]
    handles += [Line2D([], [], color = MUTED, linestyle = linestyles[gi % len(linestyles)], linewidth = 1.2,
                       label = f'after {gen} generations') for gi, gen in enumerate(gens)]
    ax.legend(handles = handles, loc = 'center left', bbox_to_anchor = (1.02, 0.5), frameon = False,
              handlelength = 2.4, labelspacing = 0.45)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300, bbox_inches = 'tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry')
    parser.add_argument('--algo-col', default = 'operator')
    parser.add_argument('--seed', type = int, default = 1, help = 'seed for choosing one instance per class (default: 1)')
    parser.add_argument('--instances', default = None, help = 'comma-separated instance paths instead of --seed')
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<registry name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy the PDFs into this directory')
    parser.add_argument('--prefix', default = None, help = 'file name prefix (default: registry name)')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = None)
    args = parser.parse_args()

    experiment_label = os.path.basename(os.path.normpath(args.registry))
    prefix = args.prefix or experiment_label
    out_dir = args.out or os.path.join('figures', 'thesis', experiment_label)
    os.makedirs(out_dir, exist_ok = True)
    algo_cols = args.algo_col.split(',')

    rows = read_params(os.path.join(args.registry, 'design.csv'))
    if args.instances:
        chosen = {infer_instance_class(i): i for i in args.instances.split(',')}
    else:
        chosen = choose_instances(rows, args.seed)
    for cls, inst in sorted(chosen.items()):
        print(f'{cls}: {inst}')

    pf_base = os.path.join(out_dir, f'{prefix}-pf-approximation')
    plot_pf_approximation(rows, args.registry, algo_cols, chosen, pf_base)

    df = compute_indicator_table([(args.registry, algo_cols)], list(range(10, 101, 10)),
                                  set(chosen.values()), ['HV'])
    agg = add_generation_column(compute_value_progress_by_class(df), df)
    hv_base = os.path.join(out_dir, f'{prefix}-hv-progress')
    plot_hv_progress(agg, hv_base)
    agg.to_csv(f'{hv_base}.csv', index = False)

    # mean ranks over ALL instances of the registry, after 10 % and 100 % of the generations
    df_all = compute_indicator_table([(args.registry, algo_cols)], [10, 100], None, ['HV'])
    ranks = compute_mean_ranks(df_all)
    rank_base = os.path.join(out_dir, f'{prefix}-mean-ranks')
    plot_mean_ranks(ranks, rank_base)
    ranks.to_csv(f'{rank_base}.csv', index = False)
    print(ranks.pivot_table(index = ['gen', 'class'], columns = 'algorithm', values = 'mean_rank').round(2).to_string())

    outputs = [f'{b}.{e}' for b in (pf_base, hv_base, rank_base) for e in ('pdf', 'png')] \
              + [f'{hv_base}.csv', f'{rank_base}.csv']
    print('written:', *outputs, sep = '\n  ')
    if args.thesis_dir:
        for b in (pf_base, hv_base, rank_base):
            shutil.copy(f'{b}.pdf', args.thesis_dir)
        print(f'PDFs copied to {args.thesis_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, 'thesis_figures',
            title = 'Thesis figures (compact)', note = args.note, copy_paths = outputs,
            changelog_header = "| Datum | Commit | Klasse | Instanz | Notiz |\n|---|---|---|---|---|\n",
            row_template = "| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {instance} | {note} |",
            changelog_rows = [{'class': c, 'instance': os.path.basename(i)} for c, i in sorted(chosen.items())])
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
