#!/usr/bin/env python
'''
Progress over the generations and runtime of several configurations on ALL instances of a
registry (e.g. ex19: UNION, INTER, XOR, USGS-PRE, HYBRID-0.4), in the compact thesis layout of
make_thesis_figures_operators.py (one panel per class).

Per instance, one reference point is shared by all runs and logged generations. Within each
(instance, repl) block, every HV is divided by the best final HV (generation 100) of all
configurations in that block (relative HV, 100 % = best final result of the block); ranks are
computed among the configurations of the block at every logged generation (1 = highest HV).

Figures:
  * rel-hv-progress:   mean relative HV over the generations (band: quartiles)
  * rank-progress:     mean rank over the generations
  * final-ecdf:        ECDF of the final relative HV (generation 100)
  * runtime:           CPU time per run (median, band: quartiles) per configuration and class
  * quality-runtime:   median CPU time vs. mean final relative HV per configuration and class

Usage:
  python analysis/analyze_progress_and_runtime.py results/approximations/ex19_hybrid_vs_pure_runtime \\
      --prefix hybrid-vs-pure --note "..."
'''

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from analyze_hypervolume import (read_params, collect_fronts, collect_trajectories, determine_reference_point,
                                 hypervolume_2d, infer_instance_class, archive_generic_run)
from analyze_edge_diversity import set_gen_axis
from make_thesis_figures_operators import new_grid, finish, styles, MUTED, FONT


def load(registry, algo_col):
    rows = read_params(os.path.join(registry, 'design.csv'))
    records = []
    for instance in sorted({r['instance'] for r in rows}):
        traj = collect_trajectories(rows, registry, instance, [algo_col])
        ref = determine_reference_point(collect_fronts(rows, registry, instance, [algo_col]), traj)
        cls = infer_instance_class(instance)
        for algo, repls in traj.items():
            for repl, gens in repls:
                for gen, pts in gens.items():
                    records.append(dict(**{'class': cls}, instance = instance, repl = int(repl),
                                        algorithm = algo, gen = gen, hv = hypervolume_2d(pts, ref)))
        print(f'{cls} {os.path.basename(instance)}', flush = True)
    df = pd.DataFrame(records)
    last = df['gen'].max()
    best = df[df['gen'] == last].groupby(['instance', 'repl'])['hv'].max().rename('best')
    df = df.join(best, on = ['instance', 'repl'])
    df['rel_hv'] = 100 * df['hv'] / df['best']
    df['rank'] = df.groupby(['instance', 'repl', 'gen'])['hv'].rank(ascending = False, method = 'average')

    times = []
    for r in rows:
        path = os.path.join(registry, 'experiments', r['jobid'])
        cpu = os.path.join(path, 'cpu_time.csv')
        times.append({'class': infer_instance_class(r['instance']), 'instance': r['instance'], 'repl': int(r['repl']),
                      'algorithm': r[algo_col], 'wall': float(open(os.path.join(path, 'runtime.csv')).read()),
                      'cpu': float(open(cpu).read()) if os.path.exists(cpu) else np.nan})
    return df, pd.DataFrame(times)


def line_panels(df, value, ylabel, out_base, band = True, start_gen = 0, invert = False, wide = False):
    """wide = True: all classes in one row at the full text width instead of the 2x2 grid."""
    classes = sorted(df['class'].unique())
    fig, axes = (new_grid(classes, rows = 1, cols = len(classes), figsize = (7.1, 2.3)) if wide
                 else new_grid(classes))
    order, style = styles(df['algorithm'].unique())
    handles = {}
    agg = df[df['gen'] >= start_gen].groupby(['class', 'algorithm', 'gen'])[value].agg(
        mean = 'mean', q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75)).reset_index()
    for ax, cls in zip(axes.flat, classes):
        sub = agg[agg['class'] == cls]
        for algo in order:
            color, marker = style[algo]
            line = sub[sub['algorithm'] == algo].sort_values('gen')
            if band:
                ax.fill_between(line['gen'], line['q25'], line['q75'], color = color, alpha = 0.15, linewidth = 0)
            handles[algo], = ax.plot(line['gen'], line['mean'], color = color, marker = marker, markersize = 2,
                                     linewidth = 0.9, label = algo)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if value == 'rank':
            n = len(order)
            ax.set_yticks(range(1, n + 1))
            ax.set_ylim(n + 0.3, 0.7)   # rank 1 (best) at the top
        elif invert:
            ax.invert_yaxis()
        set_gen_axis(ax, int(sub['gen'].max()))
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, labelpad = 1)
    finish(fig, axes, [handles[a] for a in order], order, out_base, ncol = len(order) if wide else None)


def final_ecdf(df, out_base):
    final = df[df['gen'] == df['gen'].max()]
    classes = sorted(final['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(final['algorithm'].unique())
    handles = {}
    for ax, cls in zip(axes.flat, classes):
        sub = final[final['class'] == cls]
        for algo in order:
            vals = np.sort(sub[sub['algorithm'] == algo]['rel_hv'].values)
            y = np.arange(1, len(vals) + 1) / len(vals)
            handles[algo], = ax.step(vals, y, where = 'post', color = style[algo][0], linewidth = 1, label = algo)
        ax.set_ylim(-0.02, 1.02)
        ax.ticklabel_format(axis = 'x', style = 'plain')
        ax.xaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[-1]:
        ax.set_xlabel('Final relative HV (%)', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('$F(x)$', labelpad = 1)
    finish(fig, axes, [handles[a] for a in order], order, out_base)


def runtime_panels(times, out_base):
    """One panel over all runs (the CPU time hardly depends on the instance class), same size and
    fonts as the 2x2 panels so it can be placed next to them: a vertical boxplot per configuration
    (box: quartiles, whiskers: 1.5 IQR), median and factor relative to USGS-PRE written above."""
    from make_thesis_figures_operators import FIGSIZE, GRID, INK
    order, style = styles(times['algorithm'].unique())
    plt.rcParams.update({'font.size': FONT, 'axes.labelsize': FONT, 'xtick.labelsize': FONT - 1,
                         'ytick.labelsize': FONT - 1, 'legend.fontsize': FONT, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize = FIGSIZE)
    data = [times[times['algorithm'] == a]['cpu'].values for a in order]
    bp = ax.boxplot(data, widths = 0.6, patch_artist = True, showfliers = False,
                    medianprops = dict(color = INK, linewidth = 1), whiskerprops = dict(color = MUTED, linewidth = 0.6),
                    capprops = dict(color = MUTED, linewidth = 0.6))
    for patch, a in zip(bp['boxes'], order):
        patch.set_facecolor(style[a][0]); patch.set_alpha(0.8); patch.set_edgecolor('white')
    ref = np.median(times[times['algorithm'] == 'USGS-PRE']['cpu']) if 'USGS-PRE' in order else None
    top = max(np.percentile(d, 75) + 1.5 * (np.percentile(d, 75) - np.percentile(d, 25)) for d in data)
    top = min(top, max(d.max() for d in data))
    for x, d in enumerate(data, start = 1):
        label = f'{np.median(d):.1f} s'
        if ref:
            label += f'\n×{np.median(d) / ref:.2f}'
        ax.text(x, top * 1.03, label, ha = 'center', va = 'bottom', fontsize = FONT - 1.5, color = INK, linespacing = 1.0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([a.replace('HYBRID-0.4', 'HYBRID\n0.4').replace('USGS-PRE', 'USGS-\nPRE') for a in order],
                       fontsize = FONT - 1.5, linespacing = 1.0)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel('CPU time per run (s)', labelpad = 1)
    ax.grid(True, axis = 'y', color = GRID, linewidth = 0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length = 2, width = 0.5, colors = MUTED, pad = 2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout(rect = (0, 0.055, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def runtime_wide(times, out_base, table_font = None, display_width_pt = None, text_size_pt = 10):
    """Standalone runtime figure: vertical boxplot per configuration over all runs, black outlines
    without fill, only the median line in the configuration's colour. Below the plot a small table
    (header: configuration names, rows: median CPU time in s with two decimals and its factor
    relative to the median of USGS-PRE), whose columns line up with the boxes; the unit is given
    only in the row label. USGS-PRE, the baseline of the factor, is placed in the first column.

    table_font / display_width_pt: font file for the table (e.g. the thesis text font) and the width
    at which the PDF is shown in the document; the table font size is then chosen so that the
    table prints at text_size_pt after LaTeX scales the (tightly cropped) figure to that width."""
    from make_thesis_figures_operators import GRID, INK
    order, style = styles(times['algorithm'].unique())
    if 'USGS-PRE' in order:   # baseline of the factor row first
        order = ['USGS-PRE'] + [a for a in order if a != 'USGS-PRE']
    plt.rcParams.update({'font.size': FONT, 'axes.labelsize': FONT, 'xtick.labelsize': FONT - 1,
                         'ytick.labelsize': FONT - 1, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize = (5.0, 3.1))
    data = [times[times['algorithm'] == a]['cpu'].values for a in order]
    bp = ax.boxplot(data, widths = 0.5, patch_artist = True, showfliers = False)
    for i, a in enumerate(order):
        color = style[a][0]
        bp['boxes'][i].set(facecolor = 'none', edgecolor = INK, linewidth = 0.9)
        bp['medians'][i].set(color = color, linewidth = 2.2)
        for part in ('whiskers', 'caps'):
            for line in bp[part][2 * i:2 * i + 2]:
                line.set(color = INK, linewidth = 0.8)
    top = max(min(np.percentile(d, 75) + 1.5 * (np.percentile(d, 75) - np.percentile(d, 25)), d.max()) for d in data)
    # equal-width table columns over the full axes width line up with boxes at x = 1..n
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([])
    ax.tick_params(axis = 'x', length = 0)
    ax.set_ylim(0, top * 1.08)
    medians = [np.median(d) for d in data]
    rows, labels = [[f'{m:.2f}' for m in medians]], ['Median runtime (s)']
    if 'USGS-PRE' in order:
        ref = medians[order.index('USGS-PRE')]
        rows.append([f'{m / ref:.2f}×' for m in medians])
        labels.append('Factor (USGS-PRE = 1)')
    table = ax.table(cellText = rows, colLabels = order, rowLabels = labels, cellLoc = 'center',
                     rowLoc = 'right', bbox = [0, -0.48, 1, 0.42])
    table.auto_set_font_size(False)
    font = matplotlib.font_manager.FontProperties(fname = table_font) if table_font else None
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(MUTED)
        cell.set_linewidth(0.5)
        cell.get_text().set_fontsize(FONT - 1)
        if font is not None:
            cell.get_text().set_fontproperties(font)
        if r == 0:                        # header row with the configuration names
            cell.set_facecolor('#f1f0ec')
            cell.get_text().set_color(INK)
        elif c == -1:                     # row label
            cell.get_text().set_color(INK)
            cell.set_facecolor('#f1f0ec')
        else:
            cell.get_text().set_color(INK)
    ax.set_ylabel('CPU time per run (s)', labelpad = 1)
    ax.grid(True, axis = 'y', color = GRID, linewidth = 0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis = 'y', length = 2, width = 0.5, colors = MUTED, pad = 2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.38)   # room for the table below the plot
    pad = 0.03
    if display_width_pt:
        # scale factor LaTeX applies = shown width / width of the cropped figure; iterate, since the
        # row label width (and thus the crop) depends on the font size
        size = text_size_pt
        for _ in range(4):
            for cell in table.get_celld().values():
                cell.get_text().set_fontsize(size)
            fig.canvas.draw()
            width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
            size = text_size_pt * width_in * 72 / display_width_pt
        for cell in table.get_celld().values():
            cell.get_text().set_fontsize(size)
        print(f'runtime table font size {size:.2f} pt (prints at {text_size_pt} pt at {display_width_pt} pt width)')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300, bbox_inches = 'tight', pad_inches = pad)
    plt.close(fig)


def quality_runtime(df, times, out_base):
    final = df[df['gen'] == df['gen'].max()].groupby(['class', 'algorithm'])['rel_hv'].agg(
        hv = 'mean', hv25 = lambda s: s.quantile(0.25), hv75 = lambda s: s.quantile(0.75))
    cpu = times.groupby(['class', 'algorithm'])['cpu'].agg(
        t = 'median', t25 = lambda s: s.quantile(0.25), t75 = lambda s: s.quantile(0.75))
    d = final.join(cpu).reset_index()
    classes = sorted(d['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(d['algorithm'].unique())
    handles = {}
    for ax, cls in zip(axes.flat, classes):
        sub = d[d['class'] == cls]
        for algo in order:
            r = sub[sub['algorithm'] == algo].iloc[0]
            color, marker = style[algo]
            ax.errorbar(r['t'], r['hv'], xerr = [[r['t'] - r['t25']], [r['t75'] - r['t']]],
                        yerr = [[r['hv'] - r['hv25']], [r['hv75'] - r['hv']]], color = color, linewidth = 0.7,
                        capsize = 0)
            handles[algo], = ax.plot(r['t'], r['hv'], marker = marker, color = color, markersize = 4,
                                     linestyle = 'none', markeredgecolor = 'white', markeredgewidth = 0.5, label = algo)
        ax.ticklabel_format(axis = 'both', style = 'plain')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[:, 0]:
        ax.set_ylabel('Final relative HV (%)', labelpad = 1)
    # one shared x label: two per-panel labels of this length overlap at half text width
    fig.text(0.55, 0.115, 'Median CPU time per run (s)', ha = 'center', fontsize = FONT)
    finish(fig, axes, [handles[a] for a in order], order, out_base)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path')
    parser.add_argument('--algo-col', default = 'operator')
    parser.add_argument('--prefix', default = None)
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<registry name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy these PDFs (names without prefix) there')
    parser.add_argument('--copy', default = '', help = 'comma-separated figure names to copy, e.g. runtime,rank-progress')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--table-font', default = None, help = 'font file for the runtime table (thesis text font)')
    parser.add_argument('--table-width-pt', type = float, default = None,
                        help = 'width at which the runtime figure is shown in the thesis (pt), to match the text size')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.registry_path))
    prefix = args.prefix or label
    out_dir = args.out or os.path.join('figures', 'thesis', label)
    os.makedirs(out_dir, exist_ok = True)

    df, times = load(args.registry_path, args.algo_col)
    base = lambda name: os.path.join(out_dir, f'{prefix}-{name}')
    line_panels(df, 'rel_hv', 'Relative HV (%)', base('rel-hv-progress'), start_gen = 10)
    line_panels(df, 'rank', 'Mean rank', base('rank-progress'), band = False, start_gen = 10, invert = True)
    line_panels(df, 'rank', 'Mean rank', base('rank-progress-wide'), band = False, start_gen = 10, invert = True,
                wide = True)
    final_ecdf(df, base('final-ecdf'))
    names = ['rel-hv-progress', 'rank-progress', 'rank-progress-wide', 'final-ecdf']
    if times['cpu'].notna().all():
        runtime_panels(times, base('runtime'))
        runtime_wide(times, base('runtime-wide'), table_font = args.table_font, display_width_pt = args.table_width_pt)
        quality_runtime(df, times, base('quality-runtime'))
        names += ['runtime', 'runtime-wide', 'quality-runtime']

    tables = [base('checkpoint-hv') + '.csv', base('times') + '.csv']
    df.to_csv(tables[0], index = False)
    times.to_csv(tables[1], index = False)
    figures = [f'{base(n)}.{e}' for n in names for e in ('pdf', 'png')]
    print('written:', *figures, sep = '\n  ')

    if args.thesis_dir and args.copy:
        for n in args.copy.split(','):
            shutil.copy(f'{base(n)}.pdf', args.thesis_dir)
        print(f'copied {args.copy} to {args.thesis_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'progress_and_runtime',
            title = 'Progress over the generations and runtime (all instances)', note = args.note,
            copy_paths = tables + figures,
            changelog_header = '| Datum | Commit | Abbildungen | Notiz |\n|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {figures} | {note} |',
            changelog_rows = [{'figures': ', '.join(names)}])
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
