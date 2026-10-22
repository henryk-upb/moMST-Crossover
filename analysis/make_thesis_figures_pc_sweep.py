#!/usr/bin/env python
'''
Compact thesis versions of the p_crossover sweep (output of analyze_pc_sweep.py:
pc_sweep_hv.csv and pc_sweep_progress_hv.csv), in the layout of
make_thesis_figures_operators.py so both can be placed side by side at half the text width:

  * pc-sweep-grid:    one panel per class, HV over p_c after 20 (dotted), 50 (dashed) and
                      100 generations (solid), each run's HV relative to the best final HV
                      (gen 100) among the p_c values of the same instance and repetition
                      (100 % = best), mean over all instances and repetitions of the class;
                      the normalisation makes instances of different HV scale comparable
  * pc-sweep-overlay: all classes in one panel, HV after 100 generations relative to the best
                      p_c of the same instance and repetition (100 % = best), mean over the
                      repetitions, band: quartiles

Usage:
  python analysis/make_thesis_figures_pc_sweep.py figures/pc_sweep/ex12_pc_sweep_exact_xor \\
      --prefix pc-sweep-exact-xor --thesis-dir ../moMST-Crossover-LaTeX/figures --note "..."
'''

import argparse
import os
import shutil

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from analyze_hypervolume import archive_generic_run
from make_thesis_figures_operators import new_grid, finish, PALETTE, MARKERS, INK, MUTED, GRID, FIGSIZE, FONT

GEN_LINES = {20: ':', 50: '--', 100: '-'}


def class_style(classes):
    return {cls: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, cls in enumerate(classes)}


def set_pc_axis(ax):
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))


def plot_grid(final, progress, out_base):
    '''final: pc_sweep_hv.csv (gen 100), progress: pc_sweep_progress_hv.csv (gens 20, 50).'''
    cols = ['class', 'instance', 'repl', 'p_crossover', 'gen', 'hv']
    data = pd.concat([progress[cols], final.assign(gen = 100)[cols]])
    best = final.groupby(['instance', 'repl'])['hv'].max().rename('best').reset_index()
    data = data.merge(best, on = ['instance', 'repl'])
    data['hv'] = 100 * data['hv'] / data['best']
    agg = data.groupby(['class', 'gen', 'p_crossover'])['hv'].mean().reset_index()
    classes = sorted(agg['class'].unique())
    style = class_style(classes)
    fig, axes = new_grid(classes)
    for ax, cls in zip(axes.flat, classes):
        color, marker = style[cls]
        for gen, ls in GEN_LINES.items():
            line = agg[(agg['class'] == cls) & (agg['gen'] == gen)].sort_values('p_crossover')
            ax.plot(line['p_crossover'], line['hv'], color = color, linestyle = ls, linewidth = 0.9,
                    marker = marker if gen == 100 else None, markersize = 2)
        set_pc_axis(ax)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[-1]:
        ax.set_xlabel('$p_c$', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative HV (%)', labelpad = 1)
    handles = [Line2D([], [], color = MUTED, linestyle = ls, linewidth = 1) for ls in GEN_LINES.values()]
    finish(fig, axes, handles, [f'after {g} gen.' for g in GEN_LINES], out_base)


def plot_overlay(final, out_base):
    agg = final.groupby(['class', 'p_crossover'])['relative_pct'].agg(
        mean = 'mean', q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75)).reset_index()
    classes = sorted(agg['class'].unique())
    style = class_style(classes)
    plt.rcParams.update({'font.size': FONT, 'axes.labelsize': FONT, 'xtick.labelsize': FONT - 1,
                         'ytick.labelsize': FONT - 1, 'legend.fontsize': FONT, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize = FIGSIZE)
    handles = []
    for cls in classes:
        color, marker = style[cls]
        line = agg[agg['class'] == cls].sort_values('p_crossover')
        ax.fill_between(line['p_crossover'], line['q25'], line['q75'], color = color, alpha = 0.15, linewidth = 0)
        h, = ax.plot(line['p_crossover'], line['mean'], color = color, marker = marker, markersize = 2.5,
                     linewidth = 1, label = cls)
        handles.append(h)
    ax.grid(True, color = GRID, linewidth = 0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length = 2, width = 0.5, colors = MUTED, pad = 2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    set_pc_axis(ax)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    ax.set_xlabel('$p_c$', labelpad = 1)
    ax.set_ylabel('Relative HV after 100 generations (%)', labelpad = 1)
    fig.legend(handles, classes, loc = 'lower center', ncol = len(classes), frameon = False,
               handletextpad = 0.3, columnspacing = 1.0, borderaxespad = 0.1, markerscale = 1.8)
    fig.tight_layout(rect = (0, 0.055, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_dir', help = 'output directory of analyze_pc_sweep.py')
    parser.add_argument('--prefix', default = 'pc-sweep')
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<sweep dir name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy the PDFs into this directory')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.sweep_dir))
    out_dir = args.out or os.path.join('figures', 'thesis', label)
    os.makedirs(out_dir, exist_ok = True)
    final = pd.read_csv(os.path.join(args.sweep_dir, 'pc_sweep_hv.csv'))
    progress = pd.read_csv(os.path.join(args.sweep_dir, 'pc_sweep_progress_hv.csv'))

    grid_base = os.path.join(out_dir, f'{args.prefix}-grid')
    overlay_base = os.path.join(out_dir, f'{args.prefix}-overlay')
    plot_grid(final, progress, grid_base)
    plot_overlay(final, overlay_base)
    outputs = [f'{b}.{e}' for b in (grid_base, overlay_base) for e in ('pdf', 'png')]
    print('written:', *outputs, sep = '\n  ')

    if args.thesis_dir:
        for b in (grid_base, overlay_base):
            shutil.copy(f'{b}.pdf', args.thesis_dir)
        print(f'copied PDFs to {args.thesis_dir}')

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'thesis_figures_pc_sweep',
            title = 'Thesis figures (compact)', note = args.note, copy_paths = outputs,
            changelog_header = '| Datum | Commit | Abbildungen | Notiz |\n|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {figures} | {note} |',
            changelog_rows = [{'figures': f'{args.prefix}-grid, {args.prefix}-overlay'}])
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
