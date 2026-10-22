#!/usr/bin/env python
'''
Best p_crossover over the WHOLE run ("anytime" view) for a p_crossover sweep registry
(e.g. ex17: exact XOR + USGS-PRE, p_crossover in {0.0, ..., 1.0}, all instances).

For every instance one reference point is shared by all runs and logged generations
(as in analyze_pc_sweep.py). For every run the HV of each logged population (generations
10, 20, ..., 100) is computed, and within each block (instance, repl, generation)
  * relative HV = 100 * hv / best hv among the p_crossover values of that block
  * rank        = rank among the p_crossover values of that block (1 = highest HV)
The anytime score of a run is its relative HV averaged over all checkpoints, i.e. the
normalised area under its HV-over-generations curve. Per class, the p_crossover with the
highest mean anytime score is "best"; every other value is compared with it by a paired
Wilcoxon signed-rank test over the (instance, repl) blocks, Holm-corrected within the class.
Values that are not significantly worse (p_holm >= alpha) are "tied with the best".

Figures (compact thesis layout of make_thesis_figures_operators.py):
  * pc-anytime-heatmap: one panel per class, x = p_c, y = generation, colour = mean rank
                        (yellow = better, purple = worse); the best p_c of every generation is marked
  * pc-anytime-score:   anytime score per class over p_c (band: quartiles), best p_c filled,
                        values tied with the best drawn as a thick segment

Usage:
  python analysis/analyze_pc_anytime.py results/approximations/ex17_pc_sweep_exact_xor_all \\
      --prefix pc-anytime-exact-xor-all --thesis-dir ../moMST-Crossover-LaTeX/figures --note "..."
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
from scipy.stats import wilcoxon

from analyze_hypervolume import (read_params, collect_fronts, collect_trajectories, determine_reference_point,
                                 hypervolume_2d, infer_instance_class, holm_correction, archive_generic_run)
from make_thesis_figures_operators import new_grid, PALETTE, MARKERS, INK, MUTED, GRID, FIGSIZE, FONT


def load_checkpoint_hv(registry):
    rows = read_params(os.path.join(registry, 'design.csv'))
    records = []
    for instance in sorted({r['instance'] for r in rows}):
        traj = collect_trajectories(rows, registry, instance, ['p_crossover'])
        # same reference point as analyze_pc_sweep.py: final fronts + all trajectory snapshots
        ref = determine_reference_point(collect_fronts(rows, registry, instance, ['p_crossover']), traj)
        cls = infer_instance_class(instance)
        for pc, repls in traj.items():
            for repl, gens in repls:
                for gen, pts in gens.items():
                    if gen == 0:
                        continue  # identical random initial population for all p_c
                    records.append(dict(**{'class': cls}, instance = instance, repl = int(repl),
                                        p_crossover = float(pc), gen = gen, hv = hypervolume_2d(pts, ref)))
        print(f'{cls} {os.path.basename(instance)}: {sum(len(r) for r in traj.values())} runs', flush = True)
    df = pd.DataFrame(records)
    block = ['instance', 'repl', 'gen']
    df['relative_pct'] = 100 * df['hv'] / df.groupby(block)['hv'].transform('max')
    df['rank'] = df.groupby(block)['hv'].rank(ascending = False, method = 'average')
    return df


def anytime_scores(df):
    '''One row per run: mean relative HV over all checkpoints.'''
    return df.groupby(['class', 'instance', 'repl', 'p_crossover'])['relative_pct'].mean().rename('score').reset_index()


def best_and_ties(scores, alpha):
    rows = []
    for cls, sub in scores.groupby('class'):
        wide = sub.pivot_table(index = ['instance', 'repl'], columns = 'p_crossover', values = 'score')
        means = wide.mean()
        best = means.idxmax()
        others = [pc for pc in wide.columns if pc != best]
        pvals = [float(wilcoxon(wide[best], wide[pc]).pvalue) if (wide[best] != wide[pc]).any() else 1.0
                 for pc in others]
        holm = dict(zip(others, holm_correction(pvals)))
        for pc in wide.columns:
            rows.append({'class': cls, 'p_crossover': pc, 'mean_score': means[pc],
                         'q25': wide[pc].quantile(0.25), 'q75': wide[pc].quantile(0.75),
                         'best': pc == best, 'p_holm_vs_best': np.nan if pc == best else holm[pc],
                         'tied_with_best': pc == best or holm[pc] >= alpha})
    return pd.DataFrame(rows)


def set_pc_axis(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])


def plot_heatmap(df, out_base):
    agg = df.groupby(['class', 'gen', 'p_crossover'])['rank'].mean().reset_index()
    classes = sorted(agg['class'].unique())
    pcs = sorted(agg['p_crossover'].unique())
    gens = sorted(agg['gen'].unique())
    n = len(pcs)
    fig, axes = new_grid(classes)
    cmap = matplotlib.colormaps['viridis_r']  # rank 1 (best) = bright yellow, worst = dark purple
    for ax, cls in zip(axes.flat, classes):
        mat = agg[agg['class'] == cls].pivot(index = 'gen', columns = 'p_crossover', values = 'rank').loc[gens, pcs]
        im = ax.imshow(mat.values, origin = 'lower', aspect = 'auto', cmap = cmap, vmin = 1, vmax = n,
                       extent = (-0.5, n - 0.5, -0.5, len(gens) - 0.5), interpolation = 'nearest')
        # best p_c of every generation (lowest mean rank); ties within 0.01 are all marked,
        # e.g. in C4 late in the run, where every p_c < 1 has reached the same front
        rows, cols = np.nonzero(mat.values <= mat.values.min(axis = 1, keepdims = True) + 0.01)
        ax.scatter(cols, rows, s = 9, facecolor = 'white', edgecolor = INK, linewidth = 0.5, zorder = 3)
        ax.grid(False)
        xt = [i for i, pc in enumerate(pcs) if round(pc * 10) % 5 == 0]
        ax.set_xticks(xt)
        ax.set_xticklabels([f'{pcs[i]:g}' for i in xt])
        yt = [i for i, g in enumerate(gens) if g % 20 == 0]
        ax.set_yticks(yt)
        ax.set_yticklabels([str(gens[i]) for i in yt])
    for ax in axes[-1]:
        ax.set_xlabel('$p_c$', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Generation', labelpad = 1)
    fig.tight_layout(rect = (0, 0.11, 1, 1), h_pad = 0.8, w_pad = 0.6)
    cax = fig.add_axes([0.3, 0.06, 0.4, 0.02])
    cb = fig.colorbar(im, cax = cax, orientation = 'horizontal')
    cb.set_ticks([1, (1 + n) / 2, n])
    cb.set_ticklabels(['1', f'{(1 + n) / 2:g}', f'{n}'])
    cb.ax.tick_params(labelsize = FONT - 1, length = 2, colors = MUTED)
    cb.outline.set_linewidth(0.4)
    cb.set_label('Mean rank (1 = best)', fontsize = FONT - 1, labelpad = 1)
    cb.ax.xaxis.set_label_position('top')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def plot_score(ties, out_base):
    classes = sorted(ties['class'].unique())
    style = {cls: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, cls in enumerate(classes)}
    plt.rcParams.update({'font.size': FONT, 'axes.labelsize': FONT, 'xtick.labelsize': FONT - 1,
                         'ytick.labelsize': FONT - 1, 'legend.fontsize': FONT, 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize = FIGSIZE)
    handles = []
    for cls in classes:
        color, marker = style[cls]
        sub = ties[ties['class'] == cls].sort_values('p_crossover')
        ax.fill_between(sub['p_crossover'], sub['q25'], sub['q75'], color = color, alpha = 0.12, linewidth = 0)
        h, = ax.plot(sub['p_crossover'], sub['mean_score'], color = color, linewidth = 0.9,
                     marker = marker, markersize = 2.5, markerfacecolor = 'white', markeredgewidth = 0.7, label = cls)
        handles.append(h)
        # values tied with the best: thick segment(s) over consecutive tied p_c values
        tied = sub[sub['tied_with_best']]
        for _, run in tied.groupby((~sub['tied_with_best']).cumsum()[tied.index]):
            ax.plot(run['p_crossover'], run['mean_score'], color = color, linewidth = 3, alpha = 0.55,
                    solid_capstyle = 'round', zorder = 2)
        b = sub[sub['best']].iloc[0]
        ax.plot(b['p_crossover'], b['mean_score'], marker = marker, markersize = 5, color = color,
                markeredgecolor = 'white', markeredgewidth = 0.6, zorder = 4)
    ax.grid(True, color = GRID, linewidth = 0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length = 2, width = 0.5, colors = MUTED, pad = 2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    set_pc_axis(ax, [0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel('$p_c$', labelpad = 1)
    ax.set_ylabel('Anytime score (%)', labelpad = 1)
    extra = [Line2D([], [], color = MUTED, linewidth = 3, alpha = 0.55, solid_capstyle = 'round'),
             Line2D([], [], color = MUTED, marker = 'o', linestyle = 'none', markersize = 5, markeredgecolor = 'white')]
    fig.legend(handles, classes, loc = 'lower center', ncol = len(classes), frameon = False,
               handletextpad = 0.3, columnspacing = 1.0, borderaxespad = 0.1, bbox_to_anchor = (0.5, 0.045))
    fig.legend(extra, ['tied with best', 'best'], loc = 'lower center', ncol = 2, frameon = False,
               handletextpad = 0.3, columnspacing = 1.0, borderaxespad = 0.1, bbox_to_anchor = (0.5, 0.0))
    fig.tight_layout(rect = (0, 0.1, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path')
    parser.add_argument('--prefix', default = 'pc-anytime')
    parser.add_argument('--alpha', type = float, default = 0.05)
    parser.add_argument('--out', default = None, help = 'default: figures/thesis/<registry name>')
    parser.add_argument('--thesis-dir', default = None, help = 'also copy the PDFs into this directory')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.registry_path))
    out_dir = args.out or os.path.join('figures', 'thesis', label)
    os.makedirs(out_dir, exist_ok = True)

    df = load_checkpoint_hv(args.registry_path)
    scores = anytime_scores(df)
    ties = best_and_ties(scores, args.alpha)

    paths = {k: os.path.join(out_dir, f'{args.prefix}-{k}.csv') for k in ('checkpoint-hv', 'scores')}
    df.to_csv(paths['checkpoint-hv'], index = False)
    ties.to_csv(paths['scores'], index = False)
    pd.set_option('display.width', 200)
    print(ties.round(4).to_string(index = False))

    heat_base = os.path.join(out_dir, f'{args.prefix}-heatmap')
    score_base = os.path.join(out_dir, f'{args.prefix}-score')
    plot_heatmap(df, heat_base)
    plot_score(ties, score_base)
    figures = [f'{b}.{e}' for b in (heat_base, score_base) for e in ('pdf', 'png')]
    if args.thesis_dir:
        for b in (heat_base, score_base):
            shutil.copy(f'{b}.pdf', args.thesis_dir)
        print(f'copied PDFs to {args.thesis_dir}')

    if not args.no_archive:
        summary = ties[ties['best']][['class', 'p_crossover', 'mean_score']].merge(
            ties[ties['tied_with_best']].groupby('class')['p_crossover'].agg(lambda s: ', '.join(f'{v:g}' for v in s))
            .rename('tied').reset_index(), on = 'class')
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'pc_anytime',
            title = 'Best p_c over the whole run (anytime)', note = args.note,
            copy_paths = list(paths.values()) + figures,
            changelog_header = '| Datum | Commit | Klasse | bestes p_c | Anytime-Score (%) | gleichauf mit bestem | Notiz |\n'
                               '|---|---|---|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {p_crossover:g} '
                           '| {mean_score:.3f} | {tied} | {note} |',
            changelog_rows = summary.to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
