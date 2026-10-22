#!/usr/bin/env python
'''
Duplicate handling for pure crossover (registry written by
cx_duplicate_handling_runner_expyrimenter.py, factors operator x variant x instance x repl).

For every instance one shared reference point is determined over all operators, variants,
repls and logged generations, so HV values are comparable within the instance. Each
variant is then compared to the baseline of the same operator, instance and repl
(common random numbers: the same initial population):

  * rel_hv_pct:   100 * (HV_variant / HV_baseline - 1), at generations 20, 50 and 100
  * better/worse: share of paired runs in which the variant has a higher / lower HV
  * wilcoxon_p:   paired Wilcoxon signed-rank test of the final HV vs. baseline, and p_holm
                  after Holm correction over all non-baseline groups
  * runtime:      wall-clock time of the run, and its ratio to the paired baseline run
  * repro_stats:  parent redraws, discarded clones, clones accepted at the attempt cap

Outputs (in --out/<registry name>/): hv_runs.csv (one row per job and logged generation),
summary.csv and summary.md, plus three figures in the compact thesis layout of
make_thesis_figures_operators.py (one panel per class):

  * hv-change:     median HV change vs. the paired baseline over the generations (median, since
                   single runs can deviate by +-10 % and dominate the mean, e.g. in C2)
                   (colour = operator, solid = distinct, dashed = distinct_noclone)
  * runtime:       mean runtime factor vs. the paired baseline per operator and variant
  * clones:        share of crossover calls whose offspring was discarded as a parent clone

All of it is archived in analysis_log/<registry name>/.

Usage:
  python analysis/analyze_cx_duplicate_handling.py results/approximations/ex13_cx_duplicate_handling --note "..."
'''

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from analyze_hypervolume import (read_params, read_pf, read_trajectory, hypervolume_2d,
                                 infer_instance_class, archive_generic_run, holm_correction)
from scipy.stats import wilcoxon
from analyze_edge_diversity import set_gen_axis
from make_thesis_figures_operators import new_grid, styles, MUTED, FONT

GENS = (20, 50, 100)          # generations reported in the summary table
VARIANTS = ['distinct', 'distinct_noclone']
VARIANT_LABELS = {'distinct': 'distinct parents', 'distinct_noclone': 'distinct parents + no clones'}
VARIANT_LINES = {'distinct': '-', 'distinct_noclone': '--'}


def load_runs(registry):
    rows = read_params(os.path.join(registry, 'design.csv'))
    records = []
    for instance in sorted({r['instance'] for r in rows}):
        jobs = [r for r in rows if r['instance'] == instance]
        fronts, all_points = {}, []
        for r in jobs:
            path = os.path.join(registry, 'experiments', r['jobid'])
            traj = read_trajectory(os.path.join(path, 'trajectory.csv'))
            all_points += [p for pts in traj.values() for p in pts]
            snaps = {g: pts for g, pts in traj.items() if 0 < g < GENS[-1]}
            snaps[GENS[-1]] = read_pf(os.path.join(path, 'pf.csv'))
            fronts[r['jobid']] = snaps
        # shared reference point of the instance as in determine_reference_point(): worst objective
        # values over all logged generations (incl. the random initial population) of all jobs, +5 %
        ref = tuple(np.array(all_points).max(axis = 0) * 1.05)
        for r in jobs:
            path = os.path.join(registry, 'experiments', r['jobid'])
            runtime = float(open(os.path.join(path, 'runtime.csv')).read().strip())
            stats_path = os.path.join(path, 'repro_stats.csv')
            stats = pd.read_csv(stats_path).iloc[0].to_dict() if os.path.exists(stats_path) else {}
            for g, pts in fronts[r['jobid']].items():
                records.append(dict(jobid = int(r['jobid']), instance = instance,
                                    **{'class': infer_instance_class(instance)},
                                    operator = r['operator'], variant = r['variant'], repl = int(r['repl']),
                                    gen = g, hv = hypervolume_2d(pts, ref), runtime = runtime, **stats))
    return pd.DataFrame.from_records(records)


def pair_with_baseline(df):
    key = ['instance', 'operator', 'repl', 'gen']
    base = df[df['variant'] == 'baseline'][key + ['hv', 'runtime']].rename(
        columns = {'hv': 'hv_base', 'runtime': 'runtime_base'})
    out = df.merge(base, on = key)
    out['rel_hv_pct'] = 100 * (out['hv'] / out['hv_base'] - 1)
    out['runtime_ratio'] = out['runtime'] / out['runtime_base']
    return out


def summarise(paired):
    rows = []
    final = paired[paired['gen'] == GENS[-1]]
    for (cls, op, var), sub in final.groupby(['class', 'operator', 'variant']):
        row = {'class': cls, 'operator': op, 'variant': var, 'runs': len(sub)}
        for g in GENS:
            s = paired[(paired['class'] == cls) & (paired['operator'] == op) &
                       (paired['variant'] == var) & (paired['gen'] == g)]
            row[f'rel_hv_g{g}_pct'] = round(s['rel_hv_pct'].mean(), 3)
        row[f'median_rel_hv_g{GENS[-1]}_pct'] = round(sub['rel_hv_pct'].median(), 3)
        row['wilcoxon_p'] = (float(wilcoxon(sub['hv'], sub['hv_base']).pvalue)
                             if var != 'baseline' else np.nan)
        row.update({
            'better_pct': round(100 * (sub['hv'] > sub['hv_base'] * (1 + 1e-12)).mean(), 1),
            'worse_pct': round(100 * (sub['hv'] < sub['hv_base'] * (1 - 1e-12)).mean(), 1),
            'runtime_s': round(sub['runtime'].mean(), 2),
            'runtime_ratio': round(sub['runtime_ratio'].mean(), 3),
            'cx_calls': round(sub['cx_calls'].mean(), 0),
            'parent_redraws': round(sub['parent_redraws'].mean(), 0),
            'clones_discarded': round(sub['clones_discarded'].mean(), 0),
            'clones_accepted_at_cap': round(sub['clones_accepted_at_cap'].mean(), 1),
            'no_distinct_partner': round(sub['no_distinct_partner'].mean(), 1),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    mask = out['variant'] != 'baseline'
    out.loc[mask, 'p_holm'] = holm_correction(out.loc[mask, 'wilcoxon_p'].tolist())
    out['wilcoxon_p'] = out['wilcoxon_p'].map(lambda p: float(f'{p:.3g}') if p == p else p)
    out['p_holm'] = out['p_holm'].map(lambda p: float(f'{p:.3g}') if p == p else p)
    return out


def to_markdown(summary):
    cols = ['class', 'operator', 'variant', 'rel_hv_g20_pct', 'rel_hv_g50_pct', 'rel_hv_g100_pct',
            'median_rel_hv_g100_pct', 'wilcoxon_p', 'p_holm', 'better_pct', 'worse_pct', 'runtime_s', 'runtime_ratio', 'parent_redraws', 'clones_discarded',
            'clones_accepted_at_cap']
    s = summary[cols]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in s.itertuples(index = False)]
    return '\n'.join(lines)


def save(fig, out_base, legend_rows):
    """Legend rows at the bottom (list of (handles, labels)), then tight layout and pdf/png."""
    kw = dict(loc = 'lower center', frameon = False, handlelength = 1.8, columnspacing = 1.0, borderaxespad = 0.1)
    for k, (handles, labels) in enumerate(legend_rows):
        fig.legend(handles, labels, ncol = len(labels), bbox_to_anchor = (0.5, 0.045 * (len(legend_rows) - 1 - k)), **kw)
    fig.tight_layout(rect = (0, 0.055 * len(legend_rows), 1, 1), h_pad = 0.8, w_pad = 0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi = 300)
    plt.close(fig)


def operator_legend(order, style):
    return [Line2D([], [], color = style[op][0], linewidth = 1.2) for op in order], order


def plot_hv_change(paired, out_base):
    classes = sorted(paired['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(paired['operator'].unique())
    agg = paired[paired['variant'].isin(VARIANTS)].groupby(
        ['class', 'operator', 'variant', 'gen'])['rel_hv_pct'].median().reset_index()
    for ax, cls in zip(axes.flat, classes):
        sub = agg[agg['class'] == cls]
        for op in order:
            for var in VARIANTS:
                line = sub[(sub['operator'] == op) & (sub['variant'] == var)].sort_values('gen')
                line = pd.concat([pd.DataFrame({'gen': [0], 'rel_hv_pct': [0.0]}), line])  # same start
                ax.plot(line['gen'], line['rel_hv_pct'], color = style[op][0], linestyle = VARIANT_LINES[var],
                        linewidth = 0.9)
        ax.axhline(0, color = MUTED, linestyle = ':', linewidth = 0.8)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, GENS[-1])
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Median HV change (%)', labelpad = 1)
    save(fig, out_base, [operator_legend(order, style),
                         ([Line2D([], [], color = MUTED, linestyle = VARIANT_LINES[v], linewidth = 1) for v in VARIANTS],
                          [VARIANT_LABELS[v] for v in VARIANTS])])


def plot_bars(summary, value_col, ylabel, out_base, variants, ref_line = None, labels = None):
    """One panel per class: a group per operator, one bar per variant (colour = operator,
    lighter fill = first variant, full fill = second)."""
    classes = sorted(summary['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(summary['operator'].unique())
    width = 0.8 / len(variants)
    alphas = [0.45, 1.0][-len(variants):]
    for ax, cls in zip(axes.flat, classes):
        sub = summary[summary['class'] == cls]
        for x, op in enumerate(order):
            for k, var in enumerate(variants):
                v = sub[(sub['operator'] == op) & (sub['variant'] == var)][value_col].iloc[0]
                ax.bar(x + (k - (len(variants) - 1) / 2) * width, v, width = width * 0.92,
                       color = style[op][0], alpha = alphas[k], linewidth = 0)
        if ref_line is not None:
            ax.axhline(ref_line, color = MUTED, linestyle = ':', linewidth = 0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_ylim(0, None)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        ax.grid(False, axis = 'x')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, labelpad = 1)
    labels = labels or VARIANT_LABELS
    save(fig, out_base, [([Patch(color = MUTED, alpha = a, linewidth = 0) for a in alphas],
                          [labels[v] for v in variants])])


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path')
    parser.add_argument('--out', default = 'figures/cx_duplicate_handling')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.registry_path))
    out_dir = os.path.join(args.out, label)
    os.makedirs(out_dir, exist_ok = True)

    df = load_runs(args.registry_path)
    paired = pair_with_baseline(df)
    summary = summarise(paired)

    runs_csv = os.path.join(out_dir, 'hv_runs.csv')
    summary_csv = os.path.join(out_dir, 'summary.csv')
    summary_md = os.path.join(out_dir, 'summary.md')
    paired.to_csv(runs_csv, index = False)
    summary.to_csv(summary_csv, index = False)
    with open(summary_md, 'w') as f:
        f.write(f'# Duplicate handling for pure crossover: {label}\n\n')
        f.write('rel_hv_gX_pct: mean HV change vs. the paired baseline run at generation X (%); '
                'median_rel_hv_g100_pct: median of the same at the end. wilcoxon_p / p_holm: paired '
                'Wilcoxon signed-rank test of the final HV vs. baseline, Holm-corrected over all '
                'non-baseline groups. runtime_ratio: mean of runtime / paired baseline runtime.\n\n')
        f.write(to_markdown(summary) + '\n')
    summary['clone_share_pct'] = 100 * summary['clones_discarded'] / summary['cx_calls']
    plot_hv_change(paired, os.path.join(out_dir, f'{label}-hv-change'))
    plot_bars(summary, 'runtime_ratio', 'Runtime factor to baseline',
              os.path.join(out_dir, f'{label}-runtime'), VARIANTS, ref_line = 1.0)
    plot_bars(summary, 'clone_share_pct', 'Discarded clones (%)',
              os.path.join(out_dir, f'{label}-clones'), ['distinct_noclone'])
    figures = [os.path.join(out_dir, f'{label}-{name}.{ext}')
               for name in ('hv-change', 'runtime', 'clones') for ext in ('pdf', 'png')]
    pd.set_option('display.width', 250)
    print(summary.drop(columns = ['runs', 'cx_calls', 'no_distinct_partner']).to_string(index = False))

    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'cx_duplicate_handling',
            title = 'Duplicate handling for pure crossover', note = args.note,
            copy_paths = [runs_csv, summary_csv, summary_md] + figures,
            changelog_header = '| Datum | Commit | Klasse | Operator | Variante | rel. HV g20 (%) | rel. HV g100 (%) '
                               '| Laufzeit-Faktor | verworfene Klone | Notiz |\n|---|---|---|---|---|---|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {operator} | {variant} '
                           '| {rel_hv_g20_pct} | {rel_hv_g100_pct} | {runtime_ratio} | {clones_discarded} | {note} |',
            changelog_rows = summary[summary['variant'] != 'baseline'].to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
