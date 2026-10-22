#!/usr/bin/env python
'''
Survival selection variants and offspring clones (registry written by
survival_variants_runner_expyrimenter.py, factors operator x variant x instance x repl).

Part 1, survival selection: every variant is compared to the baseline of the same operator,
instance and repl (shared reference point per instance, as in analyze_cx_duplicate_handling.py):
mean / median HV change at generations 20, 50, 100, paired Wilcoxon test with Holm correction,
runtime factor.

Part 2, offspring clones (baseline variant only, i.e. the original algorithm): share of offspring
identical to a parent, per class and operator.
  * crossover: identical to either parent ("any parent"), and the per-parent rate
    (eq_p1 + eq_p2) / (2 * calls), i.e. the probability that the offspring equals one given
    parent. The per-parent rate is the one comparable to mutation, which has a single parent.
    Also the rate among calls with two different parents.
  * mutation: identical to its parent.

Figures (compact thesis layout, one panel per class):
  * hv-change:   median HV change vs. baseline over the generations
                 (colour = operator, solid = cd_fix, dashed = dedup)
  * runtime:     mean runtime factor vs. baseline per operator and variant
  * clones:      share of offspring identical to a parent over the generations
                 (crossover: solid = any parent, dashed = per parent; mutation: its parent)

Usage:
  python analysis/analyze_survival_variants.py results/approximations/ex14_survival_variants --note "..."
'''

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from analyze_hypervolume import read_params, infer_instance_class, archive_generic_run, holm_correction
from analyze_cx_duplicate_handling import load_runs, pair_with_baseline, save, plot_bars, GENS
from analyze_edge_diversity import set_gen_axis
from make_thesis_figures_operators import new_grid, styles, MUTED

VARIANTS = ['cd_fix', 'dedup']
VARIANT_LABELS = {'cd_fix': 'fixed crowding distance', 'dedup': 'duplicates removed'}
VARIANT_LINES = {'cd_fix': '-', 'dedup': '--'}
CROSSOVER = ['UNION', 'INTER', 'XOR']


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index = False)]
    return '\n'.join(lines)


def summarise(paired):
    rows = []
    final = paired[paired['gen'] == GENS[-1]]
    for (cls, op, var), sub in final.groupby(['class', 'operator', 'variant']):
        row = {'class': cls, 'operator': op, 'variant': var, 'runs': len(sub)}
        for g in GENS:
            s = paired[(paired['class'] == cls) & (paired['operator'] == op) &
                       (paired['variant'] == var) & (paired['gen'] == g)]
            row[f'rel_hv_g{g}_pct'] = round(s['rel_hv_pct'].mean(), 3)
        row['median_rel_hv_g100_pct'] = round(sub['rel_hv_pct'].median(), 3)
        same = (sub['hv'] == sub['hv_base']).all()
        row['wilcoxon_p'] = (np.nan if var == 'baseline' else
                             1.0 if same else float(wilcoxon(sub['hv'], sub['hv_base']).pvalue))
        row.update({
            'better_pct': round(100 * (sub['hv'] > sub['hv_base'] * (1 + 1e-12)).mean(), 1),
            'worse_pct': round(100 * (sub['hv'] < sub['hv_base'] * (1 - 1e-12)).mean(), 1),
            'runtime_s': round(sub['runtime'].mean(), 2),
            'runtime_ratio': round(sub['runtime_ratio'].mean(), 3),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    mask = out['variant'] != 'baseline'
    out.loc[mask, 'p_holm'] = holm_correction(out.loc[mask, 'wilcoxon_p'].tolist())
    for col in ('wilcoxon_p', 'p_holm'):
        out[col] = out[col].map(lambda p: float(f'{p:.3g}') if p == p else p)
    return out


def load_clones(registry):
    frames = []
    for r in read_params(os.path.join(registry, 'design.csv')):
        path = os.path.join(registry, 'experiments', r['jobid'], 'clone_stats.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['class'] = infer_instance_class(r['instance'])
        df['operator'] = r['operator']
        df['variant'] = r['variant']
        frames.append(df)
    return pd.concat(frames, ignore_index = True)


def clone_rates(df):
    '''Pooled rates (%) per group; works on any grouping of clone_stats rows.'''
    cx = df['cx_calls'].sum()
    mut = df['mut_calls'].sum()
    if cx > 0:
        distinct = cx - df['cx_parents_identical'].sum()
        return pd.Series({
            'calls': cx,
            'any_parent_pct': 100 * df['cx_eq_any'].sum() / cx,
            'per_parent_pct': 100 * (df['cx_eq_p1'].sum() + df['cx_eq_p2'].sum()) / (2 * cx),
            'parents_identical_pct': 100 * df['cx_parents_identical'].sum() / cx,
            'any_parent_distinct_parents_pct': 100 * df['cx_eq_any_distinct_parents'].sum() / distinct
                                                if distinct else np.nan})
    return pd.Series({'calls': mut, 'any_parent_pct': 100 * df['mut_eq_parent'].sum() / mut,
                      'per_parent_pct': 100 * df['mut_eq_parent'].sum() / mut,
                      'parents_identical_pct': np.nan, 'any_parent_distinct_parents_pct': np.nan})


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
                line = pd.concat([pd.DataFrame({'gen': [0], 'rel_hv_pct': [0.0]}), line])
                ax.plot(line['gen'], line['rel_hv_pct'], color = style[op][0], linestyle = VARIANT_LINES[var],
                        linewidth = 0.9)
        ax.axhline(0, color = MUTED, linestyle = ':', linewidth = 0.8)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, GENS[-1])
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Median HV change (%)', labelpad = 1)
    save(fig, out_base, [([Line2D([], [], color = style[op][0], linewidth = 1.2) for op in order], order),
                         ([Line2D([], [], color = MUTED, linestyle = VARIANT_LINES[v], linewidth = 1) for v in VARIANTS],
                          [VARIANT_LABELS[v] for v in VARIANTS])])


def plot_clones(clones, out_base):
    base = clones[clones['variant'] == 'baseline']
    classes = sorted(base['class'].unique())
    fig, axes = new_grid(classes)
    order, style = styles(base['operator'].unique())
    per_gen = base.groupby(['class', 'operator', 'gen']).apply(clone_rates).reset_index()
    for ax, cls in zip(axes.flat, classes):
        sub = per_gen[per_gen['class'] == cls]
        for op in order:
            line = sub[sub['operator'] == op].sort_values('gen')
            ax.plot(line['gen'], line['any_parent_pct'], color = style[op][0], linewidth = 0.9)
            if op in CROSSOVER:
                ax.plot(line['gen'], line['per_parent_pct'], color = style[op][0], linewidth = 0.9, linestyle = '--')
        ax.set_ylim(0, 100)
        ax.ticklabel_format(axis = 'y', style = 'plain')
        set_gen_axis(ax, GENS[-1])
    for ax in axes[-1]:
        ax.set_xlabel('Generation', labelpad = 1)
    for ax in axes[:, 0]:
        ax.set_ylabel('Copies of a parent (%)', labelpad = 1)
    save(fig, out_base, [([Line2D([], [], color = style[op][0], linewidth = 1.2) for op in order], order),
                         ([Line2D([], [], color = MUTED, linewidth = 1), Line2D([], [], color = MUTED, linestyle = '--', linewidth = 1)],
                          ['any parent', 'per parent (crossover)'])])


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path')
    parser.add_argument('--out', default = 'figures/survival_variants')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = '')
    args = parser.parse_args()

    label = os.path.basename(os.path.normpath(args.registry_path))
    out_dir = os.path.join(args.out, label)
    os.makedirs(out_dir, exist_ok = True)
    base = os.path.join(out_dir, label)

    paired = pair_with_baseline(load_runs(args.registry_path))
    summary = summarise(paired)
    clones = load_clones(args.registry_path)
    clone_table = clones[clones['variant'] == 'baseline'].groupby(['class', 'operator']).apply(clone_rates).reset_index()
    clone_table = clone_table.round(2)
    clone_all = clones[clones['variant'] == 'baseline'].groupby('operator').apply(clone_rates).reset_index().round(2)

    plot_hv_change(paired, f'{base}-hv-change')
    plot_bars(summary, 'runtime_ratio', 'Runtime factor to baseline', f'{base}-runtime', VARIANTS, ref_line = 1.0,
              labels = VARIANT_LABELS)
    plot_clones(clones, f'{base}-clones')

    paths = {name: os.path.join(out_dir, name) for name in ('hv_runs.csv', 'summary.csv', 'clone_rates.csv', 'summary.md')}
    paired.to_csv(paths['hv_runs.csv'], index = False)
    summary.to_csv(paths['summary.csv'], index = False)
    clone_table.to_csv(paths['clone_rates.csv'], index = False)
    with open(paths['summary.md'], 'w') as f:
        f.write(f'# Survival selection variants and offspring clones: {label}\n\n')
        f.write('## Survival selection (vs. paired baseline)\n\n')
        f.write(md_table(summary.drop(columns = ['runs'])) + '\n\n')
        f.write('## Offspring identical to a parent (baseline variant)\n\n')
        f.write('per_parent_pct: crossover (eq_p1 + eq_p2) / (2 calls), mutation = any_parent_pct.\n\n')
        f.write(md_table(clone_table) + '\n\nAll classes pooled:\n\n')
        f.write(md_table(clone_all) + '\n')

    pd.set_option('display.width', 250)
    print(summary.drop(columns = ['runs']).to_string(index = False))
    print(clone_table.to_string(index = False))
    print(clone_all.to_string(index = False))

    figures = [f'{base}-{name}.{ext}' for name in ('hv-change', 'runtime', 'clones') for ext in ('pdf', 'png')]
    if not args.no_archive:
        entry_dir = archive_generic_run(
            args.archive_dir, label, 'survival_variants',
            title = 'Survival selection variants and offspring clones', note = args.note,
            copy_paths = list(paths.values()) + figures,
            changelog_header = '| Datum | Commit | Klasse | Operator | Variante | Median rel. HV g100 (%) | p (Holm) '
                               '| Laufzeit-Faktor | Notiz |\n|---|---|---|---|---|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {operator} | {variant} '
                           '| {median_rel_hv_g100_pct} | {p_holm} | {runtime_ratio} | {note} |',
            changelog_rows = summary[summary['variant'] != 'baseline'].to_dict('records'))
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
