#!/usr/bin/env python
'''
Pairwise Wilcoxon-Mann-Whitney tests on the HV-indicator, per instance, in the style of
Bossek & Grimme (ECJ 2023, "On single-objective sub-graph-based mutation ...", Tab. 1/2 and
Fig. 10) and Bossek & Grimme (GECCO 2024, Tab. 2).

Protocol
  * Per instance, the final front of every run is reduced to a scalar with the SAME reference
    point that the other analysis scripts use (determine_reference_point() in
    analyze_hypervolume.py: nadir of all final fronts and trajectory snapshots of the instance,
    +5%).
  * Indicator (--indicator):
      hv-indicator (default, as in the paper; lower is better, 0 = best):
          I(A) = (HV(R) - HV(A)) / HV(R),
          where R is the non-dominated union of ALL final fronts of all algorithms of the
          instance (best known approximation of the Pareto front).
      hv  (raw hypervolume; higher is better).
    I is a strictly decreasing affine map of HV per instance, so the test decisions are
    IDENTICAL for both choices -- the indicator only makes the table values comparable
    between instances (0..1 instead of 1e6..1e7).
  * For every instance and every ordered pair (A, B) of algorithms: one-sided
    Wilcoxon-Mann-Whitney (rank-sum) test, H1: A is better than B (I(A) stochastically lower).
    Exact null distribution if there are no ties, otherwise normal approximation.
  * p-values adjusted per instance over all k(k-1) one-sided tests (Bonferroni-Holm by
    default, --correction bh for Benjamini-Hochberg as in the GECCO 2026 USGS paper).
  * Significance level alpha = 0.01 (--alpha).
  * Effect size: Vargha-Delaney A12 = P(A better than B) + 0.5 P(tie).

Why the UNPAIRED rank-sum test although repl k uses the same seed for every operator?
  The seed only fixes the initial population; the runs diverge after the first variation
  step (Spearman correlation of paired final HVs in ex10 ~0.35). Moreover, with R = 10
  repetitions the exact two-sided signed-rank test cannot get below p = 2/2^10 ~ 0.00195,
  which after Holm over 6 pairs is ~0.0117 > 0.01 -- nothing could ever be significant.
  The rank-sum test reaches p = 1/C(20, 10) ~ 5.4e-6 with 10 vs. 10 runs.

Outputs (in --out, default figures/wilcoxon/<experiment>/):
  wmw_values.csv      indicator values per (instance, algorithm, repl) -- reusable via --reuse-values
  wmw_tests.csv       one row per (instance, A, B): medians, p_raw, p_adj, A12, significant
  wmw_summary.txt     per class: number of instances on which A is significantly better than B
  wmw_table.tex       Tab.-1-style LaTeX table: mean +- sd and stat column (X^+ notation);
                      needs booktabs, multirow and adjustbox (all in the thesis preamble)
  wmw_heatmap.png/.pdf  Fig.-10-style heatmap: fraction of instances on which the COLUMN
                        algorithm is significantly better than the ROW algorithm, per class
Each run is archived under analysis_log/<experiment>/ like the other analysis scripts.

Usage:
  python analysis/analyze_wilcoxon.py results/approximations/ex10_pure_cx_vs_usgs_pre
  python analysis/analyze_wilcoxon.py results/approximations/ex06 --algo-col crossover,mutator \
      --order UNION+USGS-PRE XOR+USGS-PRE --correction bh --note "..."
  # re-render table/heatmap without recomputing HV:
  python analysis/analyze_wilcoxon.py results/approximations/ex10_pure_cx_vs_usgs_pre --reuse-values
'''

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

from analyze_hypervolume import (
    read_params, collect_fronts, collect_trajectories, determine_reference_point, compute_hv_table,
    hypervolume_2d, infer_instance_class, holm_correction, archive_generic_run,
)

INSTANCE_NO_RE = re.compile(r'_(\d+)\.graph$')
# fixed operator order (same as analyze_operators_by_class.py); unknown labels are appended sorted
OPERATOR_ORDER = ['UNION', 'INTER', 'XOR', 'USGS-PRE']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQ_LIGHT, SEQ_DARK = '#eef4fc', '#1c4f91'   # single-hue sequential ramp (blue of the operator palette)


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------

def p_adjust(pvals, method = 'holm'):
    '''Adjusted p-values, order-preserving w.r.t. the input. 'holm' = Bonferroni-Holm (FWER),
    'bh' = Benjamini-Hochberg (FDR).'''
    pvals = [1.0 if (p is None or np.isnan(p)) else float(p) for p in pvals]
    if method == 'holm':
        return np.array(holm_correction(pvals))
    if method == 'bh':
        m = len(pvals)
        order = sorted(range(m), key = lambda i: pvals[i])
        adjusted = [0.0] * m
        running_min = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running_min = min(running_min, pvals[i] * m / (rank + 1))
            adjusted[i] = running_min
        return np.array(adjusted)
    raise ValueError(f"unknown correction '{method}' (use 'holm' or 'bh')")


def one_sided_wmw(x, y, alternative = 'less'):
    '''One-sided Wilcoxon-Mann-Whitney p-value for H1: x stochastically less ('less') or
    greater ('greater') than y. Exact null distribution if there are no ties, otherwise the
    normal approximation with tie correction. Two constant, identical samples give p = 1.'''
    x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
    pooled = np.concatenate([x, y])
    if np.all(pooled == pooled[0]):
        return 1.0
    method = 'exact' if len(np.unique(pooled)) == len(pooled) else 'asymptotic'
    p = stats.mannwhitneyu(x, y, alternative = alternative, method = method).pvalue
    return 1.0 if np.isnan(p) else float(p)


def vargha_delaney_a12(x, y, lower_is_better = True):
    '''Probability that a random run of x is better than a random run of y (ties count 1/2).
    0.5 = no effect; >= 0.71 is commonly called a large effect.'''
    x, y = np.asarray(x, dtype = float), np.asarray(y, dtype = float)
    better = (x[:, None] < y[None, :]) if lower_is_better else (x[:, None] > y[None, :])
    ties = x[:, None] == y[None, :]
    return float((better.sum() + 0.5 * ties.sum()) / (len(x) * len(y)))


def run_pairwise_tests(df, algos, value_col = 'hv_indicator', lower_is_better = True,
                        alpha = 0.01, correction = 'holm'):
    '''df: long format with columns instance, class, no, algorithm, repl, <value_col>.
    Returns one row per (instance, ordered pair A != B) with the one-sided test of
    H1: A better than B, p-values adjusted per instance.'''
    alternative = 'less' if lower_is_better else 'greater'
    rows = []
    for instance, g in df.groupby('instance', sort = True):
        samples = {a: g.loc[g['algorithm'] == a, value_col].to_numpy() for a in algos}
        inst_rows = []
        for a in algos:
            for b in algos:
                if a == b or len(samples[a]) == 0 or len(samples[b]) == 0:
                    continue
                inst_rows.append({
                    'instance': instance, 'class': g['class'].iloc[0], 'no': g['no'].iloc[0],
                    'A': a, 'B': b, 'n_A': len(samples[a]), 'n_B': len(samples[b]),
                    'median_A': float(np.median(samples[a])), 'median_B': float(np.median(samples[b])),
                    'p_raw': one_sided_wmw(samples[a], samples[b], alternative),
                    'A12': vargha_delaney_a12(samples[a], samples[b], lower_is_better),
                })
        adjusted = p_adjust([r['p_raw'] for r in inst_rows], correction)
        for r, p_adj in zip(inst_rows, adjusted):
            r['p_adj'] = float(p_adj)
            r['significant'] = bool(p_adj < alpha)
        rows.extend(inst_rows)
    return pd.DataFrame(rows)


def significance_fractions(tests, algos):
    '''class -> DataFrame[row = B, column = A]: fraction of instances of that class on which
    the column algorithm A is significantly better than the row algorithm B (paper Fig. 10).'''
    out = {}
    for cls, g in tests.groupby('class', sort = True):
        n_inst = g['instance'].nunique()
        counts = g[g['significant']].groupby(['B', 'A']).size()
        mat = pd.DataFrame(0.0, index = algos, columns = algos)
        for (b, a), c in counts.items():
            mat.loc[b, a] = c / n_inst
        for a in algos:
            mat.loc[a, a] = np.nan
        out[cls] = mat
    return out


# ---------------------------------------------------------------------------------------------
# indicator values from the registry
# ---------------------------------------------------------------------------------------------

def nondominated_2d(points):
    '''Non-dominated subset (minimisation, both objectives) of a list of (c1, c2) points;
    duplicates are kept only once.'''
    pts = sorted(set(map(tuple, points)))
    front, best_c2 = [], float('inf')
    for c1, c2 in pts:            # ascending c1, for equal c1 ascending c2
        if c2 < best_c2:
            front.append((c1, c2))
            best_c2 = c2
    return front


def instance_number(instance):
    m = INSTANCE_NO_RE.search(os.path.basename(instance))
    return int(m.group(1)) if m else 0


def complete_instances(params_rows, registry_path):
    '''Instances for which every job has written its pf.csv (registry may still be filling).'''
    done = {}
    for row in params_rows:
        ok = os.path.exists(os.path.join(registry_path, 'experiments', row['jobid'], 'pf.csv'))
        done[row['instance']] = done.get(row['instance'], True) and ok
    return sorted(i for i, ok in done.items() if ok)


def compute_indicator_table(params_rows, registry_path, algo_cols, instances):
    '''Long DataFrame (instance, class, no, algorithm, repl, hv, hv_ref, hv_indicator).'''
    frames = []
    for k, instance in enumerate(instances, 1):
        print(f'  [{k}/{len(instances)}] {os.path.basename(instance)}', flush = True)
        by_algo = collect_fronts(params_rows, registry_path, instance, algo_cols)
        by_algo_traj = collect_trajectories(params_rows, registry_path, instance, algo_cols)
        ref = determine_reference_point(by_algo, by_algo_traj)
        df = compute_hv_table(by_algo, ref)
        union = [p for repls in by_algo.values() for _, points in repls for p in points]
        hv_ref = hypervolume_2d(nondominated_2d(union), ref)
        df['hv_ref'] = hv_ref
        df['hv_indicator'] = (hv_ref - df['hv']) / hv_ref
        df['instance'] = instance
        df['class'] = infer_instance_class(instance)
        df['no'] = instance_number(instance)
        frames.append(df)
    return pd.concat(frames, ignore_index = True)


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------

def summary_text(tests, algos, alpha, correction, indicator):
    corr = 'Bonferroni-Holm' if correction == 'holm' else 'Benjamini-Hochberg'
    lines = [f'One-sided Wilcoxon-Mann-Whitney tests per instance on {indicator}, alpha = {alpha}, '
             f'{corr} adjustment per instance.',
             'Entry (row A, column B) = number of instances on which A is significantly BETTER than B.', '']
    for cls, g in list(tests.groupby('class', sort = True)) + [('all classes', tests)]:
        sig = g[g['significant']].groupby(['A', 'B']).size().unstack(fill_value = 0)
        sig = sig.reindex(index = algos, columns = algos, fill_value = 0)
        lines.append(f'### {cls}  ({g["instance"].nunique()} instances)')
        lines.append(sig.to_string())
        wins = sig.sum(axis = 1)
        losses = sig.sum(axis = 0)
        lines.append('wins / losses (summed over instances and opponents): '
                     + ', '.join(f'{a} {wins[a]}/{losses[a]}' for a in algos))
        lines.append('')
    return '\n'.join(lines)


def latex_escape(s):
    return str(s).replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')


def latex_table(values, tests, algos, value_col, lower_is_better, alpha, correction, digits, sd_digits):
    '''Tab. 1 of Bossek & Grimme (2023): per instance mean +- sd per algorithm and a stat column.
    A value X^+ in the stat column of algorithm A means: A is significantly better than X.'''
    num = {a: i + 1 for i, a in enumerate(algos)}
    agg = values.groupby(['instance', 'algorithm'])[value_col].agg(['mean', 'std'])
    sig = tests[tests['significant']]
    corr = 'Bonferroni-Holm' if correction == 'holm' else 'Benjamini-Hochberg'
    if value_col == 'hv_indicator':
        what = ('the HV-indicator $(\\mathrm{HV}(R) - \\mathrm{HV}(A)) / \\mathrm{HV}(R)$, where $R$ is the '
                'non-dominated union of all final fronts of the instance (the closer to zero, the better)')
    else:
        what = 'the hypervolume (the higher, the better)'
    better_word = 'lower' if lower_is_better else 'higher'
    best_word = 'Lowest' if lower_is_better else 'Highest'

    out = ['\\begin{table}[t]', '\\centering', '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
           f'\\caption{{Mean values, standard deviations and results of one-sided Wilcoxon-Mann-Whitney tests '
           f'at significance level $\\alpha={alpha}$ with {corr} $p$-value adjustment (\\textbf{{stat}}) with '
           f'respect to {what}. The \\textbf{{stat}}-column is to be read as follows: a value $X^{{+}}$ '
           f'indicates that the indicator of the column algorithm is significantly {better_word} than the one '
           f'of algorithm $X$. {best_word} mean values are highlighted in \\textbf{{bold-face}}.}}',
           '\\label{tab:wmw-hv}',
           '\\begin{adjustbox}{max width=\\textwidth}',
           '\\begin{tabular}{rr' + 'rl' * len(algos) + '}', '\\toprule',
           ' & & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{latex_escape(a)} ({num[a]})}}}}'
                                for a in algos) + ' \\\\',
           ' '.join(f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(algos))),
           'Class & $I$ & ' + ' & '.join(['\\textbf{mean $\\pm$ sd} & \\textbf{stat}'] * len(algos)) + ' \\\\',
           '\\midrule']
    inst_info = values[['instance', 'class', 'no']].drop_duplicates().sort_values(['class', 'no'])
    classes = list(dict.fromkeys(inst_info['class']))
    for c_idx, cls in enumerate(classes):
        block = inst_info[inst_info['class'] == cls]
        for r_idx, (_, row) in enumerate(block.iterrows()):
            inst = row['instance']
            means = {a: agg.loc[(inst, a), 'mean'] for a in algos if (inst, a) in agg.index}
            best = min(means.values()) if lower_is_better else max(means.values())
            cells = []
            for a in algos:
                if a not in means:
                    cells += ['--', '']
                    continue
                m_str = f'{means[a]:.{digits}f}'
                if round(means[a], digits) == round(best, digits):
                    m_str = f'\\textbf{{{m_str}}}'
                sd = agg.loc[(inst, a), 'std']
                beaten = sorted(num[b] for b in sig.loc[(sig['instance'] == inst) & (sig['A'] == a), 'B'])
                cells += [f'{m_str} $\\pm$ {sd:.{sd_digits}f}', ','.join(f'${b}^{{+}}$' for b in beaten)]
            label = f'\\multirow{{{len(block)}}}{{*}}{{{cls}}}' if r_idx == 0 else ''
            out.append(f'{label} & {row["no"]} & ' + ' & '.join(cells) + ' \\\\')
        if c_idx < len(classes) - 1:
            out.append(f'\\cmidrule{{1-{2 + 2 * len(algos)}}}')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
    return '\n'.join(out) + '\n'


def plot_heatmap(fractions, algos, alpha, correction, out_base):
    '''Fig. 10 of Bossek & Grimme (2023): one panel per class, cell (row B, column A) = fraction of
    instances on which the column algorithm A is significantly better than the row algorithm B.'''
    classes = list(fractions)
    k = len(algos)
    cell = 0.62
    fig, axes = plt.subplots(1, len(classes), figsize = (max(2.2, cell * k + 0.6) * len(classes) + 1.2,
                                                          cell * k + 1.9),
                              facecolor = SURFACE, squeeze = False)
    axes = axes[0]
    cmap = LinearSegmentedColormap.from_list('seq', [SEQ_LIGHT, SEQ_DARK])
    cmap.set_bad(GRID)
    for i, (ax, cls) in enumerate(zip(axes, classes)):
        mat = fractions[cls].loc[algos, algos].to_numpy(dtype = float)
        im = ax.imshow(np.ma.masked_invalid(mat), cmap = cmap, vmin = 0, vmax = 1)
        for r in range(k):
            for c in range(k):
                if np.isnan(mat[r, c]):
                    continue
                v = mat[r, c]
                ax.text(c, r, f'{v:.1f}'.rstrip('0').rstrip('.') if v not in (0, 1) else f'{int(v)}',
                        ha = 'center', va = 'center', fontsize = 8,
                        color = SURFACE if v >= 0.55 else INK)
        ax.set_xticks(range(k))
        ax.set_xticklabels(algos, rotation = 45, ha = 'right', fontsize = 8, color = INK)
        ax.set_yticks(range(k))
        ax.set_yticklabels(algos if i == 0 else [''] * k, fontsize = 8, color = INK)
        ax.tick_params(length = 0)
        ax.set_xticks(np.arange(-0.5, k), minor = True)
        ax.set_yticks(np.arange(-0.5, k), minor = True)
        ax.grid(which = 'minor', color = SURFACE, linewidth = 2)
        ax.tick_params(which = 'minor', length = 0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(cls, fontsize = 10, color = INK, loc = 'left')
    axes[0].set_ylabel('row algorithm', fontsize = 9, color = MUTED)
    fig.supxlabel('column algorithm (significantly better than row algorithm?)', fontsize = 9, color = MUTED)
    cbar = fig.colorbar(im, ax = list(axes), fraction = 0.025, pad = 0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize = 7, colors = MUTED, length = 0)
    cbar.set_label('fraction of instances', fontsize = 8, color = MUTED)
    corr = 'Holm' if correction == 'holm' else 'BH'
    fig.suptitle(f'Pairwise one-sided Wilcoxon-Mann-Whitney tests on HV (alpha = {alpha}, {corr})',
                 fontsize = 10, color = INK, x = 0.01, ha = 'left')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', dpi = 200, facecolor = SURFACE, bbox_inches = 'tight')
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def default_algo_col(registry_path):
    header = read_params(os.path.join(registry_path, 'design.csv'))[0].keys()
    for col in ('operator', 'crossover', 'mode', 'mutator'):
        if col in header:
            return col
    raise ValueError('could not guess --algo-col from design.csv, please pass it explicitly')


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry', help = 'expyrimenter registry, e.g. results/approximations/ex10_pure_cx_vs_usgs_pre')
    parser.add_argument('--algo-col', default = None,
                         help = 'design.csv column(s) naming the algorithm, comma-separated for combinations '
                                '(e.g. crossover,mutator). Default: operator / crossover / mode / mutator, '
                                'whichever exists first.')
    parser.add_argument('--order', nargs = '+', default = None,
                         help = 'algorithms to include, in this order (default: all, fixed operator order first)')
    parser.add_argument('--indicator', choices = ['hv-indicator', 'hv'], default = 'hv-indicator')
    parser.add_argument('--alpha', type = float, default = 0.01)
    parser.add_argument('--correction', choices = ['holm', 'bh'], default = 'holm')
    parser.add_argument('--digits', type = int, default = 4, help = 'digits of the mean in the LaTeX table')
    parser.add_argument('--sd-digits', type = int, default = 3, help = 'digits of the sd in the LaTeX table')
    parser.add_argument('--hv-scale', type = float, default = 1e6,
                         help = 'only for --indicator hv: divide HV by this in the LaTeX table')
    parser.add_argument('--reuse-values', action = 'store_true',
                         help = 'reuse wmw_values.csv from --out instead of recomputing all HV values')
    parser.add_argument('--out', default = None, help = 'default: figures/wilcoxon/<registry name>')
    parser.add_argument('--archive-dir', default = 'analysis_log')
    parser.add_argument('--no-archive', action = 'store_true')
    parser.add_argument('--note', default = None, help = 'short note, ends up in summary.md and CHANGELOG.md')
    args = parser.parse_args()

    experiment_label = os.path.basename(os.path.normpath(args.registry))
    out_dir = args.out or os.path.join('figures', 'wilcoxon', experiment_label)
    os.makedirs(out_dir, exist_ok = True)
    values_path = os.path.join(out_dir, 'wmw_values.csv')

    if args.reuse_values and os.path.exists(values_path):
        values = pd.read_csv(values_path)
        print(f'reusing {values_path}')
    else:
        algo_cols = (args.algo_col or default_algo_col(args.registry)).split(',')
        params_rows = read_params(os.path.join(args.registry, 'design.csv'))
        instances = complete_instances(params_rows, args.registry)
        print(f'{len(instances)} of {len(set(r["instance"] for r in params_rows))} instances complete, '
              f'algorithm column(s): {",".join(algo_cols)}')
        rows = [r for r in params_rows if r['instance'] in instances]
        values = compute_indicator_table(rows, args.registry, algo_cols, instances)
        values.to_csv(values_path, index = False)

    found = set(values['algorithm'])
    if args.order:
        missing = [a for a in args.order if a not in found]
        if missing:
            raise SystemExit(f'unknown algorithm(s) {missing}; available: {sorted(found)}')
        algos = args.order
    else:
        algos = [a for a in OPERATOR_ORDER if a in found] + sorted(found - set(OPERATOR_ORDER))
    values = values[values['algorithm'].isin(algos)]

    value_col = 'hv_indicator' if args.indicator == 'hv-indicator' else 'hv'
    lower_is_better = value_col == 'hv_indicator'
    tests = run_pairwise_tests(values, algos, value_col, lower_is_better, args.alpha, args.correction)
    tests.to_csv(os.path.join(out_dir, 'wmw_tests.csv'), index = False)

    text = summary_text(tests, algos, args.alpha, args.correction, args.indicator)
    with open(os.path.join(out_dir, 'wmw_summary.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)

    table_values = values if lower_is_better else values.assign(hv = values['hv'] / args.hv_scale)
    with open(os.path.join(out_dir, 'wmw_table.tex'), 'w') as f:
        f.write(latex_table(table_values, tests, algos, value_col, lower_is_better, args.alpha,
                            args.correction, args.digits, args.sd_digits))
    plot_heatmap(significance_fractions(tests, algos), algos, args.alpha, args.correction,
                 os.path.join(out_dir, 'wmw_heatmap'))
    print(f'\nwritten to {out_dir}')

    if not args.no_archive:
        changelog_rows = []
        for cls, g in tests.groupby('class', sort = True):
            sig = g[g['significant']]
            for a in algos:
                changelog_rows.append({'class': cls, 'algorithm': a,
                                       'wins': int((sig['A'] == a).sum()), 'losses': int((sig['B'] == a).sum()),
                                       'n_inst': g['instance'].nunique()})
        # own sub-folder (analysis_log/<experiment>/wilcoxon/) so that this script's CHANGELOG.md
        # table does not get mixed with the differently-shaped tables of the other scripts
        entry_dir = archive_generic_run(
            args.archive_dir, os.path.join(experiment_label, 'wilcoxon'), 'wilcoxon',
            title = f'Wilcoxon-Mann-Whitney tests ({args.indicator}, alpha={args.alpha}, {args.correction})',
            note = args.note,
            copy_paths = [os.path.join(out_dir, f) for f in
                          ('wmw_summary.txt', 'wmw_tests.csv', 'wmw_table.tex', 'wmw_heatmap.png', 'wmw_values.csv')],
            changelog_header = '| Datum | Commit | Klasse | Operator | sign. Siege | sign. Niederlagen | Instanzen | Notiz |\n'
                               '|---|---|---|---|---|---|---|---|\n',
            row_template = '| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {algorithm} | '
                           '{wins} | {losses} | {n_inst} | {note} |',
            changelog_rows = changelog_rows)
        print(f'archived in: {entry_dir}')


if __name__ == '__main__':
    main()
