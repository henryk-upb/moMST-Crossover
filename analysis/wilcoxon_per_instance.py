#!/usr/bin/env python
'''
Per-instance pairwise Wilcoxon-Mann-Whitney (rank-sum) tests on the final HV,
following the protocol of Bossek & Grimme (ECJ 2023, GECCO 2024) and the
GECCO 2026 USGS paper:

  * for every instance and every ordered pair (A, B) of algorithms a ONE-SIDED
    Wilcoxon-Mann-Whitney test with H1: HV(A) is stochastically larger than HV(B)
    (= A is better, since higher HV is better),
  * p-values adjusted per instance for multiple testing (Holm by default,
    Benjamini-Hochberg via --correction bh),
  * significance level alpha = 0.01 by default,
  * Vargha-Delaney A12 as effect size (P[HV_A > HV_B] + 0.5 P[HV_A = HV_B]).

Why rank-sum (unpaired) and not the signed-rank test (paired)?
  With R = 10 repetitions the exact two-sided signed-rank test cannot go below
  p = 2/2^10 ~ 0.00195; after Holm over 6 pairs (4 algorithms) the smallest
  attainable adjusted p is ~0.0117 > 0.01, i.e. NO difference could ever become
  significant. The rank-sum test reaches p ~ 1e-5 with 10 vs. 10 runs. The shared
  seed only fixes the initial population, so treating runs as independent
  samples is the standard (and conservative-enough) choice.

Input: a CSV with columns algorithm, repl, hv, instance (and optionally class),
e.g. figures/operators_by_class/<experiment>/hv_by_block.csv.

Outputs (in --out, default next to the input CSV):
  wmw_tests.csv        one row per (instance, A, B): p_raw, p_adj, A12, significant
  wmw_summary.txt      per class: on how many instances A is significantly better than B
  wmw_table.tex        Bossek-style table: mean +- sd and stat column (X^+ notation)

Usage:
  python analysis/wilcoxon_per_instance.py figures/operators_by_class/ex10_pure_cx_vs_usgs_pre/hv_by_block.csv
  python analysis/wilcoxon_per_instance.py <csv> --order UNION INTER XOR USGS-PRE --alpha 0.01 --correction holm
'''

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def p_adjust(pvals, method = 'holm'):
    '''Holm or Benjamini-Hochberg adjusted p-values (order-preserving w.r.t. input).'''
    p = np.asarray(pvals, dtype = float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == 'holm':
        running = 0.0
        for k, i in enumerate(order):
            running = max(running, (m - k) * p[i])
            adj[i] = min(1.0, running)
    elif method == 'bh':
        running = 1.0
        for k in range(m - 1, -1, -1):
            i = order[k]
            running = min(running, p[i] * m / (k + 1))
            adj[i] = running
    else:
        raise ValueError(f'unknown correction {method}')
    return adj


def vargha_delaney_a12(x, y):
    '''P(X > Y) + 0.5 P(X = Y); 0.5 = no effect, >0.71 / <0.29 usually called "large".'''
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def one_sided_wmw(x, y):
    '''p-value for H1: x stochastically larger than y. Exact null distribution if there
    are no ties (the usual case for HV), otherwise normal approximation with tie correction.'''
    has_ties = len(np.unique(np.concatenate([x, y]))) < len(x) + len(y)
    method = 'asymptotic' if has_ties else 'exact'
    return stats.mannwhitneyu(x, y, alternative = 'greater', method = method).pvalue


def infer_class(instance):
    m = re.search(r'CLASS(\d)', os.path.basename(str(instance)))
    return f'C{m.group(1)}' if m else 'all'


def instance_number(instance):
    m = re.search(r'_(\d+)\.graph$', str(instance))
    return int(m.group(1)) if m else 0


def run_tests(df, algos, alpha, correction):
    rows = []
    for inst, g in df.groupby('instance'):
        samples = {a: g.loc[g['algorithm'] == a, 'hv'].to_numpy() for a in algos}
        inst_rows = []
        for a in algos:
            for b in algos:
                if a == b:
                    continue
                inst_rows.append(dict(instance = inst, cls = g['class'].iloc[0], no = instance_number(inst),
                                      A = a, B = b, n_A = len(samples[a]), n_B = len(samples[b]),
                                      median_A = np.median(samples[a]), median_B = np.median(samples[b]),
                                      p_raw = one_sided_wmw(samples[a], samples[b]),
                                      A12 = vargha_delaney_a12(samples[a], samples[b])))
        # correction over all k(k-1) one-sided tests of this instance
        adj = p_adjust([r['p_raw'] for r in inst_rows], correction)
        for r, pa in zip(inst_rows, adj):
            r['p_adj'] = pa
            r['significant'] = pa < alpha
        rows.extend(inst_rows)
    return pd.DataFrame(rows)


def summary_text(tests, algos, alpha, correction):
    lines = [f'One-sided Wilcoxon-Mann-Whitney tests on final HV, alpha = {alpha}, '
             f'{correction}-adjusted per instance.',
             'Entry (row A, column B) = number of instances on which A has a significantly HIGHER HV than B.\n']
    for cls, g in tests.groupby('cls'):
        mat = g[g['significant']].groupby(['A', 'B']).size().unstack(fill_value = 0)
        mat = mat.reindex(index = algos, columns = algos, fill_value = 0)
        lines.append(f'### {cls}  ({g["instance"].nunique()} instances)')
        lines.append(mat.to_string())
        lines.append('')
    return '\n'.join(lines)


def latex_table(df, tests, algos, alpha, correction, digits, scale = 1.0):
    '''Bossek-style table: per instance mean +- sd and stat column; X^+ in the column of
    algorithm A means: A is significantly better (higher HV) than algorithm X.'''
    idx = {a: i + 1 for i, a in enumerate(algos)}
    scale_note = f' HV values are divided by $10^{{{int(round(np.log10(scale)))}}}$.' if scale != 1 else ''
    agg = df.groupby(['instance', 'algorithm'])['hv'].agg(['mean', 'std']).reset_index()
    out = []
    out.append('\\begin{table}[t]')
    out.append('\\centering\\scriptsize')
    corr = 'Holm' if correction == 'holm' else 'Benjamini-Hochberg'
    out.append(f'\\caption{{Mean $\\pm$ sd of the HV-indicator (higher is better) and results of one-sided '
               f'Wilcoxon-Mann-Whitney tests at significance level $\\alpha={alpha}$ with {corr} '
               f'$p$-value adjustment (\\textbf{{stat}}). A value $X^{{+}}$ indicates that the column '
               f'algorithm has a significantly higher HV than algorithm $X$. '
               f'Highest mean values are in bold-face.{scale_note}}}')
    out.append('\\label{tab:wmw-hv}')
    out.append('\\begin{tabular}{rr' + 'rl' * len(algos) + '}')
    out.append('\\toprule')
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{a} ({idx[a]})}}}}' for a in algos)
    out.append(f' & & {head} \\\\')
    out.append(' '.join(f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(algos))))
    out.append('Class & $I$ & ' + ' & '.join(['mean $\\pm$ sd & stat'] * len(algos)) + ' \\\\')
    out.append('\\midrule')
    insts = (df[['instance', 'class']].drop_duplicates()
             .assign(no = lambda d: d['instance'].map(instance_number))
             .sort_values(['class', 'no']))
    prev_cls = None
    for _, row in insts.iterrows():
        inst, cls, no = row['instance'], row['class'], row['no']
        if prev_cls is not None and cls != prev_cls:
            out.append('\\midrule')
        a_ = agg[agg['instance'] == inst].set_index('algorithm')
        best = a_['mean'].idxmax()
        t = tests[(tests['instance'] == inst) & tests['significant']]
        cells = []
        for a in algos:
            mean, sd = a_.loc[a, 'mean'], a_.loc[a, 'std']
            m_str = f'{mean:.{digits}f}'
            if a == best:
                m_str = f'\\textbf{{{m_str}}}'
            beaten = sorted(idx[b] for b in t.loc[t['A'] == a, 'B'])
            stat = ','.join(f'${b}^{{+}}$' for b in beaten)
            cells += [f'{m_str} $\\pm$ {sd:.{digits}f}', stat]
        label = cls if cls != prev_cls else ''
        out.append(f'{label} & {no} & ' + ' & '.join(cells) + ' \\\\')
        prev_cls = cls
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    out.append('\\end{table}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', help = 'CSV with columns algorithm, repl, hv, instance[, class]')
    ap.add_argument('--order', nargs = '+', default = None, help = 'algorithm order (default: as found)')
    ap.add_argument('--alpha', type = float, default = 0.01)
    ap.add_argument('--correction', choices = ['holm', 'bh'], default = 'holm')
    ap.add_argument('--scale', type = float, default = 1e6,
                    help = 'divide HV by this for the LaTeX table (default 1e6; state it in the caption)')
    ap.add_argument('--digits', type = int, default = 2)
    ap.add_argument('--out', default = None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'class' not in df.columns:
        df['class'] = df['instance'].map(infer_class)
    algos = args.order or list(dict.fromkeys(df['algorithm']))
    df = df[df['algorithm'].isin(algos)]
    out = args.out or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out, exist_ok = True)

    tests = run_tests(df, algos, args.alpha, args.correction)
    tests.to_csv(os.path.join(out, 'wmw_tests.csv'), index = False)

    text = summary_text(tests, algos, args.alpha, args.correction)
    with open(os.path.join(out, 'wmw_summary.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)

    df_tab = df.assign(hv = df['hv'] / args.scale)
    with open(os.path.join(out, 'wmw_table.tex'), 'w') as f:
        f.write(latex_table(df_tab, tests, algos, args.alpha, args.correction, args.digits, args.scale) + '\n')
    print(f'written to {out}/wmw_tests.csv, wmw_summary.txt, wmw_table.tex')


if __name__ == '__main__':
    main()
