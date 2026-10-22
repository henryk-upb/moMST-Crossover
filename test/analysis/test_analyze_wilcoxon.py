'''Unit tests for analysis/analyze_wilcoxon.py (run with `make test` / `pytest -v`).'''

import os
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')
pytest.importorskip('pymoo')

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'analysis')
sys.path.insert(0, os.path.abspath(ANALYSIS_DIR))

from scipy import stats  # noqa: E402
import analyze_wilcoxon as aw  # noqa: E402


def make_values(rng, means, n_inst = 3, repl = 10, sd = 1.0, cls = 'C1'):
    '''Synthetic raw-HV table: algorithm -> mean HV (higher is better).'''
    rows = []
    for i in range(1, n_inst + 1):
        for algo, mu in means.items():
            for r in range(1, repl + 1):
                rows.append({'instance': f'instances/100/CLASS1_x_{i}.graph', 'class': cls, 'no': i,
                             'algorithm': algo, 'repl': r, 'hv': rng.normal(mu, sd)})
    df = pd.DataFrame(rows)
    hv_ref = df.groupby('instance')['hv'].transform('max') * 1.01
    df['hv_indicator'] = (hv_ref - df['hv']) / hv_ref
    return df


def test_exact_one_sided_minimum_p_value():
    # all 10 values of x below all 10 values of y: exactly 1 of C(20, 10) rank assignments
    p = aw.one_sided_wmw(np.arange(10), np.arange(10, 20), 'less')
    assert p == pytest.approx(1 / comb(20, 10))
    assert aw.one_sided_wmw(np.arange(10), np.arange(10, 20), 'greater') == pytest.approx(1.0)


def test_identical_constant_samples_are_not_significant():
    assert aw.one_sided_wmw(np.ones(10), np.ones(10)) == 1.0


def test_signed_rank_cannot_reach_alpha_with_ten_repls():
    '''Documents why the paired test is not used: even with all 10 differences positive the
    Holm-adjusted p over 6 pairs stays above 0.01.'''
    p_min = stats.wilcoxon(np.arange(1, 11), np.zeros(10)).pvalue
    assert p_min == pytest.approx(2 / 2 ** 10)
    assert 6 * p_min > 0.01


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert aw.p_adjust(p, 'holm') == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert aw.p_adjust(p, 'bh') == pytest.approx([0.02, 0.04, 0.04, 0.02])
    with pytest.raises(ValueError):
        aw.p_adjust(p, 'bonferroni')


def test_vargha_delaney():
    assert aw.vargha_delaney_a12([1, 2, 3], [4, 5, 6]) == 1.0          # lower is better
    assert aw.vargha_delaney_a12([1, 2, 3], [4, 5, 6], lower_is_better = False) == 0.0
    assert aw.vargha_delaney_a12([1, 1], [1, 1]) == 0.5


def test_nondominated_2d():
    pts = [(1, 5), (2, 3), (2, 4), (3, 3), (4, 1), (4, 1), (5, 5)]
    assert aw.nondominated_2d(pts) == [(1, 5), (2, 3), (4, 1)]


def test_clear_winner_is_detected_on_every_instance():
    rng = np.random.default_rng(1)
    df = make_values(rng, {'A': 110, 'B': 100, 'C': 100})
    tests = aw.run_pairwise_tests(df, ['A', 'B', 'C'], 'hv_indicator', True, 0.01, 'holm')
    sig = tests[tests['significant']]
    assert set(map(tuple, sig[['A', 'B']].drop_duplicates().to_numpy())) == {('A', 'B'), ('A', 'C')}
    assert len(sig) == 2 * 3                     # both opponents on all 3 instances
    assert (tests.loc[tests['A'] == 'A', 'A12'] > 0.9).all()


def test_indicator_and_raw_hv_give_identical_decisions():
    rng = np.random.default_rng(2)
    df = make_values(rng, {'A': 101, 'B': 100, 'C': 99.5}, n_inst = 4)
    t_ind = aw.run_pairwise_tests(df, ['A', 'B', 'C'], 'hv_indicator', True)
    t_hv = aw.run_pairwise_tests(df, ['A', 'B', 'C'], 'hv', False)
    assert (t_ind['significant'] == t_hv['significant']).all()
    assert t_ind['p_raw'].to_numpy() == pytest.approx(t_hv['p_raw'].to_numpy())


def test_fractions_and_latex_table():
    rng = np.random.default_rng(3)
    algos = ['A', 'B']
    df = make_values(rng, {'A': 110, 'B': 100}, n_inst = 2)
    tests = aw.run_pairwise_tests(df, algos)
    frac = aw.significance_fractions(tests, algos)['C1']
    assert frac.loc['B', 'A'] == 1.0 and frac.loc['A', 'B'] == 0.0   # row B, column A: A beats B
    tex = aw.latex_table(df, tests, algos, 'hv_indicator', True, 0.01, 'holm', 4, 3)
    assert tex.count('$2^{+}$') == 2                                   # A beats algorithm 2 on both instances
    assert '\\multirow{2}{*}{C1}' in tex
