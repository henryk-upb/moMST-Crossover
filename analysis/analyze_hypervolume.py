#!/usr/bin/env python
'''
Hypervolume-basierte Analyse von K Algorithmen auf einer einzelnen Instanz.

Strategie:
  Fuer eine Instanz liegen K Algorithmen (hier: die Crossover-Operatoren
  UNION/INTER/XOR aus ex05, ueber --algo-col austauschbar, z.B. gegen einen
  reinen Mutations-Baseline-Lauf) mit je R Wiederholungen vor, d.h. R x K
  Pareto-Front-Approximationen (pf.csv je Job). Jede Front wird ueber einen
  gemeinsamen, instanzweiten Referenzpunkt auf einen skalaren Hypervolume-
  Indikator reduziert. Man erhaelt so K empirische Verteilungen von je R
  HV-Werten, die

  * visuell verglichen werden (Boxplot, Violinplot, Histogramme, ECDF) und
  * statistisch getestet werden: da ex05 fuer jede Wiederholung repl=k
    denselben Seed ueber alle Algorithmen verwendet (common random numbers),
    sind die Wiederholungen ueber die Algorithmen hinweg gepaart. Als
    Omnibus-Test dient daher der Friedman-Test, gefolgt von paarweisen
    Wilcoxon-Vorzeichen-Rang-Tests mit Holm-Korrektur.

  Zusaetzlich: Entwicklung ueber die Zeit. Statt nur der finalen Front
  (pf.csv) wird auch trajectory.csv ausgewertet, die T Zwischenstaende der
  Population bei 0%, 10%, ..., 100% der maximalen Generationen enthaelt
  (log_every in EA/Algorithms.py). Jeder dieser T Zwischenstaende wird wie
  die finale Front auf einen HV-Wert reduziert -> T x R x K Werte. Der
  Verlauf von Mittelwert/Median je Algorithmus ueber die Generationen wird
  als Linienplot dargestellt (hv_progress.png), analog zur ECJ-2023-Studie.

  Alles oben bezieht sich auf eine einzelne Instanz. Fuer eine instanz-
  UEBERGREIFENDE Aggregation (--all-instances) werden die rohen HV-Werte
  aller Instanzen direkt gepoolt und geplottet (auf Wunsch, statt Rang-
  basiert). Achtung: die Skala von HV haengt von der Instanz/n ab, d.h. die
  Streuung je Algorithmus in den Plots wird von Instanz-zu-Instanz-
  Unterschieden mitdominiert, nicht nur von Algorithmus-Unterschieden.
  compute_cross_instance_table() berechnet trotzdem 'rank'/'relative_pct'
  je (Instanz, Repl)-Block mit in hv_cross_instance.csv, falls eine
  skalenunabhaengige Sicht spaeter doch gebraucht wird. Friedman-/Wilcoxon-
  Tests laufen geblockt ueber (Instanz, Repl) statt nur ueber Repl -- das
  bleibt korrekt (Friedman rankt intern pro Block), unabhaengig davon, ob
  rohe hv oder rank als value_col uebergeben wird.

Erwartet das Ausgabelayout von *_runner_expyrimenter.py (expyrimenter's
Registry): <registry_path>/design.csv (Spalten jobid, instance, <algo-col>,
repl, ...) und Job-Ergebnisse unter <registry_path>/experiments/<jobid>/pf.csv.

Jeder Lauf wird automatisch dauerhaft archiviert (siehe --archive-dir,
default: analysis_log/), damit aufeinanderfolgende Laeufe sich nicht
gegenseitig ueberschreiben und der Fortschritt ueber die Zeit sichtbar
bleibt. --out bleibt zusaetzlich als schnelle "aktueller Stand"-Ansicht
bestehen und WIRD bei jedem Lauf ueberschrieben.

Sowohl --out als auch --archive-dir werden nach Experiment getrennt: aus
registry_path wird automatisch ein Experiment-Label abgeleitet (der
Ordnername, z.B. "ex05", "ex06", "ex07" -- je nachdem, welcher *_runner_
expyrimenter.py Skript die Daten erzeugt hat), sodass
  figures/hv_analysis/<experiment>/<instanz>/...
  analysis_log/<experiment>/<timestamp>_<commit>_<instanz>/...
  analysis_log/<experiment>/CHANGELOG.md
nie mit den Ergebnissen eines anderen Runners/Experiments vermischt werden
-- wichtig, weil mehrere Runner dieselben Instanzdateien (z.B. instances/50)
verwenden koennen und sonst gleiche instance_label-Ordner kollidieren wuerden.

Nutzung:
  python analysis/analyze_hypervolume.py results/approximations/ex05 \
      --instance instances/50/CLASS1_50_1225_0_2_UNG_CEG_RWG-RWG_1.graph \
      --note "XOR-CX Bugfix: Algorithmus 5 statt fixierter Schnittmenge"
'''

import argparse
import csv
import math
import os
import shutil
import subprocess
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pymoo.indicators.hv import HV


import re

INSTANCE_CLASS_RE = re.compile(r'^CLASS(\d+)_')


def infer_instance_class(instance_path):
    '''Instanzklasse "C<k>" aus dem Dateinamen (z.B. CLASS1_50_..._1.graph -> "C1").
    Absichtlich hier dupliziert statt aus analyze_quality_by_size.py importiert, um
    einen Zirkelimport zu vermeiden (jenes Modul importiert bereits aus diesem hier).'''
    m = INSTANCE_CLASS_RE.match(os.path.basename(instance_path))
    if not m:
        raise ValueError(f"Instanzklasse konnte nicht aus '{instance_path}' abgeleitet werden.")
    return f'C{m.group(1)}'


def read_params(path):
    with open(path, newline = '') as f:
        return list(csv.DictReader(f))


def read_pf(path):
    with open(path, newline = '') as f:
        return [(float(row['c1']), float(row['c2'])) for row in csv.DictReader(f)]


def read_trajectory(path):
    '''gen -> list of (c1, c2) points, i.e. the whole population logged at that generation.'''
    gens = {}
    with open(path, newline = '') as f:
        for row in csv.DictReader(f):
            gens.setdefault(int(row['gen']), []).append((float(row['c1']), float(row['c2'])))
    return gens


def hypervolume_2d(points, ref):
    '''Dominated hypervolume of a minimization front w.r.t. reference point ref, via pymoo.'''
    return HV(ref_point = np.array(ref))(np.array(points))


def get_algo_label(row, algo_cols):
    '''Combines one or more design.csv columns into a single algorithm label,
    e.g. algo_cols=['crossover', 'mutator'] -> "UNION+USGS-F". Needed for designs
    like ex06 where each (crossover, mutator) COMBINATION is its own algorithm,
    not crossover and mutator separately.'''
    return "+".join(row[c] for c in algo_cols)


def collect_fronts(params_rows, registry_path, instance, algo_cols):
    '''algorithm -> [(repl, points), ...] for all jobs of the given instance.'''
    by_algo = {}
    for row in params_rows:
        if row['instance'] != instance:
            continue
        algo = get_algo_label(row, algo_cols)
        pf_path = os.path.join(registry_path, 'experiments', row['jobid'], 'pf.csv')
        points = read_pf(pf_path)
        by_algo.setdefault(algo, []).append((row['repl'], points))
    return by_algo


def collect_trajectories(params_rows, registry_path, instance, algo_cols):
    '''algorithm -> [(repl, {gen: points}), ...] for all jobs of the given instance.'''
    by_algo = {}
    for row in params_rows:
        if row['instance'] != instance:
            continue
        algo = get_algo_label(row, algo_cols)
        traj_path = os.path.join(registry_path, 'experiments', row['jobid'], 'trajectory.csv')
        gens = read_trajectory(traj_path)
        by_algo.setdefault(algo, []).append((row['repl'], gens))
    return by_algo


def determine_reference_point(by_algo, by_algo_traj = None, margin = 0.05):
    '''
    Shared nadir-style reference point across all algorithms/repls of the instance.
    Includes trajectory snapshots (if given) so early, worse generations are also
    dominated by the reference point -- otherwise their HV would be truncated.
    '''
    all_points = [p for repls in by_algo.values() for _, points in repls for p in points]
    if by_algo_traj is not None:
        all_points += [p for repls in by_algo_traj.values() for _, gens in repls
                        for points in gens.values() for p in points]
    xs, ys = zip(*all_points)
    return (max(xs) * (1 + margin), max(ys) * (1 + margin))


def compute_hv_table(by_algo, ref):
    records = [
        {'algorithm': algo, 'repl': int(repl), 'hv': hypervolume_2d(points, ref)}
        for algo, repls in by_algo.items()
        for repl, points in repls
    ]
    return pd.DataFrame.from_records(records)


def compute_hv_progress_table(by_algo_traj, ref):
    '''Long-format DataFrame: algorithm, repl, gen, pct (of max gen), hv -- one row per (algo, repl, snapshot).'''
    max_gen = max(gen for repls in by_algo_traj.values() for _, gens in repls for gen in gens)
    records = [
        {'algorithm': algo, 'repl': int(repl), 'gen': gen, 'pct': (100.0 * gen / max_gen) if max_gen > 0 else 0.0,
         'hv': hypervolume_2d(points, ref)}
        for algo, repls in by_algo_traj.items()
        for repl, gens in repls
        for gen, points in gens.items()
    ]
    return pd.DataFrame.from_records(records)


def compute_cross_instance_table(params_rows, registry_path, algo_cols):
    '''
    Runs the per-instance HV pipeline (own reference point per instance, since HV
    scale is instance-dependent) for EVERY instance in params_rows, then combines
    all instances into one long DataFrame with two instance-independent measures
    per (instance, repl, algorithm) block:
      * rank: 1 = best (highest HV) among the K algorithms in that block
      * relative_pct: 100 * hv / (best hv in that block) -- 100% = tied for best
    Raw hv itself is NOT comparable across instances (different scale/n), so
    aggregate plots/tests should use rank or relative_pct, not hv, directly.
    '''
    instances = sorted(set(row['instance'] for row in params_rows))
    frames = []
    for instance in instances:
        by_algo = collect_fronts(params_rows, registry_path, instance, algo_cols)
        if not by_algo:
            continue
        by_algo_traj = collect_trajectories(params_rows, registry_path, instance, algo_cols)
        ref = determine_reference_point(by_algo, by_algo_traj)
        df = compute_hv_table(by_algo, ref)
        df['instance'] = instance
        frames.append(df)

    df_all = pd.concat(frames, ignore_index = True)
    df_all['rank'] = df_all.groupby(['instance', 'repl'])['hv'].rank(ascending = False, method = 'average')
    block_max = df_all.groupby(['instance', 'repl'])['hv'].transform('max')
    df_all['relative_pct'] = 100.0 * df_all['hv'] / block_max
    return df_all


def holm_correction(pvals):
    '''Holm-Bonferroni adjusted p-values, order-preserving w.r.t. the input.'''
    m = len(pvals)
    order = sorted(range(m), key = lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, pvals[i] * (m - rank))
        adjusted[i] = min(running_max, 1.0)
    return adjusted


def plot_distributions(df, out_dir, instance_label):
    os.makedirs(out_dir, exist_ok = True)
    algos = sorted(df['algorithm'].unique())
    data = [df.loc[df['algorithm'] == a, 'hv'].values for a in algos]
    n = len(algos)
    # rotate + shrink labels once there are more than a handful of categories,
    # and widen the figure so they don't overlap (e.g. ex06's 15 crossover+mutator combos)
    rotate = n > 5
    width = max(5, 0.6 * n + 1.5)
    label_fontsize = 9 if n <= 15 else 7

    fig, ax = plt.subplots(figsize = (width, 4.5))
    ax.boxplot(data, tick_labels = algos, showmeans = True)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV Distribution (Boxplot)\n{instance_label}')
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', fontsize = label_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_boxplot.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (width, 4.5))
    ax.violinplot(data, showmeans = True, showmedians = True)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(algos)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV Distribution (Violin Plot)\n{instance_label}')
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', fontsize = label_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_violinplot.png'), dpi = 150)
    plt.close(fig)

    # grid instead of a single row once there are many algorithms, so the figure
    # grows in both directions instead of becoming absurdly wide
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize = (3.2 * ncols, 3 * nrows), sharey = True, squeeze = False)
    axes_flat = axes.flatten()
    for ax, algo in zip(axes_flat, algos):
        ax.hist(df.loc[df['algorithm'] == algo, 'hv'], bins = 10, color = 'tab:blue', alpha = 0.8)
        ax.set_title(algo, fontsize = 10)
        ax.set_xlabel('Hypervolume', fontsize = 9)
        ax.tick_params(axis = 'x', labelrotation = 30, labelsize = 8)
    for ax in axes_flat[n:]:
        ax.axis('off')
    axes_flat[0].set_ylabel('Number of Repetitions')
    fig.suptitle(f'HV Distribution (Histograms)\n{instance_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_histograms.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (width + 2, 4.5))
    for algo in algos:
        vals = np.sort(df.loc[df['algorithm'] == algo, 'hv'].values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where = 'post', label = algo)
    ax.set_xlabel('Hypervolume')
    ax.set_ylabel('F(x)')
    ax.set_title(f'Empirical Distribution Function (ECDF)\n{instance_label}')
    if n > 8:
        ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8, ncol = 1 + n // 20)
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_ecdf.png'), dpi = 150)
    plt.close(fig)


def plot_progress(df_progress, out_dir, instance_label):
    '''Line plot of mean HV over generations (as % of max generation), one line per algorithm,
    with a shaded band spanning the 25%-75% quantile across repetitions at each snapshot.'''
    os.makedirs(out_dir, exist_ok = True)
    algos = sorted(df_progress['algorithm'].unique())
    n = len(algos)

    fig, ax = plt.subplots(figsize = (6 + 2 if n > 8 else 6, 4.5))
    for algo in algos:
        sub = df_progress[df_progress['algorithm'] == algo]
        agg = sub.groupby('pct')['hv'].agg(['mean', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        agg.columns = ['mean', 'q25', 'q75']
        agg = agg.sort_index()
        line, = ax.plot(agg.index, agg['mean'], marker = 'o', label = algo)
        ax.fill_between(agg.index, agg['q25'], agg['q75'], color = line.get_color(), alpha = 0.15)

    ax.set_xlabel('% of Maximum Generations')
    ax.set_ylabel('Hypervolume (Mean over Repetitions)')
    ax.set_title(f'HV Progress over Time\n{instance_label}')
    if n > 8:
        ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8, ncol = 1 + n // 20)
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_progress.png'), dpi = 150)
    plt.close(fig)


def plot_distributions_by_class(by_class, out_dir):
    '''Facet-Grid-Variante von plot_distributions: EIN Bild je Diagrammtyp
    (Violin/ECDF) mit einem Panel je Instanzklasse, statt vier separate Bilder
    je Klasse -- rohe HV bleibt dabei nie ueber Instanzen gemischt, jedes Panel
    zeigt weiterhin genau eine Instanz. by_class: {class_label: (df, instance_label)}.
    Kein Boxplot hier: quality_by_class_boxplot.png (analyze_quality_by_size.py
    --facet-by class) zeigt exakt dieselben rohen HV-Werte bereits als Boxplot,
    ein zweiter waere reine Duplikation.'''
    os.makedirs(out_dir, exist_ok = True)
    classes = sorted(by_class.keys())
    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)

    def new_grid(figsize_per_panel):
        fig, axes = plt.subplots(nrows, ncols, squeeze = False,
                                  figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
        return fig, axes.flatten()

    paths = {}

    fig, axes_flat = new_grid((5, 4.3))
    for ax, cls in zip(axes_flat, classes):
        df, instance_label = by_class[cls]
        algos = sorted(df['algorithm'].unique())
        data = [df.loc[df['algorithm'] == a, 'hv'].values for a in algos]
        n = len(algos)
        ax.violinplot(data, showmeans = True, showmedians = True)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(algos)
        ax.set_title(f'Class {cls}\n{instance_label}', fontsize = 9)
        ax.set_ylabel('Hypervolume')
        plt.setp(ax.get_xticklabels(), rotation = 45 if n > 5 else 0,
                 ha = 'right' if n > 5 else 'center', fontsize = 8)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')
    fig.suptitle('HV Distribution (Violin Plot) -- per instance class')
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    paths['violinplot'] = os.path.join(out_dir, 'hv_violinplot.png')
    fig.savefig(paths['violinplot'], dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    fig, axes_flat = new_grid((5.5, 4.3))
    for ax, cls in zip(axes_flat, classes):
        df, instance_label = by_class[cls]
        algos = sorted(df['algorithm'].unique())
        for algo in algos:
            vals = np.sort(df.loc[df['algorithm'] == algo, 'hv'].values)
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where = 'post', label = algo)
        ax.set_xlabel('Hypervolume')
        ax.set_ylabel('F(x)')
        ax.set_title(f'Class {cls}\n{instance_label}', fontsize = 9)
        ax.legend(fontsize = 7)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')
    fig.suptitle('Empirical Distribution Function (ECDF) -- per instance class')
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    paths['ecdf'] = os.path.join(out_dir, 'hv_ecdf.png')
    fig.savefig(paths['ecdf'], dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    return paths


def plot_cross_instance(df_cross, out_dir, n_instances):
    '''
    Aggregate view pooling ALL instances together (one point per algorithm per
    (instance, repl) block), using raw HV values directly, sorted best-to-worst
    by mean HV:
      * hv_boxplot.png / hv_violinplot.png -- absolute HV distribution
      * hv_ecdf.png -- ECDF of absolute HV

    Note: raw HV is only really comparable within one instance (its scale
    depends on the instance/n); pooling it across instances of different
    scale/size means the spread within each algorithm's box is dominated by
    inter-instance differences, not just inter-algorithm ones. Kept as
    plain/absolute values on request; see compute_cross_instance_table's
    'rank'/'relative_pct' columns in hv_cross_instance.csv for a
    scale-independent alternative if needed later.
    '''
    os.makedirs(out_dir, exist_ok = True)
    mean_hv = df_cross.groupby('algorithm')['hv'].mean().sort_values(ascending = False)
    algos = list(mean_hv.index)
    n = len(algos)
    rotate = n > 5
    width = max(5, 0.6 * n + 1.5)
    label_fontsize = 9 if n <= 15 else 7
    subtitle = f"pooled over {n_instances} instances"

    data = [df_cross.loc[df_cross['algorithm'] == a, 'hv'].values for a in algos]

    fig, ax = plt.subplots(figsize = (width, 4.5))
    ax.boxplot(data, tick_labels = algos, showmeans = True)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV Distribution over All Instances (sorted by Mean HV)\n{subtitle}')
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', fontsize = label_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_boxplot.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (width, 4.5))
    ax.violinplot(data, showmeans = True, showmedians = True)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(algos)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV Distribution over All Instances\n{subtitle}')
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', fontsize = label_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_violinplot.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (width, 4.5))
    for algo in algos:
        vals = np.sort(df_cross.loc[df_cross['algorithm'] == algo, 'hv'].values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where = 'post', label = algo)
    ax.set_xlabel('Hypervolume')
    ax.set_ylabel('F(x)')
    ax.set_title(f'ECDF of HV over All Instances\n{subtitle}')
    if n > 8:
        ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8, ncol = 1 + n // 20)
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_ecdf.png'), dpi = 150)
    plt.close(fig)


def run_statistics(df, block_cols = 'repl', value_col = 'hv', block_label = 'gepaarte Wiederholungen',
                    higher_is_better = True):
    '''Runs the descriptive + inferential statistics and returns (report_text, desc_table).
    block_cols is the pairing/blocking key for the Friedman/Wilcoxon tests -- 'repl' for a single
    instance, ['instance', 'repl'] to pool across all instances (each (instance, repl) is one block;
    Friedman ranks WITHIN each block internally, so raw hv from differently-scaled instances is still
    valid input as long as block_cols keeps them from being compared across instances directly).
    Set higher_is_better = False for value_col = 'rank', where lower is better.'''
    algos = sorted(df['algorithm'].unique())
    pivot = df.pivot(index = block_cols, columns = 'algorithm', values = value_col).dropna()
    lines = []
    better = (lambda x, y: x > y) if higher_is_better else (lambda x, y: x < y)

    lines.append(f"=== Deskriptive Statistik (n = {len(pivot)} {block_label}) ===")
    desc = pivot.describe().T[['count', 'mean', 'std', '50%', 'min', 'max']].rename(columns = {'50%': 'median'})
    lines.append(desc.to_string())

    if len(algos) >= 3:
        stat, p = stats.friedmanchisquare(*[pivot[a] for a in algos])
        lines.append(f"\nFriedman-Test (gepaart ueber {block_label}, {len(algos)} Gruppen): chi2 = {stat:.4f}, p = {p:.4g}")
    elif len(algos) == 2:
        stat, p = stats.wilcoxon(pivot[algos[0]], pivot[algos[1]])
        lines.append(f"\nWilcoxon-Vorzeichen-Rang-Test ({algos[0]} vs. {algos[1]}): W = {stat:.4f}, p = {p:.4g}")

    if len(algos) >= 3:
        lines.append("\n=== Paarweise Wilcoxon-Vorzeichen-Rang-Tests (Holm-korrigiert) ===")
        pairs = list(combinations(algos, 2))
        raw_p = []
        for a, b in pairs:
            try:
                _, p = stats.wilcoxon(pivot[a], pivot[b])
            except ValueError:
                p = float('nan')
            raw_p.append(p)
        adj_p = holm_correction(raw_p)
        for (a, b), p_raw, p_adj in zip(pairs, raw_p, adj_p):
            direction = a if better(pivot[a].median(), pivot[b].median()) else b
            lines.append(f"{a} vs. {b}: p_raw = {p_raw:.4g}, p_holm = {p_adj:.4g}"
                          f"  (besser im Median: {direction})")

    return "\n".join(lines), desc


def get_git_info():
    '''(commit, subject, dirty_diffstat) of the current HEAD; empty strings if not in a git repo.'''
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text = True).strip()
        subject = subprocess.check_output(['git', 'log', '-1', '--format=%s'], text = True).strip()
        diffstat = subprocess.check_output(['git', 'diff', '--stat'], text = True).strip()
        return commit, subject, diffstat
    except (subprocess.CalledProcessError, FileNotFoundError):
        return '', '', ''


DEFAULT_PLOT_NAMES = ['hv_boxplot.png', 'hv_violinplot.png', 'hv_histograms.png', 'hv_ecdf.png', 'hv_progress.png']


def archive_run(archive_root, experiment_label, instance_label, algo_col, report_text, desc, note, plot_dir, csv_path,
                 progress_csv_path = None, extra_plot_names = None):
    '''Copies plots + raw HV values into a timestamped entry under archive_root/experiment_label/ and
    appends a row per algorithm to archive_root/experiment_label/CHANGELOG.md. experiment_label is the
    basename of the registry_path (e.g. "ex05", "ex06", "ex07"), so different runners/experiments never
    mix in the same folder or table.'''
    commit, subject, diffstat = get_git_info()
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H%M%S')
    commit_label = f"{commit}-dirty" if diffstat else (commit or "no-git")

    archive_root = os.path.join(archive_root, experiment_label)
    entry_dir = os.path.join(archive_root, f"{timestamp}_{commit_label}_{instance_label}")
    os.makedirs(entry_dir, exist_ok = True)
    for fname in extra_plot_names if extra_plot_names is not None else DEFAULT_PLOT_NAMES:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            shutil.copy(fpath, entry_dir)
    shutil.copy(csv_path, entry_dir)
    if progress_csv_path is not None and os.path.exists(progress_csv_path):
        shutil.copy(progress_csv_path, entry_dir)

    with open(os.path.join(entry_dir, 'summary.md'), 'w') as f:
        f.write(f"# HV-Analyse: {instance_label}\n\n")
        f.write(f"- Experiment: `{experiment_label}`\n")
        f.write(f"- Datum: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})" + (" -- **uncommitted changes!**\n" if diffstat else "\n"))
        f.write(f"- Vergleichsspalte: `{algo_col}`\n")
        if note:
            f.write(f"\n## Was hat sich seit dem letzten Eintrag geaendert\n\n{note}\n")
        if diffstat:
            f.write(f"\n## Uncommitted Aenderungen zum Zeitpunkt des Laufs\n\n```\n{diffstat}\n```\n")
        f.write(f"\n## Ergebnisse\n\n```\n{report_text}\n```\n")

    index_path = os.path.join(archive_root, 'CHANGELOG.md')
    is_new = not os.path.exists(index_path)
    with open(index_path, 'a') as f:
        if is_new:
            f.write("# HV-Analyse-Verlauf\n\n")
            f.write("| Datum | Commit | Instanz | Algorithmus | Median HV | Mean HV | n | Notiz |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
        rel_dir = os.path.relpath(entry_dir, archive_root)
        for algo, row in desc.iterrows():
            f.write(f"| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {instance_label} | {algo} "
                     f"| {row['median']:.2f} | {row['mean']:.2f} | {int(row['count'])} | {note or ''} |\n")

    return entry_dir


def archive_generic_run(archive_root, experiment_label, entry_label, title, note, copy_paths,
                         changelog_header, row_template, changelog_rows):
    '''Reusable version of archive_run() for analysis scripts other than this module's own HV
    summary/desc format (analyze_pf_and_convergence.py, analyze_quality_by_size.py, ...). Copies
    copy_paths into a timestamped, never-overwritten entry under archive_root/experiment_label/,
    writes a summary.md, and appends one row per dict in changelog_rows to
    archive_root/experiment_label/CHANGELOG.md (created with changelog_header if new). Each row
    dict is merged with {timestamp, rel_dir, commit_label, note} and formatted into row_template.'''
    commit, subject, diffstat = get_git_info()
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H%M%S')
    commit_label = f"{commit}-dirty" if diffstat else (commit or "no-git")

    archive_root = os.path.join(archive_root, experiment_label)
    entry_dir = os.path.join(archive_root, f"{timestamp}_{commit_label}_{entry_label}")
    os.makedirs(entry_dir, exist_ok = True)
    for fpath in copy_paths:
        if fpath and os.path.exists(fpath):
            shutil.copy(fpath, entry_dir)

    with open(os.path.join(entry_dir, 'summary.md'), 'w') as f:
        f.write(f"# {title}: {entry_label}\n\n")
        f.write(f"- Experiment: `{experiment_label}`\n")
        f.write(f"- Datum: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (" -- **uncommitted changes!**\n" if diffstat else "\n"))
        if note:
            f.write(f"\n## Was hat sich seit dem letzten Eintrag geaendert\n\n{note}\n")
        if diffstat:
            f.write(f"\n## Uncommitted Aenderungen zum Zeitpunkt des Laufs\n\n```\n{diffstat}\n```\n")

    index_path = os.path.join(archive_root, 'CHANGELOG.md')
    is_new = not os.path.exists(index_path)
    with open(index_path, 'a') as f:
        if is_new:
            f.write(f"# {title} -- Verlauf\n\n")
            f.write(changelog_header)
        rel_dir = os.path.relpath(entry_dir, archive_root)
        for row in changelog_rows:
            fields = {'timestamp': timestamp, 'rel_dir': rel_dir, 'commit_label': commit_label,
                      'note': note or '', **row}
            f.write(row_template.format(**fields) + "\n")

    return entry_dir


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path',
                         help = 'z.B. results/approximations/ex05 (enthaelt design.csv und experiments/<jobid>/pf.csv)')
    parser.add_argument('--instance', default = None,
                         help = 'Instanzpfad wie in design.csv; Default: erste in design.csv gefundene Instanz')
    parser.add_argument('--all-instances', action = 'store_true',
                         help = 'Statt einer einzelnen Instanz alle Instanzen aus design.csv gepoolt auswerten '
                                '(Rang- und Relativ-HV-basiert, da rohe HV-Werte zwischen Instanzen nicht '
                                'vergleichbar sind); --instance wird dabei ignoriert.')
    parser.add_argument('--facet-class-instances', default = None,
                         help = 'Komma-getrennte Liste von Instanzpfaden (eine je Klasse) -- statt Einzel-'
                                'Instanz oder --all-instances wird EIN Bild je Diagrammtyp (Violin/ECDF) mit '
                                'einem Panel je Instanzklasse erzeugt; rohe HV bleibt dabei nie ueber '
                                'Instanzen gemischt. Boxplot/Progress bewusst weggelassen, da '
                                'analyze_quality_by_size.py (--facet-by class) bzw. analyze_indicator_'
                                'progress.py (--facet-by-class) dieselben Werte bereits liefern. '
                                'Ueberschreibt --instance/--all-instances.')
    parser.add_argument('--algo-col', default = 'crossover',
                         help = "Spaltenname(n) der Algorithmus-/Operator-Bezeichnung, komma-getrennt fuer "
                                "kombinierte Algorithmen (default: 'crossover'). Bei ex06 z.B. 'crossover,mutator', "
                                "weil dort jede Crossover+Mutator-KOMBINATION ein eigener Algorithmus ist "
                                "(nicht crossover und mutator getrennt vergleichen -- das wuerde ueber die "
                                "jeweils andere Dimension mitteln).")
    parser.add_argument('--out', default = 'figures/hv_analysis',
                         help = 'Ausgabeverzeichnis fuer den "aktueller Stand"-Blick auf Plots + HV-CSV; wird bei jedem Lauf ueberschrieben')
    parser.add_argument('--archive-dir', default = 'analysis_log',
                         help = 'Basisverzeichnis des Verlaufs-Archivs (default: analysis_log). Jeder Lauf legt hier '
                                'zusaetzlich eine mit Zeitstempel+Commit gelabelte, nie ueberschriebene Kopie ab.')
    parser.add_argument('--no-archive', action = 'store_true',
                         help = 'Verlaufs-Archivierung fuer diesen Lauf abschalten (z.B. fuer Testlaeufe)')
    parser.add_argument('--note', default = None,
                         help = 'Kurze Notiz, was sich seit dem letzten Archiv-Eintrag geaendert hat; landet in summary.md und CHANGELOG.md')
    args = parser.parse_args()

    design_path = os.path.join(args.registry_path, 'design.csv')
    rows = read_params(design_path)
    # basename of registry_path (e.g. "ex05", "ex06", "ex07") -- keeps output from
    # different runners/experiments in separate subfolders instead of mixed together
    experiment_label = os.path.basename(os.path.normpath(args.registry_path))
    algo_cols = args.algo_col.split(',')

    if args.facet_class_instances:
        instances = args.facet_class_instances.split(',')
        by_class_df = {}
        for instance in instances:
            cls = infer_instance_class(instance)
            by_algo = collect_fronts(rows, args.registry_path, instance, algo_cols)
            if not by_algo:
                print(f"WARNUNG: keine Jobs fuer Instanz '{instance}' gefunden -- uebersprungen.")
                continue
            by_algo_traj = collect_trajectories(rows, args.registry_path, instance, algo_cols)
            ref = determine_reference_point(by_algo, by_algo_traj)
            df = compute_hv_table(by_algo, ref)
            instance_label = os.path.basename(instance)
            by_class_df[cls] = (df, instance_label)
            print(f'Klasse {cls}: Instanz = {instance} (Referenzpunkt: {ref})')

        out_dir = os.path.join(args.out, experiment_label, '_by_class')
        paths = plot_distributions_by_class(by_class_df, out_dir)

        print(f'\nPlots gespeichert in: {out_dir}')
        for p in paths.values():
            print(f'  {p}')
        return

    if args.all_instances:
        n_instances = len(set(row['instance'] for row in rows))
        instance_label = '_all_instances'
        df_cross = compute_cross_instance_table(rows, args.registry_path, algo_cols)
        print(f"Instanzen gepoolt: {n_instances}")
        print(f"Algorithmen ({args.algo_col}): {sorted(df_cross['algorithm'].unique())}")

        out_dir = os.path.join(args.out, experiment_label, instance_label)
        plot_cross_instance(df_cross, out_dir, n_instances)
        report_text, desc = run_statistics(
            df_cross, block_cols = ['instance', 'repl'], value_col = 'hv',
            block_label = f"(Instanz,Repl)-Bloecke ueber {n_instances} Instanzen", higher_is_better = True)
        print(f"\n{report_text}")

        csv_out = os.path.join(out_dir, 'hv_cross_instance.csv')
        df_cross.sort_values(['algorithm', 'instance', 'repl']).to_csv(csv_out, index = False)
        print(f"\nPlots gespeichert in: {out_dir}")
        print(f"Rohwerte (hv, rank, relative_pct je Instanz/Repl) gespeichert in: {csv_out}")

        if not args.no_archive:
            entry_dir = archive_run(args.archive_dir, experiment_label, instance_label, args.algo_col, report_text,
                                     desc, args.note, out_dir, csv_out,
                                     extra_plot_names = ['hv_boxplot.png', 'hv_violinplot.png', 'hv_ecdf.png'])
            print(f"Archiviert in: {entry_dir}")
            print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, experiment_label, 'CHANGELOG.md')}")
        return

    instance = args.instance or rows[0]['instance']
    instance_label = os.path.basename(instance)

    by_algo = collect_fronts(rows, args.registry_path, instance, algo_cols)
    if not by_algo:
        raise SystemExit(f"Keine Jobs fuer Instanz '{instance}' in {design_path} gefunden.")
    by_algo_traj = collect_trajectories(rows, args.registry_path, instance, algo_cols)

    ref = determine_reference_point(by_algo, by_algo_traj)
    print(f"Instanz: {instance}")
    print(f"Algorithmen ({args.algo_col}): {sorted(by_algo.keys())}")
    print(f"Referenzpunkt fuer HV (max(c1), max(c2)) * 1.05 ueber alle Algorithmen/Wiederholungen/Generationen: {ref}")

    df = compute_hv_table(by_algo, ref)
    df_progress = compute_hv_progress_table(by_algo_traj, ref)

    out_dir = os.path.join(args.out, experiment_label, instance_label)
    plot_distributions(df, out_dir, instance_label)
    plot_progress(df_progress, out_dir, instance_label)
    report_text, desc = run_statistics(df)
    print(f"\n{report_text}")

    csv_out = os.path.join(out_dir, 'hv_values.csv')
    df.sort_values(['algorithm', 'repl']).to_csv(csv_out, index = False)
    progress_csv_out = os.path.join(out_dir, 'hv_progress.csv')
    df_progress.sort_values(['algorithm', 'repl', 'gen']).to_csv(progress_csv_out, index = False)
    print(f"\nPlots gespeichert in: {out_dir}")
    print(f"HV-Rohwerte gespeichert in: {csv_out}")
    print(f"HV-Verlaufsdaten gespeichert in: {progress_csv_out}")

    if not args.no_archive:
        entry_dir = archive_run(args.archive_dir, experiment_label, instance_label, args.algo_col, report_text, desc,
                                 args.note, out_dir, csv_out, progress_csv_out)
        print(f"Archiviert in: {entry_dir}")
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, experiment_label, 'CHANGELOG.md')}")


if __name__ == '__main__':
    main()
