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

  Alles bezieht sich auf eine einzelne Instanz; die Erweiterung auf mehrere
  Instanzen ist die Wiederholung dieser Prozedur je Instanz (z.B. in einer
  Schleife ueber --instance).

Erwartet das Ausgabelayout von *_runner_expyrimenter.py (expyrimenter's
Registry): <registry_path>/design.csv (Spalten jobid, instance, <algo-col>,
repl, ...) und Job-Ergebnisse unter <registry_path>/experiments/<jobid>/pf.csv.

Jeder Lauf wird automatisch dauerhaft archiviert (siehe --archive-dir,
default: analysis_log/), damit aufeinanderfolgende Laeufe sich nicht
gegenseitig ueberschreiben und der Fortschritt ueber die Zeit sichtbar
bleibt. --out bleibt zusaetzlich als schnelle "aktueller Stand"-Ansicht
bestehen und WIRD bei jedem Lauf ueberschrieben.

Nutzung:
  python analyze_hypervolume.py results/approximations/ex05 \
      --instance instances/50/CLASS1_50_1225_0_2_UNG_CEG_RWG-RWG_1.graph \
      --note "XOR-CX Bugfix: Algorithmus 5 statt fixierter Schnittmenge"
'''

import argparse
import csv
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


def read_params(path):
    with open(path, newline = '') as f:
        return list(csv.DictReader(f))


def read_pf(path):
    with open(path, newline = '') as f:
        return [(float(row['c1']), float(row['c2'])) for row in csv.DictReader(f)]


def hypervolume_2d(points, ref):
    '''Dominated hypervolume of a minimization front w.r.t. reference point ref, via pymoo.'''
    return HV(ref_point = np.array(ref))(np.array(points))


def collect_fronts(params_rows, registry_path, instance, algo_col):
    '''algorithm -> [(repl, points), ...] for all jobs of the given instance.'''
    by_algo = {}
    for row in params_rows:
        if row['instance'] != instance:
            continue
        algo = row[algo_col]
        pf_path = os.path.join(registry_path, 'experiments', row['jobid'], 'pf.csv')
        points = read_pf(pf_path)
        by_algo.setdefault(algo, []).append((row['repl'], points))
    return by_algo


def determine_reference_point(by_algo, margin = 0.05):
    '''Shared nadir-style reference point across all algorithms/repls of the instance.'''
    all_points = [p for repls in by_algo.values() for _, points in repls for p in points]
    xs, ys = zip(*all_points)
    return (max(xs) * (1 + margin), max(ys) * (1 + margin))


def compute_hv_table(by_algo, ref):
    records = [
        {'algorithm': algo, 'repl': int(repl), 'hv': hypervolume_2d(points, ref)}
        for algo, repls in by_algo.items()
        for repl, points in repls
    ]
    return pd.DataFrame.from_records(records)


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

    fig, ax = plt.subplots(figsize = (5, 4))
    ax.boxplot(data, tick_labels = algos, showmeans = True)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV-Verteilung (Boxplot)\n{instance_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_boxplot.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (5, 4))
    ax.violinplot(data, showmeans = True, showmedians = True)
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos)
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'HV-Verteilung (Violinplot)\n{instance_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_violinplot.png'), dpi = 150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(algos), figsize = (4 * len(algos), 4), sharey = True)
    axes = [axes] if len(algos) == 1 else axes
    for ax, algo in zip(axes, algos):
        ax.hist(df.loc[df['algorithm'] == algo, 'hv'], bins = 10, color = 'tab:blue', alpha = 0.8)
        ax.set_title(algo)
        ax.set_xlabel('Hypervolume')
    axes[0].set_ylabel('Anzahl Wiederholungen')
    fig.suptitle(f'HV-Verteilung (Histogramme)\n{instance_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_histograms.png'), dpi = 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (5, 4))
    for algo in algos:
        vals = np.sort(df.loc[df['algorithm'] == algo, 'hv'].values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where = 'post', label = algo)
    ax.set_xlabel('Hypervolume')
    ax.set_ylabel('F(x)')
    ax.set_title(f'Empirische Verteilungsfunktion (ECDF)\n{instance_label}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'hv_ecdf.png'), dpi = 150)
    plt.close(fig)


def run_statistics(df):
    '''Runs the descriptive + inferential statistics and returns (report_text, desc_table).'''
    algos = sorted(df['algorithm'].unique())
    pivot = df.pivot(index = 'repl', columns = 'algorithm', values = 'hv').dropna()
    lines = []

    lines.append(f"=== Deskriptive Statistik (n = {len(pivot)} gepaarte Wiederholungen) ===")
    desc = pivot.describe().T[['count', 'mean', 'std', '50%', 'min', 'max']].rename(columns = {'50%': 'median'})
    lines.append(desc.to_string())

    if len(algos) >= 3:
        stat, p = stats.friedmanchisquare(*[pivot[a] for a in algos])
        lines.append(f"\nFriedman-Test (gepaart ueber repl, {len(algos)} Gruppen): chi2 = {stat:.4f}, p = {p:.4g}")
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
            direction = a if pivot[a].median() > pivot[b].median() else b
            lines.append(f"{a} vs. {b}: p_raw = {p_raw:.4g}, p_holm = {p_adj:.4g}"
                          f"  (hoehere Median-HV: {direction})")

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


def archive_run(archive_root, instance_label, algo_col, report_text, desc, note, plot_dir, csv_path):
    '''Copies plots + raw HV values into a timestamped entry and appends a row per algorithm to CHANGELOG.md.'''
    commit, subject, diffstat = get_git_info()
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H%M%S')
    commit_label = f"{commit}-dirty" if diffstat else (commit or "no-git")

    entry_dir = os.path.join(archive_root, f"{timestamp}_{commit_label}_{instance_label}")
    os.makedirs(entry_dir, exist_ok = True)
    for fname in ['hv_boxplot.png', 'hv_violinplot.png', 'hv_histograms.png', 'hv_ecdf.png']:
        shutil.copy(os.path.join(plot_dir, fname), entry_dir)
    shutil.copy(csv_path, entry_dir)

    with open(os.path.join(entry_dir, 'summary.md'), 'w') as f:
        f.write(f"# HV-Analyse: {instance_label}\n\n")
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


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path',
                         help = 'z.B. results/approximations/ex05 (enthaelt design.csv und experiments/<jobid>/pf.csv)')
    parser.add_argument('--instance', default = None,
                         help = 'Instanzpfad wie in design.csv; Default: erste in design.csv gefundene Instanz')
    parser.add_argument('--algo-col', default = 'crossover',
                         help = "Spaltenname der Algorithmus-/Operator-Bezeichnung (default: 'crossover')")
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
    instance = args.instance or rows[0]['instance']
    instance_label = os.path.basename(instance)

    by_algo = collect_fronts(rows, args.registry_path, instance, args.algo_col)
    if not by_algo:
        raise SystemExit(f"Keine Jobs fuer Instanz '{instance}' in {design_path} gefunden.")

    ref = determine_reference_point(by_algo)
    print(f"Instanz: {instance}")
    print(f"Algorithmen ({args.algo_col}): {sorted(by_algo.keys())}")
    print(f"Referenzpunkt fuer HV (max(c1), max(c2)) * 1.05 ueber alle Algorithmen/Wiederholungen: {ref}")

    df = compute_hv_table(by_algo, ref)

    out_dir = os.path.join(args.out, instance_label)
    plot_distributions(df, out_dir, instance_label)
    report_text, desc = run_statistics(df)
    print(f"\n{report_text}")

    csv_out = os.path.join(out_dir, 'hv_values.csv')
    df.sort_values(['algorithm', 'repl']).to_csv(csv_out, index = False)
    print(f"\nPlots gespeichert in: {out_dir}")
    print(f"HV-Rohwerte gespeichert in: {csv_out}")

    if not args.no_archive:
        entry_dir = archive_run(args.archive_dir, instance_label, args.algo_col, report_text, desc,
                                 args.note, out_dir, csv_out)
        print(f"Archiviert in: {entry_dir}")
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, 'CHANGELOG.md')}")


if __name__ == '__main__':
    main()
