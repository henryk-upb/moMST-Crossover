#!/usr/bin/env python
'''


  Zwei Modi, ueber --pool-instances gesteuert:

  * Default (representative instance je Groesse): fuer jede Groesse wird EINE
    Instanz herangezogen (per --instance mehrfach explizit waehlbar, sonst
    die erste in design.csv gefundene je Groesse) und die rohe Hypervolume
    ueber die Wiederholungen geboxplottet -- direktes Analogon zu Fig. 7, wo
    laut Bildunterschrift ebenfalls "exemplary instances" je Groesse gezeigt
    werden. Rohe HV ist innerhalb einer Instanz zwischen Algorithmen
    vergleichbar (gleicher Referenzpunkt).

  * --pool-instances: pro Groesse werden ALLE gefundenen Instanzen jeder
    Klasse gepoolt, aber nicht als rohe HV (deren Skala instanzabhaengig ist,
    siehe analyze_hypervolume.py), sondern als relative_pct (100 * hv /
    beste-hv-im-selben-(Instanz,Repl)-Block) -- robustere, weil auf mehr
    Instanzen gestuetzte Variante desselben Vergleichs.

Algorithmus-Spalte(n) pro Registry: entweder global per --algo-col, oder pro
Registry ueberschreibbar per "PFAD:SPALTE(N)" (z. B. weil ex05 die Spalte
"crossover" nutzt, ex07 aber "operator" -- beide meinen fuer UNION/INTER/XOR
dasselbe und sind so direkt kombinierbar).

Nutzung:
  # eine Registry, ein Facet (waechst automatisch mit, sobald weitere
  # Instanzgroessen fuer denselben Runner vorliegen, z. B. ex07 auch fuer n=100)
  python analysis/analyze_quality_by_size.py results/approximations/ex07 --algo-col operator

  # zwei Registries mit unterschiedlichen Spaltennamen, gepoolt ueber alle
  # Instanzen je Groesse
  python analysis/analyze_quality_by_size.py results/approximations/ex07 \
      results/approximations/ex05:crossover \
      --algo-col operator --pool-instances --log-scale
'''

import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from analyze_hypervolume import (
    read_params, collect_fronts, collect_trajectories, determine_reference_point,
    compute_hv_table, get_git_info,
)

INSTANCE_NAME_RE = re.compile(r'^CLASS(\d+)_(\d+)_')


def infer_instance_size(instance_path):
    '''Instanzgroesse n, abgeleitet aus der grapherator-Namenskonvention
    CLASS<k>_<n>_<m>_... (z. B. CLASS1_50_1225_0_2_UNG_CEG_RWG-RWG_1.graph
    -> 50). Funktioniert unabhaengig davon, ob die Datei unter instances/<n>/
    liegt oder lose im instances/-Wurzelverzeichnis.'''
    m = INSTANCE_NAME_RE.match(os.path.basename(instance_path))
    if not m:
        raise ValueError(
            f"Instanzgroesse konnte nicht aus '{instance_path}' abgeleitet werden "
            "(erwartetes Namensschema: CLASS<k>_<n>_...).")
    return int(m.group(2))


def infer_instance_class(instance_path):
    '''Instanzklasse "C<k>", abgeleitet aus derselben Namenskonvention (z. B.
    CLASS1_50_..._1.graph -> "C1"). Wird u. a. von analyze_mean_rankings.py
    genutzt, um Instanzen fuer das class-weise Ranking zu gruppieren.'''
    m = INSTANCE_NAME_RE.match(os.path.basename(instance_path))
    if not m:
        raise ValueError(
            f"Instanzklasse konnte nicht aus '{instance_path}' abgeleitet werden "
            "(erwartetes Namensschema: CLASS<k>_<n>_...).")
    return f'C{m.group(1)}'


def parse_registry_arg(arg, default_algo_cols):
    '''"PFAD" oder "PFAD:SPALTE1,SPALTE2" -> (pfad, algo_cols).'''
    if ':' in arg:
        path, cols = arg.split(':', 1)
        return path, cols.split(',')
    return arg, default_algo_cols


def collect_rows_by_size(registries):
    '''[(registry_path, algo_cols), ...] -> {size: [(registry_path, algo_cols, row), ...]}'''
    by_size = {}
    for registry_path, algo_cols in registries:
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        for row in rows:
            size = infer_instance_size(row['instance'])
            by_size.setdefault(size, []).append((registry_path, algo_cols, row))
    return by_size


def pick_representative_instances(by_size, explicit_instances):
    '''size -> (registry_path, algo_cols, instance). explicit_instances is an optional
    list of instance paths (as they appear in design.csv); if given, the matching
    size/registry/instance combination is used instead of the first one found.'''
    chosen = {}
    for size, entries in by_size.items():
        match = None
        if explicit_instances:
            for registry_path, algo_cols, row in entries:
                if row['instance'] in explicit_instances:
                    match = (registry_path, algo_cols, row['instance'])
                    break
        if match is None:
            registry_path, algo_cols, row = entries[0]
            match = (registry_path, algo_cols, row['instance'])
        chosen[size] = match
    return chosen


def compute_hv_by_size_representative(by_size, explicit_instances):
    '''Fig.-7-Modus: eine Instanz je Groesse, rohe HV je Algorithmus/Wiederholung.
    Returns long-format DataFrame mit Spalten size, algorithm, repl, hv, instance.'''
    representatives = pick_representative_instances(by_size, explicit_instances)
    frames = []
    for size, (registry_path, algo_cols, instance) in sorted(representatives.items()):
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        by_algo = collect_fronts(rows, registry_path, instance, algo_cols)
        if not by_algo:
            continue
        by_algo_traj = collect_trajectories(rows, registry_path, instance, algo_cols)
        ref = determine_reference_point(by_algo, by_algo_traj)
        df = compute_hv_table(by_algo, ref)
        df['size'] = size
        df['instance'] = instance
        frames.append(df)
    if not frames:
        raise SystemExit('Keine Daten gefunden -- Registries/--instance pruefen.')
    return pd.concat(frames, ignore_index = True)


def compute_hv_by_size_pooled(by_size):
    '''--pool-instances-Modus: alle Instanzen je Groesse gepoolt, als relative_pct
    (100 * hv / beste hv im selben (Instanz,Repl)-Block), damit die instanzabhaengige
    HV-Skala den Groessenvergleich nicht verzerrt. Returns long-format DataFrame mit
    Spalten size, algorithm, repl, instance, hv, relative_pct.'''
    frames = []
    for size, entries in sorted(by_size.items()):
        by_registry = {}
        for registry_path, algo_cols, row in entries:
            by_registry.setdefault((registry_path, tuple(algo_cols)), []).append(row)

        for (registry_path, algo_cols), rows in by_registry.items():
            algo_cols = list(algo_cols)
            instances = sorted(set(row['instance'] for row in rows))
            for instance in instances:
                by_algo = collect_fronts(rows, registry_path, instance, algo_cols)
                if not by_algo:
                    continue
                by_algo_traj = collect_trajectories(rows, registry_path, instance, algo_cols)
                ref = determine_reference_point(by_algo, by_algo_traj)
                df = compute_hv_table(by_algo, ref)
                df['instance'] = instance
                frames.append(df)

    if not frames:
        raise SystemExit('Keine Daten gefunden -- Registries pruefen.')
    df_all = pd.concat(frames, ignore_index = True)
    # relative_pct pro (instance, repl)-Block, dann Groesse wieder anfuegen
    block_max = df_all.groupby(['instance', 'repl'])['hv'].transform('max')
    df_all['relative_pct'] = 100.0 * df_all['hv'] / block_max
    size_lookup = {row['instance']: size for size, entries in by_size.items()
                   for _, _, row in entries}
    df_all['size'] = df_all['instance'].map(size_lookup)
    return df_all


def plot_quality_by_size(df, value_col, out_dir, log_scale, title_suffix):
    '''Facet-Grid: ein Panel je Instanzgroesse (aufsteigend sortiert), darin ein
    Boxplot je Algorithmus -- direktes Analogon zu Fig. 7 (Bossek & Grimme, 2017).'''
    os.makedirs(out_dir, exist_ok = True)
    sizes = sorted(df['size'].unique())
    algos = sorted(df['algorithm'].unique())
    n_algos = len(algos)
    rotate = n_algos > 4
    label_fontsize = 8 if n_algos <= 8 else 7

    ncols = min(len(sizes), 3)
    nrows = math.ceil(len(sizes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize = (4.2 * ncols, 4.5 * nrows), squeeze = False)
    axes_flat = axes.flatten()

    for ax, size in zip(axes_flat, sizes):
        sub = df[df['size'] == size]
        data = [sub.loc[sub['algorithm'] == a, value_col].values for a in algos]
        ax.boxplot(data, tick_labels = algos, showmeans = True)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(f'n = {size}')
        label_kwargs = dict(rotation = 45, ha = 'right') if rotate else {}
        plt.setp(ax.get_xticklabels(), fontsize = label_fontsize, **label_kwargs)
    for ax in axes_flat[len(sizes):]:
        ax.axis('off')

    if value_col == 'hv':
        ylabel = 'Dominierte Hypervolume' + (' (log-skaliert)' if log_scale else '')
    else:
        ylabel = 'Relative Hypervolume [%] (100 = beste im Block)'
    fig.suptitle(f'Qualitaet ueber Instanzgroessen{title_suffix}')
    fig.text(0.02, 0.5, ylabel, va = 'center', rotation = 'vertical')
    fig.tight_layout(rect = (0.03, 0, 1, 0.96))

    out_path = os.path.join(out_dir, 'quality_by_size_boxplot.png')
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    return out_path


def write_run_info(out_dir, note, registries, mode_desc):
    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write('# Qualitaet ueber Instanzgroessen -- Lauf-Info\n\n')
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (' -- **uncommitted changes!**\n' if diffstat else '\n'))
        f.write(f'- Modus: {mode_desc}\n')
        cols_desc = ', '.join(f"`{p}` (Spalte(n): {','.join(c)})" for p, c in registries)
        f.write(f'- Registries: {cols_desc}\n')
        if note:
            f.write(f'\n## Notiz\n\n{note}\n')


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registries', nargs = '+',
                         help = 'Ein oder mehrere Registry-Pfade, optional mit ":SPALTE(N)" '
                                'ueberschrieben, z. B. results/approximations/ex05:crossover')
    parser.add_argument('--algo-col', default = 'operator',
                         help = "Default-Algorithmus-Spalte(n) (komma-getrennt) fuer Registries "
                                "ohne eigene ':SPALTE'-Angabe (default: 'operator')")
    parser.add_argument('--pool-instances', action = 'store_true',
                         help = 'Statt einer repraesentativen Instanz je Groesse alle Instanzen '
                                'je Groesse poolen (skalenunabhaengig via relative_pct statt roher HV)')
    parser.add_argument('--instance', action = 'append', default = None,
                         help = 'Nur im Default-Modus: explizite repraesentative Instanz (mehrfach '
                                'nutzbar, eine pro gewuenschter Groesse); sonst erste gefundene je Groesse')
    parser.add_argument('--log-scale', action = 'store_true',
                         help = 'y-Achse log-skalieren (wie in Fig. 7 des 2017er-Papers)')
    parser.add_argument('--out', default = 'figures/quality_by_size',
                         help = 'Ausgabeverzeichnis (default: figures/quality_by_size)')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out fuer diesen Vergleich (default: _all)')
    parser.add_argument('--note', default = None, help = 'Kurze Notiz, landet in run_info.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    registries = [parse_registry_arg(r, default_algo_cols) for r in args.registries]

    by_size = collect_rows_by_size(registries)
    print(f'Gefundene Instanzgroessen: {sorted(by_size.keys())}')

    if args.pool_instances:
        df = compute_hv_by_size_pooled(by_size)
        value_col = 'relative_pct'
        mode_desc = 'gepoolt ueber alle Instanzen je Groesse (relative_pct)'
        title_suffix = ' (gepoolt, relative HV)'
    else:
        df = compute_hv_by_size_representative(by_size, args.instance)
        value_col = 'hv'
        mode_desc = 'eine repraesentative Instanz je Groesse (rohe HV)'
        title_suffix = ''
        for size in sorted(df['size'].unique()):
            inst = df.loc[df['size'] == size, 'instance'].iloc[0]
            print(f'  n = {size}: Instanz = {inst}')

    out_dir = os.path.join(args.out, args.label)
    out_path = plot_quality_by_size(df, value_col, out_dir, args.log_scale, title_suffix)
    csv_out = os.path.join(out_dir, 'quality_by_size.csv')
    df.sort_values(['size', 'algorithm', 'repl']).to_csv(csv_out, index = False)
    write_run_info(out_dir, args.note, registries, mode_desc)

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Rohwerte gespeichert in: {csv_out}')


if __name__ == '__main__':
    main()
