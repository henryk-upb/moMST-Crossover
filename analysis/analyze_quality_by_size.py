#!/usr/bin/env python
'''
Facet-Grid-Vergleich der Loesungsqualitaet (rohe oder relative Hypervolume),
gefacettet ueber --facet-by: entweder Instanzgroesse n (Fig. 7, Bossek &
Grimme 2017) oder Instanzklasse C1-C4 (z. B. wenn wie bei ex07 alle Instanzen
dieselbe Groesse haben, aber sich in der Klasse -- Kanten-Gewichtsverteilung
etc. -- unterscheiden).

  Zwei Modi, ueber --pool-instances gesteuert:

  * Default (representative instance je Facet-Wert): fuer jeden Facet-Wert
    (Groesse oder Klasse) wird EINE Instanz herangezogen (per --instance
    mehrfach explizit waehlbar, sonst die erste in design.csv gefundene) und
    die rohe Hypervolume ueber die Wiederholungen geboxplottet -- direktes
    Analogon zu Fig. 7, wo laut Bildunterschrift ebenfalls "exemplary
    instances" gezeigt werden. Rohe HV ist innerhalb einer Instanz zwischen
    Algorithmen vergleichbar (gleicher Referenzpunkt), wird hier aber NIE
    ueber mehrere Instanzen hinweg gemischt -- jedes Panel zeigt exakt eine
    Instanz.

  * --pool-instances: pro Facet-Wert werden ALLE gefundenen Instanzen
    gepoolt, aber nicht als rohe HV (deren Skala instanzabhaengig ist, siehe
    analyze_hypervolume.py), sondern als relative_pct (100 * hv /
    beste-hv-im-selben-(Instanz,Repl)-Block) -- robustere, weil auf mehr
    Instanzen gestuetzte Variante desselben Vergleichs.

Algorithmus-Spalte(n) pro Registry: entweder global per --algo-col, oder pro
Registry ueberschreibbar per "PFAD:SPALTE(N)" (z. B. weil ex05 die Spalte
"crossover" nutzt, ex07 aber "operator" -- beide meinen fuer UNION/INTER/XOR
dasselbe und sind so direkt kombinierbar).

Nutzung:
  # eine Registry, gefacettet nach Groesse (default; waechst automatisch mit,
  # sobald weitere Instanzgroessen fuer denselben Runner vorliegen)
  python analysis/analyze_quality_by_size.py results/approximations/ex07 --algo-col operator

  # gefacettet nach Instanzklasse (C1-C4) statt Groesse -- sinnvoll wenn wie
  # bei ex07 alle Instanzen dieselbe Groesse haben; rohe HV bleibt erhalten,
  # aber jedes Panel vergleicht nur innerhalb EINER Instanz
  python analysis/analyze_quality_by_size.py results/approximations/ex07 \
      --algo-col operator --facet-by class

  # zwei Registries mit unterschiedlichen Spaltennamen, gepoolt ueber alle
  # Instanzen je Facet-Wert
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


INFER_FACET = {'size': infer_instance_size, 'class': infer_instance_class}
FACET_LABELS = {'size': 'Instanzgroessen', 'class': 'Instanzklassen'}


def collect_rows_by_facet(registries, facet_by):
    '''[(registry_path, algo_cols), ...] -> {facet_value: [(registry_path, algo_cols, row), ...]}
    facet_by ist 'size' (Instanzgroesse n) oder 'class' (Instanzklasse C1-C4).'''
    infer = INFER_FACET[facet_by]
    by_facet = {}
    for registry_path, algo_cols in registries:
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        for row in rows:
            key = infer(row['instance'])
            by_facet.setdefault(key, []).append((registry_path, algo_cols, row))
    return by_facet


def pick_representative_instances(by_facet, explicit_instances):
    '''facet_value -> (registry_path, algo_cols, instance). explicit_instances is an optional
    list of instance paths (as they appear in design.csv); if given, the matching
    facet/registry/instance combination is used instead of the first one found.'''
    chosen = {}
    for key, entries in by_facet.items():
        match = None
        if explicit_instances:
            for registry_path, algo_cols, row in entries:
                if row['instance'] in explicit_instances:
                    match = (registry_path, algo_cols, row['instance'])
                    break
        if match is None:
            registry_path, algo_cols, row = entries[0]
            match = (registry_path, algo_cols, row['instance'])
        chosen[key] = match
    return chosen


def compute_hv_by_facet_representative(by_facet, explicit_instances, facet_by):
    '''Fig.-7-Modus: eine Instanz je Facet-Wert, rohe HV je Algorithmus/Wiederholung -- nie ueber
    mehrere Instanzen gemischt. Returns long-format DataFrame mit Spalten facet_by, algorithm,
    repl, hv, instance.'''
    representatives = pick_representative_instances(by_facet, explicit_instances)
    frames = []
    for key, (registry_path, algo_cols, instance) in sorted(representatives.items()):
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        by_algo = collect_fronts(rows, registry_path, instance, algo_cols)
        if not by_algo:
            continue
        by_algo_traj = collect_trajectories(rows, registry_path, instance, algo_cols)
        ref = determine_reference_point(by_algo, by_algo_traj)
        df = compute_hv_table(by_algo, ref)
        df[facet_by] = key
        df['instance'] = instance
        frames.append(df)
    if not frames:
        raise SystemExit('Keine Daten gefunden -- Registries/--instance pruefen.')
    return pd.concat(frames, ignore_index = True)


def compute_hv_by_facet_pooled(by_facet, facet_by):
    '''--pool-instances-Modus: alle Instanzen je Facet-Wert gepoolt, als relative_pct
    (100 * hv / beste hv im selben (Instanz,Repl)-Block), damit die instanzabhaengige
    HV-Skala den Vergleich nicht verzerrt. Returns long-format DataFrame mit Spalten
    facet_by, algorithm, repl, instance, hv, relative_pct.'''
    frames = []
    for key, entries in sorted(by_facet.items()):
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
    # relative_pct pro (instance, repl)-Block, dann Facet-Wert wieder anfuegen
    block_max = df_all.groupby(['instance', 'repl'])['hv'].transform('max')
    df_all['relative_pct'] = 100.0 * df_all['hv'] / block_max
    facet_lookup = {row['instance']: key for key, entries in by_facet.items() for _, _, row in entries}
    df_all[facet_by] = df_all['instance'].map(facet_lookup)
    return df_all


def plot_quality_by_facet(df, value_col, out_dir, log_scale, title_suffix, facet_by):
    '''Facet-Grid: ein Panel je Facet-Wert (Groesse oder Klasse, aufsteigend sortiert), darin ein
    Boxplot je Algorithmus -- direktes Analogon zu Fig. 7 (Bossek & Grimme, 2017), auf Instanz-
    klassen uebertragen wenn facet_by = 'class'.'''
    os.makedirs(out_dir, exist_ok = True)
    keys = sorted(df[facet_by].unique())
    algos = sorted(df['algorithm'].unique())
    n_algos = len(algos)
    rotate = n_algos > 4
    label_fontsize = 8 if n_algos <= 8 else 7
    panel_title = (lambda v: f'n = {v}') if facet_by == 'size' else (lambda v: f'Klasse {v}')

    ncols = min(len(keys), 3)
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize = (4.2 * ncols, 4.5 * nrows), squeeze = False)
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, keys):
        sub = df[df[facet_by] == key]
        data = [sub.loc[sub['algorithm'] == a, value_col].values for a in algos]
        ax.boxplot(data, tick_labels = algos, showmeans = True)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(panel_title(key))
        label_kwargs = dict(rotation = 45, ha = 'right') if rotate else {}
        plt.setp(ax.get_xticklabels(), fontsize = label_fontsize, **label_kwargs)
    for ax in axes_flat[len(keys):]:
        ax.axis('off')

    if value_col == 'hv':
        ylabel = 'Dominierte Hypervolume' + (' (log-skaliert)' if log_scale else '')
    else:
        ylabel = 'Relative Hypervolume [%] (100 = beste im Block)'
    fig.suptitle(f'Qualitaet ueber {FACET_LABELS[facet_by]}{title_suffix}')
    fig.text(0.02, 0.5, ylabel, va = 'center', rotation = 'vertical')
    fig.tight_layout(rect = (0.03, 0, 1, 0.96))

    out_path = os.path.join(out_dir, f'quality_by_{facet_by}_boxplot.png')
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    return out_path


def write_run_info(out_dir, note, registries, mode_desc, facet_by):
    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write(f'# Qualitaet ueber {FACET_LABELS[facet_by]} -- Lauf-Info\n\n')
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
    parser.add_argument('--facet-by', choices = ['size', 'class'], default = 'size',
                         help = "Wonach gefacettet wird: 'size' = Instanzgroesse n (Fig. 7, Bossek "
                                "& Grimme 2017), 'class' = Instanzklasse C1-C4 -- z.B. sinnvoll wenn "
                                "wie bei ex07 alle Instanzen dieselbe Groesse haben (default: size)")
    parser.add_argument('--pool-instances', action = 'store_true',
                         help = 'Statt einer repraesentativen Instanz je Facet-Wert alle Instanzen '
                                'je Facet-Wert poolen (skalenunabhaengig via relative_pct statt roher HV)')
    parser.add_argument('--instance', action = 'append', default = None,
                         help = 'Nur im Default-Modus: explizite repraesentative Instanz (mehrfach '
                                'nutzbar, eine pro gewuenschtem Facet-Wert); sonst erste gefundene')
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

    by_facet = collect_rows_by_facet(registries, args.facet_by)
    print(f'Gefundene {FACET_LABELS[args.facet_by]}: {sorted(by_facet.keys())}')

    if args.pool_instances:
        df = compute_hv_by_facet_pooled(by_facet, args.facet_by)
        value_col = 'relative_pct'
        mode_desc = f'gepoolt ueber alle Instanzen je Facet-Wert ({args.facet_by}, relative_pct)'
        title_suffix = ' (gepoolt, relative HV)'
    else:
        df = compute_hv_by_facet_representative(by_facet, args.instance, args.facet_by)
        value_col = 'hv'
        mode_desc = f'eine repraesentative Instanz je Facet-Wert ({args.facet_by}, rohe HV)'
        title_suffix = ''
        for key in sorted(df[args.facet_by].unique()):
            inst = df.loc[df[args.facet_by] == key, 'instance'].iloc[0]
            print(f'  {args.facet_by} = {key}: Instanz = {inst}')

    out_dir = os.path.join(args.out, args.label)
    out_path = plot_quality_by_facet(df, value_col, out_dir, args.log_scale, title_suffix, args.facet_by)
    csv_out = os.path.join(out_dir, f'quality_by_{args.facet_by}.csv')
    df.sort_values([args.facet_by, 'algorithm', 'repl']).to_csv(csv_out, index = False)
    write_run_info(out_dir, args.note, registries, mode_desc, args.facet_by)

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Rohwerte gespeichert in: {csv_out}')


if __name__ == '__main__':
    main()
