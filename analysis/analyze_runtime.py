#!/usr/bin/env python
'''

Aggregation: anders als analyze_quality_by_size.py (das bei rohen HV-Werten aus
gutem Grund auf EINE Instanz je Panel besteht, siehe dortiger Docstring) wird
hier bewusst UEBER ALLE Instanzen einer Klasse/Groesse gepoolt: Laufzeit in
Sekunden ist nicht in derselben Weise instanzskalenabhaengig wie die rohe
Hypervolume (die vom Kostenniveau der konkreten Instanz abhaengt) -- sie haengt
primaer von n ab, das hier ohnehin schon die Zeilen-Facette ist. Pooling liefert
daher robustere Mittelwerte/Standardabweichungen als nur 10 Wiederholungen einer
einzelnen Instanz.

Nutzung:
  python analysis/analyze_runtime.py results/approximations/ex07 --algo-col operator

  # mehrere Registries kombiniert, z. B. um weitere Instanzgroessen hinzuzunehmen
  python analysis/analyze_runtime.py results/approximations/ex07 \
      results/approximations/ex05:crossover --algo-col operator
'''

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analyze_hypervolume import read_params, get_algo_label, get_git_info
from analyze_quality_by_size import infer_instance_size, infer_instance_class, parse_registry_arg

def is_subgraph_config(algo_label):
    return 'USGS' in algo_label


def collect_runtime_table(registries, instances_filter = None):
    '''Long-format DataFrame mit Spalten size, class, instance, algorithm, repl, runtime,
    column -- eine Zeile je Job mit vorhandener runtime.csv. instances_filter (optional):
    nur diese Instanzpfade beruecksichtigen statt aller in design.csv gefundenen.

    'column' splittet Baseline (kein Subgraph-Mutator) von Subgraph Operators (enthaelt
    USGS), aber NUR wenn tatsaechlich Hybrid-Operatoren (Name enthaelt "+") im Datensatz
    vorkommen -- dort sinnvoll, um z.B. 15 Konfigurationen (ex07) auf zwei lesbare Panels
    zu verteilen. Bei einem reinen Crossover-vs-Mutation-Vergleich ohne Hybride waere der
    Split nur eine willkuerliche Aufspaltung weniger Operatoren auf zwei Diagramme, daher
    dann ein einziges 'All Operators'-Panel.'''
    records = []
    for registry_path, algo_cols in registries:
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        for row in rows:
            if instances_filter is not None and row['instance'] not in instances_filter:
                continue
            rt_path = os.path.join(registry_path, 'experiments', row['jobid'], 'runtime.csv')
            if not os.path.exists(rt_path):
                continue
            with open(rt_path) as f:
                runtime = float(f.read().strip())
            algo = get_algo_label(row, algo_cols)
            records.append({
                'size': infer_instance_size(row['instance']),
                'class': infer_instance_class(row['instance']),
                'instance': row['instance'],
                'algorithm': algo,
                'repl': int(row['repl']),
                'runtime': runtime,
            })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    has_hybrids = df['algorithm'].str.contains('+', regex = False).any()
    if has_hybrids:
        df['column'] = df['algorithm'].apply(lambda a: 'Subgraph Operators' if is_subgraph_config(a) else 'Baseline')
    else:
        df['column'] = 'All Operators'
    return df


def plot_runtime(df, out_dir, title_suffix):
    '''Facet-Grid: Zeilen = Instanzgroesse n, Spalten = 'column' (Baseline/Subgraph
    Operators bei Hybriden im Datensatz, sonst ein einziges 'All Operators'-Panel,
    siehe collect_runtime_table), je Panel ein gruppierter Balkenplot (Operator x
    Instanzklasse, Fehlerbalken = Std) -- direktes Analogon zu Fig. 13 (Bossek & Grimme).'''
    os.makedirs(out_dir, exist_ok = True)
    sizes = sorted(df['size'].unique())
    classes = sorted(df['class'].unique())
    columns = sorted(df['column'].unique())
    colors = dict(zip(classes, matplotlib.colormaps['tab10'](range(len(classes)))))

    agg = df.groupby(['size', 'column', 'algorithm', 'class'])['runtime'].agg(['mean', 'std', 'count']).reset_index()

    fig, axes = plt.subplots(len(sizes), len(columns), squeeze = False,
                              figsize = (6.5 * len(columns), 4.2 * len(sizes)))

    for row_idx, size in enumerate(sizes):
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx][col_idx]
            sub = agg[(agg['size'] == size) & (agg['column'] == column)]
            algos = sorted(sub['algorithm'].unique())
            if not algos:
                ax.axis('off')
                continue

            n_algos = len(algos)
            n_classes = len(classes)
            width = 0.8 / max(n_classes, 1)
            x = range(n_algos)

            for i, cls in enumerate(classes):
                heights, errs = [], []
                for algo in algos:
                    cell = sub[(sub['algorithm'] == algo) & (sub['class'] == cls)]
                    heights.append(cell['mean'].iloc[0] if not cell.empty else 0)
                    errs.append(cell['std'].iloc[0] if not cell.empty and pd.notna(cell['std'].iloc[0]) else 0)
                offsets = [xi + (i - (n_classes - 1) / 2) * width for xi in x]
                label = cls if (row_idx == 0 and col_idx == 0) else None
                ax.bar(offsets, heights, width, yerr = errs, capsize = 3, color = colors[cls], label = label)

            ax.set_xticks(list(x))
            ax.set_xticklabels(algos, rotation = 45 if n_algos > 4 else 0, ha = 'right' if n_algos > 4 else 'center',
                                fontsize = 8 if n_algos <= 10 else 7)
            ax.set_ylim(bottom = 0)
            ax.set_title(f'n = {size}' if len(columns) == 1 else f'n = {size} -- {column}')
            if col_idx == 0:
                ax.set_ylabel('Runtime [s]')

    fig.legend(title = 'Instance Class', loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8)
    n_instances = df['instance'].nunique()
    if n_instances == 1:
        instance_info = os.path.basename(df['instance'].iloc[0])
    else:
        instance_info = f'pooled over {n_instances} instances'
    fig.suptitle(f'Average Runtime per Operator Configuration{title_suffix}\n{instance_info}')
    fig.tight_layout(rect = (0, 0, 0.94, 0.94))

    out_path = os.path.join(out_dir, 'runtime_barplot.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def write_run_info(out_dir, note, registries):
    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write('# Laufzeit-Analyse -- Lauf-Info\n\n')
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (' -- **uncommitted changes!**\n' if diffstat else '\n'))
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
    parser.add_argument('--instances', default = None,
                         help = 'Komma-getrennte Liste von Instanzpfaden (wie in design.csv); nur '
                                'diese werden ausgewertet statt aller in der Registry gefundenen')
    parser.add_argument('--out', default = 'figures/runtime',
                         help = 'Ausgabeverzeichnis (default: figures/runtime)')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out fuer diesen Vergleich (default: _all)')
    parser.add_argument('--note', default = None, help = 'Kurze Notiz, landet in run_info.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    registries = [parse_registry_arg(r, default_algo_cols) for r in args.registries]
    instances_filter = set(args.instances.split(',')) if args.instances else None
    if instances_filter:
        print(f'Instanz-Filter: {sorted(instances_filter)}')

    df = collect_runtime_table(registries, instances_filter)
    if df.empty:
        raise SystemExit('Keine runtime.csv-Dateien gefunden -- Registries pruefen.')
    print(f'Instanzgroessen: {sorted(df["size"].unique())}')
    print(f'Instanzklassen: {sorted(df["class"].unique())}')
    print(f'Konfigurationen: {sorted(df["algorithm"].unique())}')

    out_dir = os.path.join(args.out, args.label)
    out_path = plot_runtime(df, out_dir, '')
    csv_out = os.path.join(out_dir, 'runtime.csv')
    df.sort_values(['size', 'column', 'algorithm', 'class', 'repl']).to_csv(csv_out, index = False)
    write_run_info(out_dir, args.note, registries)

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Rohwerte gespeichert in: {csv_out}')


if __name__ == '__main__':
    main()
