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

COLUMNS = ['Baseline', 'Subgraph-Operatoren']


def is_subgraph_config(algo_label):
    return 'USGS' in algo_label


def collect_runtime_table(registries):
    '''Long-format DataFrame mit Spalten size, class, instance, algorithm, repl, runtime,
    column (Baseline/Subgraph-Operatoren) -- eine Zeile je Job mit vorhandener runtime.csv.'''
    records = []
    for registry_path, algo_cols in registries:
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        for row in rows:
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
                'column': 'Subgraph-Operatoren' if is_subgraph_config(algo) else 'Baseline',
            })
    return pd.DataFrame.from_records(records)


def plot_runtime(df, out_dir, title_suffix):
    '''Facet-Grid: Zeilen = Instanzgroesse n, Spalten = Baseline/Subgraph-Operatoren,
    je Panel ein gruppierter Balkenplot (Operator x Instanzklasse, Fehlerbalken = Std) --
    direktes Analogon zu Fig. 13 (Bossek & Grimme).'''
    os.makedirs(out_dir, exist_ok = True)
    sizes = sorted(df['size'].unique())
    classes = sorted(df['class'].unique())
    colors = dict(zip(classes, matplotlib.colormaps['tab10'](range(len(classes)))))

    agg = df.groupby(['size', 'column', 'algorithm', 'class'])['runtime'].agg(['mean', 'std', 'count']).reset_index()

    fig, axes = plt.subplots(len(sizes), len(COLUMNS), squeeze = False,
                              figsize = (6.5 * len(COLUMNS), 4.2 * len(sizes)))

    for row_idx, size in enumerate(sizes):
        for col_idx, column in enumerate(COLUMNS):
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
            ax.set_title(f'n = {size} -- {column}')
            if col_idx == 0:
                ax.set_ylabel('Laufzeit [s]')

    fig.legend(title = 'Instanzklasse', loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8)
    fig.suptitle(f'Durchschnittliche Laufzeit je Operator-Konfiguration{title_suffix}')
    fig.tight_layout(rect = (0, 0, 0.94, 0.96))

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
    parser.add_argument('--out', default = 'figures/runtime',
                         help = 'Ausgabeverzeichnis (default: figures/runtime)')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out fuer diesen Vergleich (default: _all)')
    parser.add_argument('--note', default = None, help = 'Kurze Notiz, landet in run_info.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    registries = [parse_registry_arg(r, default_algo_cols) for r in args.registries]

    df = collect_runtime_table(registries)
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
