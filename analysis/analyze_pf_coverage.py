#!/usr/bin/env python
'''
Pareto-Front-Abdeckung je Instanzklasse.

Im Gegensatz zu analyze_pf_and_convergence.py (eine Instanz je Klasse, Farbe
= Algorithmus) zeigt dieses Skript ALLE Instanzen einer Klasse in einem
gemeinsamen Panel, Farbe = Instanz. Ueber alle Algorithmen/Wiederholungen
gepoolte Endpopulationen je Instanz werden dabei per-Instanz auf [0, 1]
normalisiert (eigenes Minimum/Maximum je Instanz), da die rohen c1/c2-Werte
zwischen Instanzen nicht direkt vergleichbar sind (unterschiedliche
Kantengewichte). So laesst sich pruefen, ob die approximierte Front eine
Klasse gleichmaessig abdeckt oder ob es instanzabhaengige Bereiche gibt, die
von keinem Lauf erreicht werden (Blindspots).
'''

import argparse
import math
import os
import re

import matplotlib
import matplotlib.pyplot as plt

from analyze_hypervolume import read_params, infer_instance_class, collect_fronts, get_algo_label, archive_generic_run


INSTANCE_SUFFIX_RE = re.compile(r'_(\d+)\.graph$')


def instance_short_label(instance_path):
    '''Kurzes Label fuer die Legende, z.B. "CLASS1_100_..._6.graph" -> "#6".'''
    m = INSTANCE_SUFFIX_RE.search(os.path.basename(instance_path))
    return f'#{m.group(1)}' if m else os.path.basename(instance_path)


def collect_pooled_points(params_rows, registry_path, instance, algo_cols, algo_filter = None):
    '''Alle pf.csv-Punkte einer Instanz, ueber Algorithmen und Wiederholungen gepoolt
    (optional auf einen einzelnen Algorithmus gefiltert).'''
    by_algo = collect_fronts(params_rows, registry_path, instance, algo_cols)
    if algo_filter is not None:
        by_algo = {algo: repls for algo, repls in by_algo.items() if algo == algo_filter}
    return [p for repls in by_algo.values() for _, points in repls for p in points]


def normalize_points(points):
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    return [((x - xmin) / xspan, (y - ymin) / yspan) for x, y in points]


def plot_pf_coverage_by_class(by_class_points, out_dir, normalize):
    '''by_class_points: {class_label: {instance_path: [(c1, c2), ...]}}.'''
    os.makedirs(out_dir, exist_ok = True)
    classes = sorted(by_class_points.keys())
    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze = False, figsize = (6.5 * ncols, 5.5 * nrows))
    axes_flat = axes.flatten()

    for ax, cls in zip(axes_flat, classes):
        by_instance = by_class_points[cls]
        instances = sorted(by_instance.keys())
        n = len(instances)
        colors = matplotlib.colormaps['tab10' if n <= 10 else 'tab20'](range(n))
        for color, instance in zip(colors, instances):
            points = by_instance[instance]
            if normalize:
                points = normalize_points(points)
            xs, ys = zip(*points)
            ax.scatter(xs, ys, s = 10, alpha = 0.5, color = color, label = instance_short_label(instance))
        axis_suffix = ' (normalized)' if normalize else ''
        ax.set_xlabel(f'$c_1${axis_suffix}')
        ax.set_ylabel(f'$c_2${axis_suffix}')
        ax.set_title(f'Class {cls} ({n} instances)', fontsize = 9)
        ax.legend(fontsize = 6, loc = 'upper right', ncol = 1 + n // 12)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')

    title_suffix = ', normalized per instance' if normalize else ''
    fig.suptitle(f'Pareto Front Coverage per Instance Class{title_suffix}')
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    out_path = os.path.join(out_dir, 'pf_coverage_by_class.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path',
                         help = 'z.B. results/approximations/ex05 (enthaelt design.csv und '
                                'experiments/<jobid>/pf.csv)')
    parser.add_argument('--algo-col', default = 'crossover',
                         help = "Spaltenname(n) der Algorithmus-/Operator-Bezeichnung, komma-getrennt "
                                "fuer kombinierte Algorithmen (default: 'crossover')")
    parser.add_argument('--algo', default = None,
                         help = 'Falls gesetzt: nur diesen einen Algorithmus einbeziehen statt ueber '
                                'alle Algorithmen zu poolen (Wert wie in --algo-col kombiniert, z.B. "XOR+USGS-PRE")')
    parser.add_argument('--no-normalize', action = 'store_true',
                         help = 'Rohe c1/c2-Werte statt Min-Max-Normalisierung je Instanz verwenden '
                                '(nur sinnvoll, wenn alle Instanzen vergleichbare Gewichtsskalen haben)')
    parser.add_argument('--out', default = 'figures/pf_coverage',
                         help = 'Ausgabeverzeichnis (default: figures/pf_coverage); wird bei jedem Lauf ueberschrieben')
    parser.add_argument('--archive-dir', default = 'analysis_log',
                         help = 'Basisverzeichnis des Verlaufs-Archivs (default: analysis_log)')
    parser.add_argument('--no-archive', action = 'store_true',
                         help = 'Verlaufs-Archivierung fuer diesen Lauf abschalten (z.B. fuer Testlaeufe)')
    parser.add_argument('--note', default = None,
                         help = 'Kurze Notiz, was sich seit dem letzten Archiv-Eintrag geaendert hat; '
                                'landet in run_info.md/summary.md/CHANGELOG.md')
    args = parser.parse_args()

    design_path = os.path.join(args.registry_path, 'design.csv')
    rows = read_params(design_path)
    experiment_label = os.path.basename(os.path.normpath(args.registry_path))
    algo_cols = args.algo_col.split(',')

    instances = sorted(set(row['instance'] for row in rows))
    by_class_points = {}
    for instance in instances:
        cls = infer_instance_class(instance)
        points = collect_pooled_points(rows, args.registry_path, instance, algo_cols, algo_filter = args.algo)
        if not points:
            print(f"WARNUNG: keine Punkte fuer Instanz '{instance}' -- uebersprungen.")
            continue
        by_class_points.setdefault(cls, {})[instance] = points

    for cls in sorted(by_class_points.keys()):
        print(f'Klasse {cls}: {len(by_class_points[cls])} Instanzen')

    out_dir = os.path.join(args.out, experiment_label)
    out_path = plot_pf_coverage_by_class(by_class_points, out_dir, normalize = not args.no_normalize)
    print(f'\nPlot gespeichert in: {out_path}')

    if not args.no_archive:
        rows_summary = [{'class': cls, 'n_instances': len(by_class_points[cls])}
                         for cls in sorted(by_class_points.keys())]
        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, 'pf_coverage',
            title = 'Pareto-Front-Abdeckung je Instanzklasse', note = args.note,
            copy_paths = [out_path],
            changelog_header = "| Datum | Commit | Klasse | # Instanzen | Notiz |\n|---|---|---|---|---|\n",
            row_template = "| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {n_instances} | {note} |",
            changelog_rows = rows_summary)
        print(f'Archiviert in: {entry_dir}')
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, experiment_label, 'CHANGELOG.md')}")


if __name__ == '__main__':
    main()
