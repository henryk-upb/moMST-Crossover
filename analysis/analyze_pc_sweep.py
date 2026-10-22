#!/usr/bin/env python
'''
Inverted-U-Analyse der Hybrid-Mischung (Mode 1, "either/or") ueber alle vier
Instanzklassen C1-C4 hinweg.

Erwartet die Ausgabe von pc_sweep_by_class_runner_expyrimenter.py: fuer jede
der vier repraesentativen Klasseninstanzen liegt eine Sweep-Reihe ueber
p_crossover = 0.0, 0.1, ..., 1.0 vor (0.0 = reine Mutation, 1.0 = reines XOR,
dazwischen NSGA2's Mode 1 -- ein Zufallswurf pro Nachkomme zwischen Crossover
und Mutation, siehe EA.Algorithms.NSGA2).

Rohe HV ist -- wie in analyze_hypervolume.py dokumentiert -- NICHT zwischen
Instanzen vergleichbar (jede Klasse hat ihre eigene Kantengewicht-Skala; C1
liegt z.B. bei ~4.8 Mio., C4 bei ~5.8 Tsd.). Da hier aber genau das
klassenuebergreifende MUSTER (der umgedrehte U-Shape) im selben Koordinaten-
system verglichen werden soll, wird -- wie compute_cross_instance_table() in
analyze_hypervolume.py es fuer den Algorithmenvergleich tut -- relative_pct
verwendet: je (Instanz, Wiederholung)-Block auf den besten dort gefundenen
p_crossover-Wert normiert (100% = beste Stufe dieser Wiederholung). Das macht
alle vier Klassen direkt vergleichbar, ohne dass die absolute Skala den
Formvergleich verzerrt. Rohe HV bleibt zusaetzlich in der CSV erhalten.

Zwei Plots:
  * pc_sweep_by_class.png: eine Linie je Instanzklasse (Farbe = Klasse) im
    gemeinsamen Koordinatensystem (relative_pct, s.o.), um ein klassen-
    uebergreifendes Muster sichtbar zu machen -- einen umgedrehten U-Shape mit
    Maximum irgendwo zwischen den reinen Operatoren (Ueberlegenheit des
    Hybrids) sowie -- durch Vergleich der beiden Endpunkte -- ob reines
    Crossover reiner Mutation ueberlegen ist.
  * pc_sweep_by_class_grid.png: dieselbe Sweep-Kurve als Facet-Grid, ein Panel
    je Klasse mit EIGENER (roher, nicht relativer) HV-Skala, damit auch
    Unterschiede sichtbar werden, die im gemeinsamen Koordinatensystem von der
    groesseren Skalen-Differenz zwischen den Klassen ueberdeckt wuerden. Jedes
    Panel zeigt zusaetzlich zur finalen Front (durchgezogen, Gen 100, aus
    pf.csv) den Zwischenstand nach 20 (gepunktet) und 50 Generationen
    (gestrichelt, aus trajectory.csv; zwischen 50 und 100 aendert sich kaum
    noch etwas, daher kein weiterer Zwischenschritt) -- so wird sichtbar, ob/wie sich die
    optimale p_crossover-Stufe im Laufe der Optimierung verschiebt.

Nutzung:
  python analysis/analyze_pc_sweep.py results/approximations/ex09_pc_sweep_by_class
'''

import argparse
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analyze_hypervolume import (
    read_params, collect_fronts, collect_trajectories, determine_reference_point,
    compute_hv_table, compute_hv_progress_table, infer_instance_class, archive_generic_run,
)

PROGRESS_GENS = (20, 50)  # intermediate generations shown as dotted/dashed lines in
# plot_pc_sweep_grid(); gen 100 is already the solid "final" line (from pf.csv), and
# 50-100 barely changes (see hv_progress.png elsewhere), so nothing beyond 50 is added.


def compute_pc_sweep_tables(rows, registry_path, progress_gens=PROGRESS_GENS):
    '''Returns (df_final, df_progress).

    df_final: class, instance, p_crossover, repl, hv, rank, relative_pct -- one row per
    (instance, p_crossover, repl), final (gen=100) front. Reference point shared within each
    instance (final fronts + trajectory snapshots, like analyze_hypervolume's
    determine_reference_point); relative_pct = 100 * hv / (best hv among the 11 p_crossover
    stages of that instance+repl) -- scale-independent across classes, same principle as
    analyze_hypervolume.compute_cross_instance_table (reimplemented here, not called directly,
    so the SAME per-instance reference point can be reused for df_progress below).

    df_progress: class, instance, p_crossover, repl, gen, hv -- HV of the population snapshot
    after `gen` generations (from trajectory.csv, log_every=10 in the runner), restricted to
    progress_gens, for the dashed/dotted intermediate-progress lines in plot_pc_sweep_grid().'''
    instances = sorted(set(row['instance'] for row in rows))
    final_frames = []
    progress_frames = []
    for instance in instances:
        by_pc = collect_fronts(rows, registry_path, instance, ['p_crossover'])
        if not by_pc:
            continue
        by_pc_traj = collect_trajectories(rows, registry_path, instance, ['p_crossover'])
        ref = determine_reference_point(by_pc, by_pc_traj)
        cls = infer_instance_class(instance)

        df_final = compute_hv_table(by_pc, ref)
        df_final['instance'] = instance
        df_final['class'] = cls
        final_frames.append(df_final)

        df_prog = compute_hv_progress_table(by_pc_traj, ref)
        df_prog = df_prog[df_prog['gen'].isin(progress_gens)].copy()
        df_prog['instance'] = instance
        df_prog['class'] = cls
        progress_frames.append(df_prog)

        print(f"{cls} ({os.path.basename(instance)}): Referenzpunkt = {ref}, {len(df_final)} Laeufe")

    df_final = pd.concat(final_frames, ignore_index=True)
    df_final = df_final.rename(columns={'algorithm': 'p_crossover'})
    df_final['p_crossover'] = df_final['p_crossover'].astype(float)
    df_final['rank'] = df_final.groupby(['instance', 'repl'])['hv'].rank(ascending=False, method='average')
    block_max = df_final.groupby(['instance', 'repl'])['hv'].transform('max')
    df_final['relative_pct'] = 100.0 * df_final['hv'] / block_max

    df_progress = pd.concat(progress_frames, ignore_index=True)
    df_progress = df_progress.rename(columns={'algorithm': 'p_crossover'})
    df_progress['p_crossover'] = df_progress['p_crossover'].astype(float)

    return df_final, df_progress


def plot_pc_sweep(df, out_dir):
    '''Eine Linie je Instanzklasse (Farbe = Klasse), x = p_crossover, y = mittlere
    relative_pct (100 % = beste p_crossover-Stufe je Instanz+Wiederholung) ueber die
    Wiederholungen, mit 25%-75%-Quantilband -- direkter, skalenunabhaengiger Vergleich
    des umgedrehten U-Shapes ueber alle vier Klassen im selben Koordinatensystem.'''
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df['class'].unique())
    colors = dict(zip(classes, matplotlib.colormaps['tab10'](range(len(classes)))))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in classes:
        sub = df[df['class'] == cls]
        agg = sub.groupby('p_crossover')['relative_pct'].agg(
            ['mean', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        agg.columns = ['mean', 'q25', 'q75']
        agg = agg.sort_index()
        line, = ax.plot(agg.index, agg['mean'], marker='o', color=colors[cls], label=cls)
        ax.fill_between(agg.index, agg['q25'], agg['q75'], color=line.get_color(), alpha=0.15)

    ax.set_xlabel(r'$p_c$ (Mode 1: 0 = pure mutation, 1 = pure crossover)')
    ax.set_ylabel('Relative Hypervolume (%, mean over repetitions)')
    ax.set_title('Hybrid Mixture Sweep: XOR (pure) x USGS-PRE, Mode 1, 100 Generations\n'
                 '(100% = best $p_c$ stage per instance+repetition, n=10 repetitions)', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(title='Instance Class')
    fig.tight_layout()

    out_path = os.path.join(out_dir, 'pc_sweep_by_class.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


PROGRESS_LINESTYLES = {20: ':', 50: '--'}  # final (gen=100, from pf.csv) is always solid '-'


def plot_pc_sweep_grid(df_final, df_progress, out_dir, progress_gens=PROGRESS_GENS):
    '''Facet-Grid-Variante von plot_pc_sweep(): ein Panel je Instanzklasse, je mit
    EIGENER y-Achsen-Skala und roher (nicht relativer) Hypervolume -- damit auch
    kleine, innerhalb der Klasse reale Unterschiede sichtbar werden, die im
    gemeinsamen Koordinatensystem (plot_pc_sweep(), alle Klassen auf einer Skala)
    von der viel groesseren Streuung zwischen den Klassen ueberdeckt wuerden (C1
    liegt HV-maessig bei ~10 Mio., C4 bei ~1.4 Mio. -- siehe Moduldoc). Gleiches
    Prinzip wie analyze_hypervolume.plot_distributions_by_class().

    Zusaetzlich zur durchgezogenen Linie (finale Front nach 100 Generationen, aus
    pf.csv) zeigt jedes Panel den Zwischenstand nach 20 (gepunktet) und 50
    Generationen (gestrichelt, aus trajectory.csv, log_every=10 im Runner)
    -- so wird sichtbar, ob sich die optimale
    p_crossover-Stufe waehrend des Laufs verschiebt oder ueber die Zeit stabil
    bleibt. Zwischen 50 und 100 Generationen aendert sich kaum noch etwas,
    daher endet die Zwischenstands-Reihe bei 50.'''
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df_final['class'].unique())
    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)
    colors = dict(zip(classes, matplotlib.colormaps['tab10'](range(len(classes)))))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5.5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten()

    def agg_line(sub, value_col):
        agg = sub.groupby('p_crossover')[value_col].agg(
            ['mean', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        agg.columns = ['mean', 'q25', 'q75']
        return agg.sort_index()

    for ax, cls in zip(axes_flat, classes):
        color = colors[cls]
        sub_final = df_final[df_final['class'] == cls]
        # display only: some instance file names (C3/C4) contain "--" (e.g. "-0.95--CORWG"),
        # normalized to "_" here purely for a visually consistent panel subtitle
        instance_label = os.path.basename(sub_final['instance'].iloc[0]).replace('--', '_')

        agg = agg_line(sub_final, 'hv')
        ax.plot(agg.index, agg['mean'], marker='o', color=color, linestyle='-', label='Gen 100 (final)')
        ax.fill_between(agg.index, agg['q25'], agg['q75'], color=color, alpha=0.15)

        sub_prog = df_progress[df_progress['class'] == cls]
        for gen in progress_gens:
            gen_sub = sub_prog[sub_prog['gen'] == gen]
            if gen_sub.empty:
                continue
            agg_g = agg_line(gen_sub, 'hv')
            ax.plot(agg_g.index, agg_g['mean'], marker='.', color=color,
                     linestyle=PROGRESS_LINESTYLES.get(gen, '--'), alpha=0.7, label=f'Gen {gen}')

        ax.set_xlabel(r'$p_c$')
        ax.set_ylabel('Hypervolume')
        ax.set_title(f'Class {cls}\n{instance_label}', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.legend(fontsize=7)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle('Hybrid Mixture Sweep per Instance Class: XOR (pure) x USGS-PRE, Mode 1, 100 Generations\n'
                 '(mean +/- 25%-75% quantile band over 10 repetitions, '
                 'dotted/dashed = gen 20/50, solid = gen 100)', fontsize=11, y=0.99)

    out_path = os.path.join(out_dir, 'pc_sweep_by_class_grid.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path',
                         help='z.B. results/approximations/ex09_pc_sweep_by_class '
                              '(enthaelt design.csv und experiments/<jobid>/pf.csv)')
    parser.add_argument('--out', default='figures/pc_sweep',
                         help='Ausgabeverzeichnis (default: figures/pc_sweep); wird bei jedem Lauf ueberschrieben')
    parser.add_argument('--archive-dir', default='analysis_log',
                         help='Basisverzeichnis des Verlaufs-Archivs (default: analysis_log)')
    parser.add_argument('--no-archive', action='store_true',
                         help='Verlaufs-Archivierung fuer diesen Lauf abschalten (z.B. fuer Testlaeufe)')
    parser.add_argument('--note', default=None,
                         help='Kurze Notiz, was sich seit dem letzten Archiv-Eintrag geaendert hat; '
                              'landet in summary.md und CHANGELOG.md')
    args = parser.parse_args()

    design_path = os.path.join(args.registry_path, 'design.csv')
    rows = read_params(design_path)
    experiment_label = os.path.basename(os.path.normpath(args.registry_path))

    df, df_progress = compute_pc_sweep_tables(rows, args.registry_path)

    out_dir = os.path.join(args.out, experiment_label)
    out_path = plot_pc_sweep(df, out_dir)
    print(f'\nPlot (klassenuebergreifend, relative_pct) gespeichert in: {out_path}')
    grid_out_path = plot_pc_sweep_grid(df, df_progress, out_dir)
    print(f'Plot (je Klasse, eigene Skala, inkl. Gen-10/50-Zwischenstand) gespeichert in: {grid_out_path}')

    csv_out = os.path.join(out_dir, 'pc_sweep_hv.csv')
    df.sort_values(['class', 'p_crossover', 'repl']).to_csv(csv_out, index=False)
    print(f'HV-Rohwerte (inkl. relative_pct) gespeichert in: {csv_out}')

    progress_csv_out = os.path.join(out_dir, 'pc_sweep_progress_hv.csv')
    df_progress.sort_values(['class', 'p_crossover', 'gen', 'repl']).to_csv(progress_csv_out, index=False)
    print(f'HV-Zwischenstand (Gen 20/50) gespeichert in: {progress_csv_out}')

    agg = df.groupby(['class', 'p_crossover'])[['hv', 'relative_pct']].agg(['mean', 'std']).reset_index()
    agg.columns = ['class', 'p_crossover', 'hv_mean', 'hv_std', 'relative_pct_mean', 'relative_pct_std']
    print('\n=== Mittlere (rohe + relative) Hypervolume je Klasse x p_crossover ===')
    print(agg.to_string(index=False))

    if not args.no_archive:
        rows_summary = agg.to_dict('records')
        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, 'pc_sweep_by_class',
            title='Hybrid Mixture Sweep (Mode 1) je Instanzklasse', note=args.note,
            copy_paths=[out_path, grid_out_path, csv_out, progress_csv_out],
            changelog_header="| Datum | Commit | Klasse | p_crossover | Mean HV | Mean relative_pct | Notiz |\n"
                              "|---|---|---|---|---|---|---|\n",
            row_template="| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {class} | {p_crossover} "
                         "| {hv_mean:.2f} | {relative_pct_mean:.2f}% | {note} |",
            changelog_rows=rows_summary)
        print(f'\nArchiviert in: {entry_dir}')
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, experiment_label, 'CHANGELOG.md')}")


if __name__ == '__main__':
    main()
