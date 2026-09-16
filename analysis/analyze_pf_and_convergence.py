#!/usr/bin/env python


import argparse
import math
import os
import random

import matplotlib
import matplotlib.pyplot as plt

from analyze_hypervolume import (
    read_params, collect_fronts, collect_trajectories, determine_reference_point,
    compute_hv_progress_table, get_git_info, archive_generic_run, infer_instance_class,
)


def plot_pf_and_convergence(by_algo, by_algo_traj, ref, out_dir, instance_label,
                             repl_for_scatter = None, show_all_repls = False, hide_first_gen = False):
    os.makedirs(out_dir, exist_ok = True)
    algos = sorted(by_algo.keys())
    n = len(algos)
    colors = matplotlib.colormaps['tab10' if n <= 10 else 'tab20'](range(n))

    fig, (ax_pf, ax_conv) = plt.subplots(1, 2, figsize = (12, 5.2))

    # --- links: Pareto-Front-Approximation (wie Fig. 8 links) ---
    for color, algo in zip(colors, algos):
        repls = by_algo[algo]
        if show_all_repls:
            for j, (repl, points) in enumerate(repls):
                xs, ys = zip(*points)
                ax_pf.scatter(xs, ys, s = 10, alpha = 0.5, color = color,
                               label = algo if j == 0 else None)
        else:
            wanted = repl_for_scatter or repls[0][0]
            match = next((points for repl, points in repls if repl == wanted), repls[0][1])
            xs, ys = zip(*match)
            ax_pf.scatter(xs, ys, s = 18, color = color, label = algo)

    ax_pf.set_xlabel('$c_1$')
    ax_pf.set_ylabel('$c_2$')
    ax_pf.set_title(f'Pareto Front Approximation\n{instance_label}')
    if n > 8:
        ax_pf.legend(loc = 'center left', bbox_to_anchor = (1.02, 0.5), fontsize = 8, ncol = 1 + n // 20)
    else:
        ax_pf.legend(fontsize = 8)

    # --- rechts: Konvergenztrajektorie (dominierte HV ueber Generation, wie Fig. 8 rechts) ---
    df_progress = compute_hv_progress_table(by_algo_traj, ref)
    conv_df = df_progress
    if hide_first_gen:
        # der Sprung von der zufaelligen Startpopulation zur ersten geloggten Generation ist um
        # Groessenordnungen groesser als alle spaeteren Unterschiede und staucht sonst die y-Achse
        # so, dass die eigentlich interessanten spaeteren Unterschiede kaum noch sichtbar sind
        conv_df = df_progress[df_progress['gen'] > df_progress['gen'].min()]
    for color, algo in zip(colors, algos):
        sub = conv_df[conv_df['algorithm'] == algo]
        agg = sub.groupby('gen')['hv'].agg(['mean', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        agg.columns = ['mean', 'q25', 'q75']
        agg = agg.sort_index()
        ax_conv.plot(agg.index, agg['mean'], marker = 'o', markersize = 3, color = color, label = algo)
        ax_conv.fill_between(agg.index, agg['q25'], agg['q75'], color = color, alpha = 0.15)

    conv_title_suffix = ' (first generation hidden)' if hide_first_gen else ''
    ax_conv.set_xlabel('Generation')
    ax_conv.set_ylabel('Dominated Hypervolume')
    ax_conv.set_title(f'Convergence Trajectory{conv_title_suffix}\n{instance_label}')
    if n > 8:
        ax_conv.legend(loc = 'center left', bbox_to_anchor = (1.02, 0.5), fontsize = 8, ncol = 1 + n // 20)
    else:
        ax_conv.legend(fontsize = 8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'pf_and_convergence.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path, df_progress


def plot_pf_by_class(by_class, out_dir, repl_for_scatter = None, show_all_repls = False):
    '''Facet-Grid-Variante der PF-Approximation: EIN Bild mit einem Panel je
    Instanzklasse (2x2 bei 4 Klassen) statt separater Bilder je Klasse.
    by_class: {class_label: (by_algo, instance_label)}.'''
    os.makedirs(out_dir, exist_ok = True)
    classes = sorted(by_class.keys())
    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze = False, figsize = (6 * ncols, 5 * nrows))
    axes_flat = axes.flatten()

    for ax, cls in zip(axes_flat, classes):
        by_algo, instance_label = by_class[cls]
        algos = sorted(by_algo.keys())
        n = len(algos)
        colors = matplotlib.colormaps['tab10' if n <= 10 else 'tab20'](range(n))
        for color, algo in zip(colors, algos):
            repls = by_algo[algo]
            if show_all_repls:
                for j, (repl, points) in enumerate(repls):
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, s = 10, alpha = 0.5, color = color, label = algo if j == 0 else None)
            else:
                wanted = repl_for_scatter or repls[0][0]
                match = next((points for repl, points in repls if repl == wanted), repls[0][1])
                xs, ys = zip(*match)
                ax.scatter(xs, ys, s = 18, color = color, label = algo)
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')
        ax.set_title(f'Class {cls}\n{instance_label}', fontsize = 9)
        ax.legend(fontsize = 7)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')

    fig.suptitle('Pareto Front Approximation -- per instance class')
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    out_path = os.path.join(out_dir, 'pf_approximation.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def plot_convergence_by_class(by_class_progress, out_dir, hide_first_gen = False):
    '''Facet-Grid-Variante der Konvergenztrajektorie: EIN Bild mit einem
    Panel je Instanzklasse. by_class_progress: {class_label: (df_progress, instance_label)}.'''
    os.makedirs(out_dir, exist_ok = True)
    classes = sorted(by_class_progress.keys())
    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze = False, figsize = (6 * ncols, 4.3 * nrows))
    axes_flat = axes.flatten()

    for ax, cls in zip(axes_flat, classes):
        df_progress, instance_label = by_class_progress[cls]
        conv_df = df_progress
        if hide_first_gen:
            conv_df = df_progress[df_progress['gen'] > df_progress['gen'].min()]
        algos = sorted(conv_df['algorithm'].unique())
        n = len(algos)
        colors = matplotlib.colormaps['tab10' if n <= 10 else 'tab20'](range(n))
        for color, algo in zip(colors, algos):
            sub = conv_df[conv_df['algorithm'] == algo]
            agg = sub.groupby('gen')['hv'].agg(['mean', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
            agg.columns = ['mean', 'q25', 'q75']
            agg = agg.sort_index()
            ax.plot(agg.index, agg['mean'], marker = 'o', markersize = 3, color = color, label = algo)
            ax.fill_between(agg.index, agg['q25'], agg['q75'], color = color, alpha = 0.15)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Dominated Hypervolume')
        ax.set_title(f'Class {cls}\n{instance_label}', fontsize = 9)
        ax.legend(fontsize = 7)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')

    title_suffix = ' (first generation hidden)' if hide_first_gen else ''
    fig.suptitle(f'Convergence Trajectory -- per instance class{title_suffix}')
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    out_path = os.path.join(out_dir, 'convergence_trajectory.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registry_path',
                         help = 'z.B. results/approximations/ex07 (enthaelt design.csv und '
                                'experiments/<jobid>/{pf,trajectory}.csv)')
    parser.add_argument('--instance', default = None,
                         help = 'Instanzpfad wie in design.csv; '
                                'Default: zufaellig gewaehlte Instanz')
    parser.add_argument('--facet-class-instances', default = None,
                         help = 'Komma-getrennte Liste von Instanzpfaden (eine je Klasse) -- statt '
                                'Einzel-Instanz werden zwei Bilder erzeugt (pf_approximation.png, '
                                'convergence_trajectory.png), jeweils mit einem Panel je Instanzklasse '
                                '(2x2 bei 4 Klassen). Ueberschreibt --instance.')
    parser.add_argument('--algo-col', default = 'crossover',
                         help = "Spaltenname(n) der Algorithmus-/Operator-Bezeichnung, komma-getrennt "
                                "fuer kombinierte Algorithmen (default: 'crossover')")
    parser.add_argument('--repl', type = int, default = None,
                         help = 'Welche Wiederholung im linken Panel gezeigt wird '
                                '(default: erste gefundene je Algorithmus)')
    parser.add_argument('--all-repls', action = 'store_true',
                         help = 'Statt einer Wiederholung ALLE im linken Panel ueberlagern (dichtere, '
                                'aber unuebersichtlichere Ansicht, aehnlich Fig. 12 aus Bossek '
                                '& Grimme 2023)')
    parser.add_argument('--hide-first-gen', action = 'store_true',
                         help = 'Die erste geloggte Generation (Sprung von der Zufalls-Startpopulation) '
                                'im rechten Konvergenz-Panel ausblenden, damit die y-Achse sich auf die '
                                'feineren Unterschiede der spaeteren Generationen fokussiert -- betrifft '
                                'nur die Darstellung, nicht die gespeicherten convergence_values.csv')
    parser.add_argument('--out', default = 'figures/pf_and_convergence',
                         help = 'Ausgabeverzeichnis fuer den "aktueller Stand"-Blick (default: '
                                'figures/pf_and_convergence); wird bei jedem Lauf ueberschrieben')
    parser.add_argument('--archive-dir', default = 'analysis_log',
                         help = 'Basisverzeichnis des Verlaufs-Archivs (default: analysis_log). Jeder Lauf legt hier '
                                'zusaetzlich eine mit Zeitstempel+Commit gelabelte, nie ueberschriebene Kopie ab.')
    parser.add_argument('--no-archive', action = 'store_true',
                         help = 'Verlaufs-Archivierung fuer diesen Lauf abschalten (z.B. fuer Testlaeufe)')
    parser.add_argument('--note', default = None,
                         help = 'Kurze Notiz, was sich seit dem letzten Archiv-Eintrag geaendert hat; '
                                'landet in run_info.md, summary.md und CHANGELOG.md')
    args = parser.parse_args()

    design_path = os.path.join(args.registry_path, 'design.csv')
    rows = read_params(design_path)
    experiment_label = os.path.basename(os.path.normpath(args.registry_path))
    algo_cols = args.algo_col.split(',')

    if args.facet_class_instances:
        instances = args.facet_class_instances.split(',')
        by_class, by_class_progress = {}, {}
        for instance in instances:
            cls = infer_instance_class(instance)
            by_algo = collect_fronts(rows, args.registry_path, instance, algo_cols)
            if not by_algo:
                print(f"WARNUNG: keine Jobs fuer Instanz '{instance}' gefunden -- uebersprungen.")
                continue
            by_algo_traj = collect_trajectories(rows, args.registry_path, instance, algo_cols)
            ref = determine_reference_point(by_algo, by_algo_traj)
            df_progress = compute_hv_progress_table(by_algo_traj, ref)
            instance_label = os.path.basename(instance)
            by_class[cls] = (by_algo, instance_label)
            by_class_progress[cls] = (df_progress, instance_label)
            print(f'Klasse {cls}: Instanz = {instance} (Referenzpunkt: {ref})')

        out_dir = os.path.join(args.out, experiment_label, '_by_class')
        pf_path = plot_pf_by_class(by_class, out_dir,
                                    repl_for_scatter = str(args.repl) if args.repl is not None else None,
                                    show_all_repls = args.all_repls)
        conv_path = plot_convergence_by_class(by_class_progress, out_dir, hide_first_gen = args.hide_first_gen)
        print(f'\nPlots gespeichert in: {out_dir}')
        print(f'  {pf_path}')
        print(f'  {conv_path}')
        return

    instance = args.instance or random.choice(sorted(set(row['instance'] for row in rows)))
    instance_label = os.path.basename(instance)

    by_algo = collect_fronts(rows, args.registry_path, instance, algo_cols)
    if not by_algo:
        raise SystemExit(f"Keine Jobs fuer Instanz '{instance}' in {design_path} gefunden.")
    by_algo_traj = collect_trajectories(rows, args.registry_path, instance, algo_cols)

    ref = determine_reference_point(by_algo, by_algo_traj)
    print(f'Instanz: {instance}')
    print(f'Algorithmen ({args.algo_col}): {sorted(by_algo.keys())}')
    print(f'Referenzpunkt fuer HV: {ref}')

    out_dir = os.path.join(args.out, experiment_label, instance_label)
    out_path, df_progress = plot_pf_and_convergence(
        by_algo, by_algo_traj, ref, out_dir, instance_label,
        repl_for_scatter = str(args.repl) if args.repl is not None else None,
        show_all_repls = args.all_repls, hide_first_gen = args.hide_first_gen)

    progress_csv = os.path.join(out_dir, 'convergence_values.csv')
    df_progress.sort_values(['algorithm', 'repl', 'gen']).to_csv(progress_csv, index = False)

    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write('# Pareto-Front + Konvergenz -- Lauf-Info\n\n')
        f.write(f'- Instanz: `{instance}`\n')
        f.write(f'- Registry: `{args.registry_path}` (Spalte(n): {args.algo_col})\n')
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (' -- **uncommitted changes!**\n' if diffstat else '\n'))
        if args.note:
            f.write(f'\n## Notiz\n\n{args.note}\n')

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Konvergenz-Rohwerte gespeichert in: {progress_csv}')

    if not args.no_archive:
        # finale (letzte) Generation je Algorithmus, ueber Wiederholungen aggregiert
        last_gen = df_progress.groupby('algorithm')['gen'].transform('max')
        final_df = df_progress[df_progress['gen'] == last_gen]
        final_stats = final_df.groupby('algorithm')['hv'].agg(['median', 'mean', 'count'])
        rows = [{'algo': algo, 'median': r['median'], 'mean': r['mean'], 'n': int(r['count'])}
                for algo, r in final_stats.iterrows()]

        entry_dir = archive_generic_run(
            args.archive_dir, experiment_label, instance_label,
            title = 'Pareto-Front + Konvergenz-Analyse', note = args.note,
            copy_paths = [out_path, progress_csv, os.path.join(out_dir, 'run_info.md')],
            changelog_header = (
                "| Datum | Commit | Instanz | Algorithmus | Finale HV (Median) | Finale HV (Mean) | n | Notiz |\n"
                "|---|---|---|---|---|---|---|---|\n"),
            row_template = ("| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | " + instance_label
                             + " | {algo} | {median:.2f} | {mean:.2f} | {n} | {note} |"),
            changelog_rows = rows)
        print(f'Archiviert in: {entry_dir}')
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, experiment_label, 'CHANGELOG.md')}")


if __name__ == '__main__':
    main()
