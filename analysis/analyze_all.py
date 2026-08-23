#!/usr/bin/env python
'''
Fuehrt alle Analyse-Skripte (analyze_hypervolume.py, analyze_pf_and_convergence.py,
analyze_quality_by_size.py -- einmal je Groesse UND einmal je Instanzklasse --,
analyze_mean_rankings.py, analyze_runtime.py) fuer denselben Lauf/dieselben
Registries aus, sammelt alle Diagramm-Saetze unter einem gemeinsamen
--out-Verzeichnis ein und archiviert sie anschliessend zusammen als EINEN
Eintrag unter analysis_log/full_report/ -- statt je einem separaten Archiv-
Eintrag pro Skript, wie es beim einzelnen Aufruf passieren wuerde.

Jedes Teilskript wird als eigener Subprozess aufgerufen (sys.executable, also
dieselbe Python-Umgebung wie dieser Orchestrator) -- so bleiben die Skripte
unveraendert und unabhaengig weiter einzeln nutzbar. analyze_hypervolume.py
und analyze_pf_and_convergence.py akzeptieren nur EINE Registry (Positional-
Argument registry_path); dafuer wird die erste der uebergebenen Registries
verwendet. analyze_quality_by_size.py, analyze_mean_rankings.py und
analyze_runtime.py akzeptieren mehrere Registries und bekommen daher immer
die volle Liste durchgereicht.

Nutzung:
  python analysis/analyze_all.py results/approximations/ex07 --algo-col operator \
      --all-instances --note "Nach Umstellung auf USGS-F/USGS-PRE"

  # Groessen-/Rankings-Vergleich zusaetzlich ueber eine zweite Registry
  python analysis/analyze_all.py results/approximations/ex07 \
      results/approximations/ex05:crossover \
      --algo-col operator --all-instances
'''

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

from analyze_hypervolume import get_git_info
from analyze_quality_by_size import parse_registry_arg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = ['analyze_hypervolume', 'analyze_pf_and_convergence', 'analyze_quality_by_size',
         'analyze_quality_by_class', 'analyze_mean_rankings', 'analyze_runtime']


def script_path(name):
    return os.path.join(SCRIPT_DIR, name)


def run_step(step_num, label, cmd):
    print(f"\n{'=' * 70}\n{step_num}/{len(STEPS)} {label}\n{'=' * 70}")
    print(' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"-- {label} FEHLGESCHLAGEN (exit code {result.returncode}) --")
        return False
    return True


def collect_diagrams(root):
    return sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, '**', '*.png'), recursive = True))


def archive_combined(archive_dir, label, note, out_root, steps_status):
    commit, subject, diffstat = get_git_info()
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H%M%S')
    commit_label = f"{commit}-dirty" if diffstat else (commit or "no-git")

    archive_root = os.path.join(archive_dir, 'full_report')
    entry_dir = os.path.join(archive_root, f"{timestamp}_{commit_label}_{label}")
    if os.path.isdir(out_root):
        shutil.copytree(out_root, entry_dir, dirs_exist_ok = True)
    else:
        os.makedirs(entry_dir, exist_ok = True)

    diagram_paths = collect_diagrams(entry_dir)

    with open(os.path.join(entry_dir, 'summary.md'), 'w') as f:
        f.write(f"# Gesamt-Analyse: {label}\n\n")
        f.write(f"- Datum: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (" -- **uncommitted changes!**\n" if diffstat else "\n"))
        f.write("\n## Teilanalysen\n\n")
        for name in STEPS:
            f.write(f"- {name}: {'ok' if steps_status.get(name) else 'FEHLER'}\n")
        if note:
            f.write(f"\n## Notiz\n\n{note}\n")
        f.write("\n## Diagramme\n\n")
        for p in diagram_paths:
            f.write(f"- `{p}`\n")

    index_path = os.path.join(archive_root, 'CHANGELOG.md')
    is_new = not os.path.exists(index_path)
    with open(index_path, 'a') as f:
        if is_new:
            f.write("# Gesamt-Analyse -- Verlauf\n\n")
            f.write("| Datum | Commit | Label | Teilanalysen | Notiz |\n|---|---|---|---|---|\n")
        rel_dir = os.path.relpath(entry_dir, archive_root)
        status_str = ', '.join(f"{name}={'ok' if steps_status.get(name) else 'FEHLER'}" for name in STEPS)
        f.write(f"| [{timestamp}]({rel_dir}/summary.md) | {commit_label} | {label} | {status_str} | {note or ''} |\n")

    return entry_dir, diagram_paths


def main():
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('registries', nargs = '+',
                         help = 'Ein oder mehrere Registry-Pfade, optional mit ":SPALTE(N)" '
                                'ueberschrieben (siehe analyze_quality_by_size.py). Fuer '
                                'analyze_hypervolume.py/analyze_pf_and_convergence.py (nur EINE '
                                'Registry moeglich) wird die erste verwendet.')
    parser.add_argument('--algo-col', default = 'operator',
                         help = "Default-Algorithmus-Spalte(n) (komma-getrennt) fuer Registries "
                                "ohne eigene ':SPALTE'-Angabe (default: 'operator')")
    parser.add_argument('--instance', default = None,
                         help = 'Instanz fuer HV (Einzel-Modus) und PF+Konvergenz; default: erste '
                                'in design.csv der ersten Registry gefundene Instanz')
    parser.add_argument('--all-instances', action = 'store_true',
                         help = 'HV-Analyse gepoolt ueber alle Instanzen der ersten Registry statt '
                                'Einzel-Instanz (PF+Konvergenz bleibt immer Einzel-Instanz)')
    parser.add_argument('--pool-instances', action = 'store_true',
                         help = 'quality_by_size gepoolt ueber alle Instanzen je Groesse statt '
                                'einer repraesentativen Instanz')
    parser.add_argument('--budget-pcts', default = '10,100',
                         help = 'Komma-getrennte Budget-Prozentsaetze fuer analyze_mean_rankings.py '
                                '(default: 10,100)')
    parser.add_argument('--out', default = 'figures/full_report',
                         help = 'Basisverzeichnis fuer den "aktueller Stand"-Blick auf alle vier '
                                'Diagramm-Saetze (default: figures/full_report); wird bei jedem '
                                'Lauf ueberschrieben')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out und im Archiv-Eintrag (default: _all)')
    parser.add_argument('--archive-dir', default = 'analysis_log',
                         help = 'Basisverzeichnis des Verlaufs-Archivs (default: analysis_log)')
    parser.add_argument('--no-archive', action = 'store_true',
                         help = 'Verlaufs-Archivierung fuer diesen Lauf abschalten (z.B. fuer Testlaeufe)')
    parser.add_argument('--note', default = None,
                         help = 'Kurze Notiz, landet in summary.md und CHANGELOG.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    primary_path, primary_cols = parse_registry_arg(args.registries[0], default_algo_cols)
    primary_algo_col = ','.join(primary_cols)

    out_root = os.path.join(args.out, args.label)
    steps_status = {}

    hv_cmd = [sys.executable, script_path('analyze_hypervolume.py'), primary_path,
              '--algo-col', primary_algo_col, '--out', os.path.join(out_root, 'hv_analysis'),
              '--no-archive']
    if args.all_instances:
        hv_cmd.append('--all-instances')
    elif args.instance:
        hv_cmd += ['--instance', args.instance]
    steps_status['analyze_hypervolume'] = run_step(1, 'analyze_hypervolume.py', hv_cmd)

    pf_cmd = [sys.executable, script_path('analyze_pf_and_convergence.py'), primary_path,
              '--algo-col', primary_algo_col, '--out', os.path.join(out_root, 'pf_and_convergence'),
              '--no-archive']
    if args.instance:
        pf_cmd += ['--instance', args.instance]
    steps_status['analyze_pf_and_convergence'] = run_step(2, 'analyze_pf_and_convergence.py', pf_cmd)

    # analyze_quality_by_size.py, analyze_mean_rankings.py und analyze_runtime.py haben (noch)
    # keine eigene --no-archive/--archive-dir-Option -- sie archivieren nie selbst, daher hier weglassen
    size_cmd = [sys.executable, script_path('analyze_quality_by_size.py'), *args.registries,
                '--algo-col', args.algo_col, '--out', os.path.join(out_root, 'quality_by_size')]
    if args.pool_instances:
        size_cmd.append('--pool-instances')
    steps_status['analyze_quality_by_size'] = run_step(3, 'analyze_quality_by_size.py (--facet-by size)', size_cmd)

    # zusaetzlich nach Instanzklasse facettiert: bei Registries wie ex07, wo alle Instanzen
    # dieselbe Groesse haben, ist das die aussagekraeftigere Ansicht (rohe HV, nie ueber
    # Instanzen gepoolt, siehe analyze_quality_by_size.py's Docstring)
    class_cmd = [sys.executable, script_path('analyze_quality_by_size.py'), *args.registries,
                 '--algo-col', args.algo_col, '--facet-by', 'class',
                 '--out', os.path.join(out_root, 'quality_by_class')]
    if args.pool_instances:
        class_cmd.append('--pool-instances')
    steps_status['analyze_quality_by_class'] = run_step(4, 'analyze_quality_by_size.py (--facet-by class)', class_cmd)

    rankings_cmd = [sys.executable, script_path('analyze_mean_rankings.py'), *args.registries,
                     '--algo-col', args.algo_col, '--budget-pcts', args.budget_pcts,
                     '--out', os.path.join(out_root, 'mean_rankings')]
    steps_status['analyze_mean_rankings'] = run_step(5, 'analyze_mean_rankings.py', rankings_cmd)

    runtime_cmd = [sys.executable, script_path('analyze_runtime.py'), *args.registries,
                    '--algo-col', args.algo_col, '--out', os.path.join(out_root, 'runtime')]
    steps_status['analyze_runtime'] = run_step(6, 'analyze_runtime.py', runtime_cmd)

    print(f"\n{'=' * 70}\nZusammenfassung\n{'=' * 70}")
    for name in STEPS:
        print(f"  {name}: {'ok' if steps_status.get(name) else 'FEHLER'}")

    if not any(steps_status.values()):
        raise SystemExit(f'\nAlle {len(STEPS)} Teilanalysen sind fehlgeschlagen -- nichts zum Archivieren.')

    if not args.no_archive:
        entry_dir, diagram_paths = archive_combined(args.archive_dir, args.label, args.note, out_root, steps_status)
        print(f"\nArchiviert in: {entry_dir}")
        print(f"Verlaufs-Uebersicht: {os.path.join(args.archive_dir, 'full_report', 'CHANGELOG.md')}")
        print(f"\n{len(diagram_paths)} Diagramme:")
        for p in diagram_paths:
            print(f"  {os.path.join(entry_dir, p)}")
    else:
        diagram_paths = collect_diagrams(out_root)
        print(f"\n{len(diagram_paths)} Diagramme (nicht archiviert) unter {out_root}:")
        for p in diagram_paths:
            print(f"  {p}")


if __name__ == '__main__':
    main()
