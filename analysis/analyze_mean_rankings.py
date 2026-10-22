#!/usr/bin/env python
'''

  Drei Indikatoren je Punktwolke:
    * HV (hoeher = besser) -- wie analyze_hypervolume.py, Referenzpunkt wie dort.
    * additiver Epsilon-Indikator (niedriger = besser) -- pymoo Epsilon(pf=Referenzmenge).
    * Delta_p (niedriger = besser) -- vereinfachte, in der EMO-Literatur uebliche
      Variante max(GD, IGD) mit euklidischem Abstand (pymoo GD/IGD, jeweils
      Mittelwert der naechsten Distanz statt Power-Mittelung; siehe Schuetze
      et al. 2012 fuer die vollstaendige Definition mit p-Norm).

  Je (Instanz, Repl)-Block werden die Konfigurationen (Algorithmus x Budget%)
  separat je Indikator gerankt (Rang 1 = bester Wert; HV absteigend sortiert,
  Epsilon/Delta_p aufsteigend), dann je Instanzklasse ueber alle zugehoerigen
  Bloecke gemittelt.

Nutzung:
  python analysis/analyze_mean_rankings.py results/approximations/ex07 --algo-col operator

  # nur finale Front (kein Budget-Vergleich), zwei Registries kombiniert
  python analysis/analyze_mean_rankings.py results/approximations/ex07 \
      results/approximations/ex05:crossover \
      --algo-col operator --budget-pcts 100
'''

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.epsilon import Epsilon
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from analyze_hypervolume import (
    read_params, collect_fronts, collect_trajectories, determine_reference_point, get_git_info,
)
from analyze_quality_by_size import infer_instance_class, parse_registry_arg

ND_SORTER = NonDominatedSorting()


def non_dominated_subset(points):
    '''Nicht-dominierte Teilmenge einer Punktwolke (Minimierung, 2D+).'''
    arr = np.asarray(points, dtype = float)
    if len(arr) == 0:
        return arr
    front_idx = ND_SORTER.do(arr, only_non_dominated_front = True)
    return arr[front_idx]


def build_reference_set(by_algo):
    '''Nicht-dominierte Vereinigung aller finalen Fronten (alle Algorithmen/
    Wiederholungen) einer Instanz -- Referenzmenge fuer Epsilon und Delta_p.'''
    all_points = [p for repls in by_algo.values() for _, points in repls for p in points]
    return non_dominated_subset(all_points)


def max_generation(by_algo_traj):
    return max(gen for repls in by_algo_traj.values() for _, gens in repls for gen in gens)


def snapshot_points_at_budget(by_algo, by_algo_traj, budget_pct):
    '''algo -> [(repl, points), ...] bei gegebenem Budget-Anteil. 100% -> finale
    Front (pf.csv); sonst die Populations-Momentaufnahme aus trajectory.csv bei
    der Generation, deren Budget-Anteil dem Zielwert am naechsten liegt,
    reduziert auf ihre nicht-dominierte Teilmenge.'''
    if budget_pct >= 100:
        return {algo: [(repl, np.asarray(points, dtype = float)) for repl, points in repls]
                for algo, repls in by_algo.items()}

    max_gen = max_generation(by_algo_traj)
    target_gen = round(budget_pct / 100.0 * max_gen) if max_gen > 0 else 0
    result = {}
    for algo, repls in by_algo_traj.items():
        out = []
        for repl, gens in repls:
            nearest_gen = min(gens.keys(), key = lambda g: abs(g - target_gen))
            out.append((repl, non_dominated_subset(gens[nearest_gen])))
        result[algo] = out
    return result


ALL_INDICATORS = ('HV', 'Epsilon', 'Delta_p')


def compute_indicator_table(registries, budget_pcts, instances_filter = None, indicators = ALL_INDICATORS):
    '''Long-format DataFrame mit Spalten instance, class, repl, config, algorithm,
    budget_pct, indicator, value -- eine Zeile je (Instanz, Repl, Konfiguration,
    Indikator). instances_filter (optional): nur diese Instanzpfade beruecksichtigen
    (z.B. eine zufaellig gewaehlte Instanz je Klasse) statt aller in design.csv
    gefundenen. indicators (optional): nur diese Indikatoren berechnen (Teilmenge
    von HV/Epsilon/Delta_p) -- spart bei grossen Instanzen auch Rechenzeit, da
    GD/IGD/Epsilon dann gar nicht erst ausgewertet werden.'''
    records = []
    for registry_path, algo_cols in registries:
        rows = read_params(os.path.join(registry_path, 'design.csv'))
        instances = sorted(set(row['instance'] for row in rows))
        if instances_filter is not None:
            instances = [i for i in instances if i in instances_filter]
        for instance in instances:
            instance_class = infer_instance_class(instance)
            by_algo = collect_fronts(rows, registry_path, instance, algo_cols)
            if not by_algo:
                continue
            by_algo_traj = collect_trajectories(rows, registry_path, instance, algo_cols)

            need_hv = 'HV' in indicators
            need_eps_or_dp = 'Epsilon' in indicators or 'Delta_p' in indicators
            if need_hv:
                hv_ref = determine_reference_point(by_algo, by_algo_traj)
                hv_indicator = HV(ref_point = np.array(hv_ref))
            if need_eps_or_dp:
                eps_ref = build_reference_set(by_algo)
                gd_indicator = GD(pf = eps_ref)
                igd_indicator = IGD(pf = eps_ref)
                eps_indicator = Epsilon(pf = eps_ref)

            max_gen = max_generation(by_algo_traj)
            for budget_pct in budget_pcts:
                snapshots = snapshot_points_at_budget(by_algo, by_algo_traj, budget_pct)
                gen = max_gen if budget_pct >= 100 else round(budget_pct / 100.0 * max_gen)
                for algo, repls in snapshots.items():
                    config = f'{algo} ({budget_pct}%)'
                    for repl, points in repls:
                        if len(points) == 0:
                            continue
                        values = {}
                        if need_hv:
                            values['HV'] = hv_indicator(points)
                        if 'Epsilon' in indicators:
                            values['Epsilon'] = eps_indicator(points)
                        if 'Delta_p' in indicators:
                            values['Delta_p'] = max(gd_indicator(points), igd_indicator(points))
                        for indicator_name, value in values.items():
                            records.append({
                                'instance': instance, 'class': instance_class, 'repl': int(repl),
                                'config': config, 'algorithm': algo, 'budget_pct': budget_pct, 'gen': gen,
                                'indicator': indicator_name, 'value': value,
                            })

    return pd.DataFrame.from_records(records)


HIGHER_IS_BETTER = {'HV': True, 'Epsilon': False, 'Delta_p': False}


def compute_mean_rankings(df):
    '''Rang je (instance, repl, indicator)-Block (1 = bester Wert), dann Mittelwert
    je (class, config, indicator). Returns long-format DataFrame mit Spalten
    class, config, algorithm, budget_pct, indicator, mean_rank, n_blocks.'''
    ranked = []
    for indicator, higher_is_better in HIGHER_IS_BETTER.items():
        sub = df[df['indicator'] == indicator].copy()
        sub['rank'] = sub.groupby(['instance', 'repl'])['value'] \
            .rank(ascending = not higher_is_better, method = 'average')
        ranked.append(sub)
    df_ranked = pd.concat(ranked, ignore_index = True)

    agg = df_ranked.groupby(['class', 'config', 'algorithm', 'budget_pct', 'indicator']).agg(
        mean_rank = ('rank', 'mean'), n_blocks = ('rank', 'size')).reset_index()
    return agg, df_ranked


def plot_mean_rankings(agg, out_dir):
    '''Ein Panel je Indikator (HV, Epsilon, Delta_p): x-Achse = Instanzklasse,
    y-Achse = mittlerer Rang, eine Linie je Konfiguration -- direktes Analogon
    zu Fig. 9 (Bossek & Grimme, 2023). Gleiche Farbe je Algorithmus ueber alle
    drei Panels; Budget-Prozentsatz per Linienstil (durchgezogen = 100%,
    gestrichelt/gepunktet = niedrigere Werte) unterschieden.'''
    os.makedirs(out_dir, exist_ok = True)
    indicators = [i for i in ('HV', 'Epsilon', 'Delta_p') if i in agg['indicator'].unique()]
    classes = sorted(agg['class'].unique())
    algorithms = sorted(agg['algorithm'].unique())
    budget_pcts = sorted(agg['budget_pct'].unique(), reverse = True)

    colors = dict(zip(algorithms, matplotlib.colormaps[
        'tab10' if len(algorithms) <= 10 else 'tab20'](range(len(algorithms)))))
    linestyles = ['-', '--', ':', '-.']
    style_for_pct = {pct: linestyles[i % len(linestyles)] for i, pct in enumerate(budget_pcts)}

    fig, axes = plt.subplots(1, len(indicators), figsize = (6 * len(indicators), 5), sharex = True)
    if len(indicators) == 1:
        axes = [axes]

    for ax, indicator in zip(axes, indicators):
        sub = agg[agg['indicator'] == indicator]
        for algo in algorithms:
            for pct in budget_pcts:
                line = sub[(sub['algorithm'] == algo) & (sub['budget_pct'] == pct)]
                if line.empty:
                    continue
                line = line.set_index('class').reindex(classes)
                label = f'{algo} ({pct}%)' if len(budget_pcts) > 1 else algo
                ax.plot(classes, line['mean_rank'], marker = 'o', color = colors[algo],
                         linestyle = style_for_pct[pct], label = label)
        ax.set_xlabel('Instance Class')
        ax.set_ylabel('Mean Rank (lower = better)')
        ax.set_title(indicator)
        ax.invert_yaxis()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'center left', bbox_to_anchor = (1.0, 0.5),
               fontsize = 8, ncol = 1 + len(handles) // 25)
    fig.suptitle('Mean Rankings per Indicator, Split by Instance Class')
    fig.tight_layout(rect = (0, 0, 0.86, 0.96))

    out_path = os.path.join(out_dir, 'mean_rankings.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def write_run_info(out_dir, note, registries, budget_pcts, indicators = None, instances_filter = None):
    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write('# Mean Rankings -- Lauf-Info\n\n')
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (' -- **uncommitted changes!**\n' if diffstat else '\n'))
        f.write(f'- Budget-Prozentsaetze: {budget_pcts}\n')
        if indicators:
            f.write(f'- Indikatoren: {indicators}\n')
        if instances_filter:
            f.write(f'- Instanz-Filter: {sorted(instances_filter)}\n')
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
    parser.add_argument('--budget-pcts', default = '10,100',
                         help = 'Komma-getrennte Budget-Prozentsaetze, je einer erzeugt eine eigene '
                                'Linie pro Algorithmus (default: 10,100, wie im 2023er-Paper)')
    parser.add_argument('--instances', default = None,
                         help = 'Komma-getrennte Liste von Instanzpfaden (wie in design.csv); nur '
                                'diese werden ausgewertet statt aller in der Registry gefundenen '
                                '(z.B. eine zufaellig gewaehlte Instanz je Klasse statt ueber alle '
                                'zu poolen)')
    parser.add_argument('--indicators', default = 'HV',
                         help = "Komma-getrennte Teilmenge von 'HV,Epsilon,Delta_p' -- nur diese "
                                'werden berechnet/geplottet (default: nur HV)')
    parser.add_argument('--out', default = 'figures/mean_rankings',
                         help = 'Ausgabeverzeichnis (default: figures/mean_rankings)')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out (default: _all)')
    parser.add_argument('--note', default = None, help = 'Kurze Notiz, landet in run_info.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    registries = [parse_registry_arg(r, default_algo_cols) for r in args.registries]
    budget_pcts = sorted({int(p) for p in args.budget_pcts.split(',')})
    instances_filter = set(args.instances.split(',')) if args.instances else None
    indicators = [i.strip() for i in args.indicators.split(',')]

    print(f'Registries: {registries}')
    print(f'Budget-Prozentsaetze: {budget_pcts}')
    print(f'Indikatoren: {indicators}')
    if instances_filter:
        print(f'Instanz-Filter: {sorted(instances_filter)}')

    df = compute_indicator_table(registries, budget_pcts, instances_filter, indicators)
    if df.empty:
        raise SystemExit('Keine Daten gefunden -- Registries pruefen.')
    print(f'Instanzen: {sorted(df["instance"].unique())}')
    print(f'Klassen: {sorted(df["class"].unique())}')
    print(f'Algorithmen: {sorted(df["algorithm"].unique())}')

    agg, df_ranked = compute_mean_rankings(df)

    out_dir = os.path.join(args.out, args.label)
    out_path = plot_mean_rankings(agg, out_dir)

    ranked_csv = os.path.join(out_dir, 'ranked_values.csv')
    df_ranked.sort_values(['class', 'instance', 'repl', 'indicator', 'config']) \
        .to_csv(ranked_csv, index = False)
    agg_csv = os.path.join(out_dir, 'mean_rankings.csv')
    agg.sort_values(['indicator', 'class', 'mean_rank']).to_csv(agg_csv, index = False)
    write_run_info(out_dir, args.note, registries, budget_pcts, indicators, instances_filter)

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Mittlere Raenge gespeichert in: {agg_csv}')
    print(f'Rohe Rang-Werte gespeichert in: {ranked_csv}')


if __name__ == '__main__':
    main()
