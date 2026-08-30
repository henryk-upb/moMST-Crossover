#!/usr/bin/env python
'''
Verlauf der (mittleren/medianen) Guete-Indikatoren ueber die Zeit (% des
Generationen-Budgets), gepoolt ueber R Wiederholungen UND K Instanzen --
direktes Pendant zu plot_progress() in analyze_hypervolume.py (dort nur HV,
nur K=1 Instanz), hier verallgemeinert auf HV/Epsilon/Delta_p und beliebig
viele Instanzen.

Hintergrund: fuer jeden Lauf werden nicht nur die finale Pareto-Front-
Approximation (pf.csv), sondern T Zwischenstaende bei 0%, 10%, ..., 100% des
Generationen-Budgets ausgewertet (trajectory.csv, siehe log_every in
EA/Algorithms.py). Reduziert man jeden dieser T x R x K Zwischenstaende auf
einen skalaren Indikatorwert, erhaelt man fuer jeden Algorithmus eine
T x R x K grosse Wertematrix; deren Entwicklung ueber die Zeit (T) wird hier
als Linienplot dargestellt (analog zur ECJ-2023-Studie).

Rohe Indikatorwerte sind nur INNERHALB einer Instanz direkt vergleichbar
(Skala haengt von Instanz/n ab, siehe analyze_hypervolume.py's
compute_cross_instance_table-Docstring) -- ueber mehrere Instanzen hinweg
wird daher standardmaessig wie in analyze_mean_rankings.py per (Instanz,
Repl)-Block gerankt (1 = bester Wert) und der RANG statt des Rohwerts
gemittelt/medianisiert. --raw-value zeigt stattdessen den Rohwert direkt --
nur sinnvoll, wenn alle betrachteten Instanzen dieselbe Skala haben (z.B.
genau eine Instanz, wie bislang bei hv_progress.png).

Nutzung:
  python analysis/analyze_indicator_progress.py results/approximations/ex07 --algo-col operator

  # feineres Zeitraster, Median statt Mittelwert
  python analysis/analyze_indicator_progress.py results/approximations/ex07 --algo-col operator \
      --budget-pcts 10,20,30,40,50,60,70,80,90,100 --stat median

  # rohe HV-Werte statt Rang (nur sinnvoll bei einer einzelnen Instanz)
  python analysis/analyze_indicator_progress.py results/approximations/ex05:crossover \
      --raw-value
'''

import argparse
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from analyze_hypervolume import get_git_info
from analyze_mean_rankings import HIGHER_IS_BETTER, compute_indicator_table
from analyze_quality_by_size import parse_registry_arg

INDICATOR_ORDER = ['HV', 'Epsilon', 'Delta_p']


def compute_rank_progress(df):
    '''Rang je (instance, repl, budget_pct, indicator)-Block (1 = bester Wert)
    -- Algorithmen werden also nur gegen andere Algorithmen auf DERSELBEN
    Budget-Stufe verglichen, nie ueber Zeitpunkte hinweg gemischt (sonst
    wuerden fruehe, notwendigerweise niedrigere Rohwerte systematisch
    schlechter gerankt als spaete, unabhaengig vom Algorithmus) -- dann
    gepoolt ueber ALLE Instanzen/Wiederholungen aggregiert je (algorithm,
    budget_pct, indicator). Long-format Rueckgabe mit Spalten algorithm,
    budget_pct, indicator, mean, median, q25, q75, n.'''
    ranked = []
    for indicator, higher_is_better in HIGHER_IS_BETTER.items():
        sub = df[df['indicator'] == indicator].copy()
        sub['rank'] = sub.groupby(['instance', 'repl', 'budget_pct'])['value'] \
            .rank(ascending = not higher_is_better, method = 'average')
        ranked.append(sub)
    df_ranked = pd.concat(ranked, ignore_index = True)

    agg = df_ranked.groupby(['algorithm', 'budget_pct', 'indicator'])['rank'].agg(
        mean = 'mean', median = 'median',
        q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75), n = 'size',
    ).reset_index()
    return agg, df_ranked


def compute_value_progress(df):
    '''Rohwert direkt aggregiert je (algorithm, budget_pct, indicator),
    gepoolt ueber Wiederholungen (und Instanzen, falls mehrere -- nur
    sinnvoll bei gleicher Instanz/Skala, siehe Modul-Docstring).'''
    agg = df.groupby(['algorithm', 'budget_pct', 'indicator'])['value'].agg(
        mean = 'mean', median = 'median',
        q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75), n = 'size',
    ).reset_index()
    return agg


def plot_indicator_progress(agg, out_dir, mode, stat):
    '''Ein Panel je Indikator: x-Achse = % des Budgets (Zeit), y-Achse =
    mean/median (je nach stat) ueber R x K Wiederholungen/Instanzen, eine
    Linie je Algorithmus, mit einem q25-q75-Band. mode='rank' plottet den
    ueber alle Instanzen gepoolten Rang (niedriger = besser, y-Achse
    invertiert); mode='value' plottet den rohen Indikatorwert.'''
    os.makedirs(out_dir, exist_ok = True)
    indicators = [i for i in INDICATOR_ORDER if i in agg['indicator'].unique()]
    algorithms = sorted(agg['algorithm'].unique())
    n = len(algorithms)

    colors = dict(zip(algorithms, matplotlib.colormaps[
        'tab10' if n <= 10 else 'tab20'](range(n))))

    fig, axes = plt.subplots(1, len(indicators), figsize = (6 * len(indicators), 5), sharex = True)
    if len(indicators) == 1:
        axes = [axes]

    for ax, indicator in zip(axes, indicators):
        sub = agg[agg['indicator'] == indicator]
        for algo in algorithms:
            line = sub[sub['algorithm'] == algo].sort_values('budget_pct')
            if line.empty:
                continue
            plotted, = ax.plot(line['budget_pct'], line[stat], marker = 'o', color = colors[algo], label = algo)
            ax.fill_between(line['budget_pct'], line['q25'], line['q75'], color = plotted.get_color(), alpha = 0.15)
        ax.set_xlabel('% of Maximum Generations/Evaluations')
        if mode == 'rank':
            ax.set_ylabel(f'{stat.capitalize()} Rank (lower = better)')
            ax.invert_yaxis()
        else:
            better = 'higher' if HIGHER_IS_BETTER.get(indicator, True) else 'lower'
            ax.set_ylabel(f'{stat.capitalize()} {indicator} ({better} = better)')
        ax.set_title(indicator)

    handles, labels = axes[-1].get_legend_handles_labels()
    if n > 8:
        fig.legend(handles, labels, loc = 'center left', bbox_to_anchor = (1.0, 0.5), fontsize = 8, ncol = 1 + n // 25)
        fig.tight_layout(rect = (0, 0, 0.86, 0.94))
    else:
        axes[-1].legend()
        fig.tight_layout(rect = (0, 0, 1, 0.94))
    title_mode = 'Rank (pooled over all instances)' if mode == 'rank' else 'Raw Value'
    fig.suptitle(f'Indicator Progress over Budget -- {title_mode}, {stat}')

    out_path = os.path.join(out_dir, 'indicator_progress.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def compute_value_progress_by_class(df):
    '''Wie compute_value_progress, aber zusaetzlich nach Instanzklasse gruppiert
    (ein Aggregat je (class, algorithm, budget_pct, indicator)) -- fuer den
    Facet-Grid-Plot mit einem Panel je Klasse (siehe plot_indicator_progress_by_class).'''
    agg = df.groupby(['class', 'algorithm', 'budget_pct', 'indicator'])['value'].agg(
        mean = 'mean', median = 'median',
        q25 = lambda s: s.quantile(0.25), q75 = lambda s: s.quantile(0.75), n = 'size',
    ).reset_index()
    return agg


def plot_indicator_progress_by_class(agg, out_dir, stat, indicator):
    '''Facet-Grid-Variante von plot_indicator_progress: EIN Bild mit einem
    Panel je Instanzklasse (2x2 bei 4 Klassen) fuer GENAU EINEN Indikator
    (typischerweise HV, da rohe Werte nur innerhalb einer Instanz/Klasse
    vergleichbar sind), statt separater Bilder je Klasse.'''
    os.makedirs(out_dir, exist_ok = True)
    classes = sorted(agg['class'].unique())
    algorithms = sorted(agg['algorithm'].unique())
    n = len(algorithms)
    colors = dict(zip(algorithms, matplotlib.colormaps['tab10' if n <= 10 else 'tab20'](range(n))))

    ncols = math.ceil(math.sqrt(len(classes)))
    nrows = math.ceil(len(classes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze = False, figsize = (6 * ncols, 4.3 * nrows))
    axes_flat = axes.flatten()

    for ax, cls in zip(axes_flat, classes):
        sub = agg[agg['class'] == cls]
        for algo in algorithms:
            line = sub[sub['algorithm'] == algo].sort_values('budget_pct')
            if line.empty:
                continue
            plotted, = ax.plot(line['budget_pct'], line[stat], marker = 'o', color = colors[algo], label = algo)
            ax.fill_between(line['budget_pct'], line['q25'], line['q75'], color = plotted.get_color(), alpha = 0.15)
        ax.set_xlabel('% of Max. Gen./Eval.')
        better = 'higher' if HIGHER_IS_BETTER.get(indicator, True) else 'lower'
        ax.set_ylabel(f'{stat.capitalize()} {indicator} ({better} = better)')
        ax.set_title(f'Class {cls}', fontsize = 9)
        ax.legend(fontsize = 7)
    for ax in axes_flat[len(classes):]:
        ax.axis('off')

    fig.suptitle(f'{indicator} Progress over Budget -- per Instance Class ({stat})')
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    out_path = os.path.join(out_dir, 'indicator_progress.png')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    return out_path


def write_run_info(out_dir, note, registries, budget_pcts, mode, stat):
    commit, subject, diffstat = get_git_info()
    with open(os.path.join(out_dir, 'run_info.md'), 'w') as f:
        f.write('# Indicator Progress -- Lauf-Info\n\n')
        f.write(f"- Git-Commit: `{commit or 'unbekannt'}` ({subject})"
                + (' -- **uncommitted changes!**\n' if diffstat else '\n'))
        f.write(f'- Budget-Prozentsaetze (T): {budget_pcts}\n')
        mode_desc = 'Rang, gepoolt ueber alle Instanzen (R x K)' if mode == 'rank' else 'Rohwert'
        f.write(f'- Modus: {mode_desc}; Statistik: {stat}\n')
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
    parser.add_argument('--budget-pcts', default = '10,20,30,40,50,60,70,80,90,100',
                         help = 'Komma-getrennte Budget-Prozentsaetze T (default: 10,20,...,100, '
                                'passend zum log_every=10-Raster in EA/Algorithms.py)')
    parser.add_argument('--stat', choices = ['mean', 'median'], default = 'mean',
                         help = 'Zentrale Linie je Snapshot: Mittelwert oder Median (default: mean)')
    parser.add_argument('--raw-value', action = 'store_true',
                         help = 'Rohen Indikatorwert statt gepooltem Rang plotten -- nur sinnvoll, '
                                'wenn alle betrachteten Instanzen dieselbe Skala haben (z.B. genau '
                                'eine Instanz). Default: Rang (Skalen-unabhaengig, beliebig viele '
                                'Instanzen).')
    parser.add_argument('--instances', default = None,
                         help = 'Komma-getrennte Liste von Instanzpfaden (wie in design.csv); nur '
                                'diese werden ausgewertet statt aller in der Registry gefundenen')
    parser.add_argument('--facet-by-class', action = 'store_true',
                         help = 'Statt EINEM Bild (gepoolter Rang oder Rohwert) wird EIN Bild mit '
                                'einem Panel je Instanzklasse erzeugt (2x2 bei 4 Klassen), rohe Werte '
                                'je Klasse -- gedacht fuer --instances mit genau einer Instanz je '
                                'Klasse. Erzwingt --raw-value und genau EINEN Indikator (--indicators).')
    parser.add_argument('--indicators', default = 'HV',
                         help = "Komma-getrennte Teilmenge von 'HV,Epsilon,Delta_p' -- nur diese "
                                'werden berechnet/geplottet (default: nur HV)')
    parser.add_argument('--out', default = 'figures/indicator_progress',
                         help = 'Ausgabeverzeichnis (default: figures/indicator_progress)')
    parser.add_argument('--label', default = '_all',
                         help = 'Unterordner-Name unter --out (default: _all)')
    parser.add_argument('--note', default = None, help = 'Kurze Notiz, landet in run_info.md')
    args = parser.parse_args()

    default_algo_cols = args.algo_col.split(',')
    registries = [parse_registry_arg(r, default_algo_cols) for r in args.registries]
    budget_pcts = sorted({int(p) for p in args.budget_pcts.split(',')})
    instances_filter = set(args.instances.split(',')) if args.instances else None
    indicators_wanted = [i.strip() for i in args.indicators.split(',')]

    print(f'Registries: {registries}')
    print(f'Budget-Prozentsaetze (T): {budget_pcts}')
    print(f'Indikatoren: {indicators_wanted}')
    if instances_filter:
        print(f'Instanz-Filter: {sorted(instances_filter)}')

    df = compute_indicator_table(registries, budget_pcts, instances_filter, indicators_wanted)
    if df.empty:
        raise SystemExit('Keine Daten gefunden -- Registries pruefen.')
    print(f'Instanzen (K): {sorted(df["instance"].unique())}')
    print(f'Algorithmen: {sorted(df["algorithm"].unique())}')

    if args.facet_by_class:
        if len(indicators_wanted) != 1:
            raise SystemExit('--facet-by-class erwartet genau EINEN Indikator via --indicators.')
        agg = compute_value_progress_by_class(df)
        out_dir = os.path.join(args.out, args.label)
        out_path = plot_indicator_progress_by_class(agg, out_dir, args.stat, indicators_wanted[0])
        agg_csv = os.path.join(out_dir, 'indicator_progress.csv')
        agg.sort_values(['indicator', 'class', 'algorithm', 'budget_pct']).to_csv(agg_csv, index = False)
        write_run_info(out_dir, args.note, registries, budget_pcts, 'value (je Klasse)', args.stat)
        print(f'\nPlot gespeichert in: {out_path}')
        print(f'Aggregierte Werte gespeichert in: {agg_csv}')
        return

    mode = 'value' if args.raw_value else 'rank'
    if mode == 'value' and df['instance'].nunique() > 1:
        print(f"WARNUNG: --raw-value mit {df['instance'].nunique()} Instanzen unterschiedlicher "
              "Skala gepoolt -- die Kurven vermischen Instanz- und Algorithmus-Unterschiede. "
              "Fuer eine belastbare Mehr-Instanzen-Sicht --raw-value weglassen (Rang-Modus).")

    if mode == 'value':
        agg = compute_value_progress(df)
    else:
        agg, _ = compute_rank_progress(df)

    out_dir = os.path.join(args.out, args.label)
    out_path = plot_indicator_progress(agg, out_dir, mode, args.stat)

    agg_csv = os.path.join(out_dir, 'indicator_progress.csv')
    agg.sort_values(['indicator', 'algorithm', 'budget_pct']).to_csv(agg_csv, index = False)
    write_run_info(out_dir, args.note, registries, budget_pcts, mode, args.stat)

    print(f'\nPlot gespeichert in: {out_path}')
    print(f'Aggregierte Werte gespeichert in: {agg_csv}')


if __name__ == '__main__':
    main()
