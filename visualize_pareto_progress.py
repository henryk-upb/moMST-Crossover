#!/usr/bin/env python
'''
Visualisiert die Annaeherung einer NSGA2-Population an die Paretofront ueber
die Generationen hinweg.

Erwartet die Ausgaben von EA/Algorithms.py::NSGA2.write_result():
  <run_dir>/trajectory.csv  mit Spalten gen,c1,c2  (Population alle log_every Generationen)
  <run_dir>/pf.csv          mit Spalten c1,c2      (finale Approximation der Paretofront)

Nutzung:
  python visualize_pareto_progress.py results/<run_dir> --out figures/pareto_progress
  python visualize_pareto_progress.py results/<run_dir> --out figures/pareto_progress --gif
'''

import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_trajectory(path):
    gens = {}
    with open(path, newline = '') as f:
        for row in csv.DictReader(f):
            gen = int(row['gen'])
            gens.setdefault(gen, []).append((float(row['c1']), float(row['c2'])))
    return gens


def read_pareto_front(path):
    with open(path, newline = '') as f:
        return [(float(row['c1']), float(row['c2'])) for row in csv.DictReader(f)]


def plot_frames(gens, pf, out_dir):
    os.makedirs(out_dir, exist_ok = True)

    all_points = [p for pop in gens.values() for p in pop] + pf
    xs, ys = zip(*all_points)
    margin_x, margin_y = 0.02 * (max(xs) - min(xs)), 0.02 * (max(ys) - min(ys))
    xlim = (min(xs) - margin_x, max(xs) + margin_x)
    ylim = (min(ys) - margin_y, max(ys) + margin_y)

    sorted_gens = sorted(gens.keys())
    frame_paths = []

    for i, gen in enumerate(sorted_gens):
        fig, ax = plt.subplots(figsize = (6, 6))

        # bisherige Generationen als blasse Spur, um den Verlauf zu zeigen
        for prev_gen in sorted_gens[:i]:
            px, py = zip(*gens[prev_gen])
            ax.scatter(px, py, s = 10, color = 'lightgray')

        cx, cy = zip(*gens[gen])
        ax.scatter(cx, cy, s = 20, color = 'tab:blue', label = f'Generation {gen}')

        fx, fy = zip(*pf)
        ax.scatter(fx, fy, s = 25, color = 'tab:red', marker = 'x', label = 'Paretofront (final)')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        ax.set_title(f'Generation {gen}')
        ax.legend(loc = 'upper right')

        frame_path = os.path.join(out_dir, f'frame_{i:04d}.png')
        fig.savefig(frame_path, dpi = 120, bbox_inches = 'tight')
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def make_gif(frame_paths, gif_path, fps = 2):
    import imageio.v2 as imageio  # pip install imageio
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, fps = fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help = 'Verzeichnis mit trajectory.csv und pf.csv')
    parser.add_argument('--out', default = 'figures/pareto_progress', help = 'Ausgabeverzeichnis fuer die Frames')
    parser.add_argument('--gif', action = 'store_true', help = 'zusaetzlich ein animiertes GIF erzeugen (benoetigt imageio)')
    args = parser.parse_args()

    gens = read_trajectory(os.path.join(args.run_dir, 'trajectory.csv'))
    pf = read_pareto_front(os.path.join(args.run_dir, 'pf.csv'))

    frame_paths = plot_frames(gens, pf, args.out)

    if args.gif:
        make_gif(frame_paths, os.path.join(args.out, 'progress.gif'))


if __name__ == '__main__':
    main()