import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.labelsize': 10, 'axes.titlesize': 10,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 10, 'ps.fonttype': 42,
                     'axes.unicode_minus': False})


def make_figure(k, allow_supplied=False):
    provenance = json.loads((ROOT / f'example_{k}_provenance.json').read_text())
    if provenance['status'] == 'supplied-unverified':
        if not allow_supplied:
            raise ValueError('Supplied results require --allow-supplied; regenerate for a current comparison')
    else:
        settings = loadmat(ROOT / f'example_{k}_easyspin_settings.mat', simplify_cells=True)
        opt = settings['Opt']
        if not np.array_equal(np.asarray(opt['GridSize']), [80, 0]):
            raise ValueError('EasySpin GridSize must be [80 0]')
        if not np.isclose(opt['ModellingAccuracy'], 1e-7, rtol=1e-12, atol=0):
            raise ValueError('EasySpin ModellingAccuracy must be 1e-7')
        exported = loadmat(ROOT / f'example_{k}.mat', simplify_cells=True)
        for key in ('Sys', 'Exp'):
            for name, value in exported[key].items():
                if name not in settings[key] or not np.allclose(value, settings[key][name]):
                    raise ValueError(f'EasySpin input differs: {key}.{name}')
    data = np.load(ROOT / f'example_{k}_mars.npz')
    fields = data['fields'] * 1e3
    meshes, tolerances, spectra = data['meshes'], data['tolerances'], data['spectra']
    i = np.flatnonzero(meshes == 80).item()
    j = np.flatnonzero(np.isclose(tolerances, 1e-7, rtol=1e-12, atol=0)).item()
    best = spectra[i, j]
    norm = np.linalg.norm(best)
    if norm == 0 or not np.isfinite(spectra).all():
        raise ValueError('Invalid MaRs spectra')
    values = np.linalg.norm(spectra - best, axis=-1) / norm
    easyspin = np.loadtxt(ROOT / f'example_{k}_easyspin.csv', delimiter=',', ndmin=2)
    if easyspin.shape != (len(fields), 2) or not np.isfinite(easyspin).all():
        raise ValueError('Expected exactly two finite columns: B in mT and the reference spectrum')
    if not np.allclose(easyspin[:, 0], fields, rtol=1e-10, atol=1e-8):
        raise ValueError('MaRs and EasySpin field axes differ')
    reference = easyspin[:, 1]
    scale = np.dot(reference, best) / np.dot(best, best)
    amplitude = np.max(np.abs(reference))
    if amplitude == 0:
        raise ValueError('EasySpin reference is zero')
    energy = np.load(ROOT / f'example_{k}_energy.npz')
    fig = plt.figure(figsize=(160 / 25.4, 126 / 25.4))
    ax_e = fig.add_axes([0.12, 0.62, 0.34, 0.34])
    ax_t = fig.add_axes([0.51, 0.62, 0.47, 0.34])
    ax_s = fig.add_axes([0.08, 0.105, 0.89, 0.39])
    ax_e.plot(energy['fields'] * 1e3, energy['energies'], lw=0.8)
    ax_e.set(xlabel='B (mT)', ylabel='Energy (GHz)', xlim=(fields[0], fields[-1]))
    ax_e.xaxis.set_major_locator(MaxNLocator(3))
    ax_e.yaxis.set_major_locator(MaxNLocator(4))
    ax_e.spines[['top', 'right']].set_visible(False)
    text = [[f'{value:.1e}' for value in row] for row in values]
    colors = np.where(values < 1e-2, '#c8f0c8', np.where(values < 1e-1, '#fff2b3', '#f7c6c6'))
    table = ax_t.table(cellText=text, cellColours=colors.tolist(),
                       rowLabels=[str(int(n)) for n in meshes],
                       colLabels=[f'{tol:.0e}' for tol in tolerances],
                       cellLoc='center', rowLoc='center', bbox=[0.10, 0.03, 0.90, 0.94])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.4)
        cell.PAD = 0.025
    ax_t.set_axis_off()
    ax_s.plot(fields, reference / amplitude, color='black', lw=1.5, label='EasySpin')
    ax_s.plot(fields, scale * best / amplitude, color='#d62728', lw=0.9, label='MaRs')
    ax_s.set(xlabel='B (mT)', xlim=(fields[0], fields[-1]))
    ax_s.spines[['top', 'right', 'left']].set_visible(False)
    ax_s.set_yticks([])
    ax_s.tick_params(axis='x', top=False, bottom=True)
    ax_s.legend(frameon=False, loc='upper right', ncol=2, handlelength=1.7)
    folder = ROOT / ('supplied_figures' if allow_supplied else 'figures')
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f'example_{k}.eps', format='eps')
    fig.savefig(folder / f'example_{k}.png', dpi=200)
    plt.close(fig)
    return {'example': k, 'scale_mars_to_easyspin': float(scale),
            'relative_l2_to_easyspin': float(np.linalg.norm(scale * best - reference) / np.linalg.norm(reference)),
            'provenance': provenance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-supplied', action='store_true')
    args = parser.parse_args()
    results = [make_figure(k, args.allow_supplied) for k in range(1, 5)]
    folder = ROOT / ('supplied_figures' if args.allow_supplied else 'figures')
    (folder / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
