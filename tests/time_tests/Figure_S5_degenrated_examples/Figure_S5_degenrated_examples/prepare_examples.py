import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import mars
from mars import visualization
import degenerate_examples as examples

ROOT = Path(__file__).resolve().parent
MESHES = [20, 40, 60, 80]
TOLERANCES = [1e-3, 1e-4, 1e-5, 1e-7]
THRESHOLD = 1e-4


def energy_data(sample, fields, destination):
    visualization.plot_energy_system(
        sample=sample, B_range=(float(fields[0]), float(fields[-1])),
        levels=None, orientation_idx=0, energy_units='GHz')
    fig = plt.gcf()
    lines = fig.axes[0].lines
    np.savez(destination, fields=lines[0].get_xdata(),
             energies=np.column_stack([line.get_ydata() for line in lines]))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supplied', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for k, build in examples.EXAMPLES.items():
        if args.supplied:
            source = args.supplied.resolve()
            data = np.load(source / f'example_{k}_mars.npz')
            np.savez(ROOT / f'example_{k}_mars.npz', **{name: data[name] for name in data.files})
            old = np.loadtxt(source / f'example_{k}_easyspin.csv', delimiter=',')
            if old.shape != (len(data['fields']), 4):
                raise ValueError('Expected the supplied field column and three reference spectra')
            np.savetxt(ROOT / f'example_{k}_easyspin.csv', old[:, [0, 3]], delimiter=',')
            sample, _, _ = build(80, 1e-7, THRESHOLD, save_path=None)
            fields = data['fields']
            provenance = {'status': 'supplied-unverified',
                          'reference': 'last column of supplied CSV; not rerun in MATLAB',
                          'energy_model': 'current Python model; supplied Example 2 MAT has a different DFrame'}
        else:
            spectra = []
            with torch.no_grad():
                for n in MESHES:
                    row = []
                    for tol in TOLERANCES:
                        sample, creator, fields = build(n, tol, THRESHOLD, save_path=None)
                        row.append(creator(sample, fields).detach().cpu().numpy())
                    spectra.append(row)
                np.savez(ROOT / f'example_{k}_mars.npz',
                         fields=fields.detach().cpu().numpy(), meshes=MESHES,
                         tolerances=TOLERANCES, spectra=np.array(spectra))
                sample, creator, fields = build(80, 1e-7, THRESHOLD,
                                               save_path=str(ROOT / f'example_{k}.mat'))
            for suffix in ['easyspin.csv', 'easyspin_settings.mat']:
                (ROOT / f'example_{k}_{suffix}').unlink(missing_ok=True)
            provenance = {'status': 'mars-recomputed',
                          'energy_model': 'current Python model',
                          'reference': 'run easyspin_degenerate_examples.m before plotting'}
        energy_data(sample, fields, ROOT / f'example_{k}_energy.npz')
        (ROOT / f'example_{k}_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'Prepared example {k}', flush=True)


if __name__ == '__main__':
    main()
