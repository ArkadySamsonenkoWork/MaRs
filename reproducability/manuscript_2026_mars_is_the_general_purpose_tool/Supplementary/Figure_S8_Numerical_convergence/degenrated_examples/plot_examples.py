from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from mars import visualization

ROOT = Path(__file__).resolve().parent


def load_mars(k):
    data = np.load(ROOT / f"example_{k}_mars.npz")
    spectra = {(int(n), float(r)): data["spectra"][i, j]
               for i, n in enumerate(data["meshes"])
               for j, r in enumerate(data["tolerances"])}
    return spectra, data["fields"]


def plot_example(k, sample, spectra, fields):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10,
                         "ps.fonttype": 42, "axes.unicode_minus": False})
    meshes = sorted({n for n, r in spectra})
    tolerances = sorted({r for n, r in spectra}, reverse=True)
    best = spectra[(80, 1e-7)]
    values = np.array([[np.linalg.norm(spectra[(n, r)] - best) /
                        np.linalg.norm(best) for r in tolerances] for n in meshes])
    easyspin = np.loadtxt(ROOT / f"example_{k}_easyspin.csv", delimiter=",")
    fields = np.asarray(fields)
    if easyspin.shape != (len(fields), 2):
        raise ValueError("EasySpin CSV must contain field and spectrum columns")
    if not np.allclose(easyspin[:, 0], fields * 1e3):
        raise ValueError("MaRs and EasySpin field axes must match")
    reference = easyspin[:, 1]
    scale = np.dot(reference, best) / np.dot(best, best)
    amplitude = np.max(np.abs(reference))
    visualization.plot_energy_system(sample=sample,
                                     B_range=(float(fields[0]), float(fields[-1])),
                                     levels=None, orientation_idx=0, energy_units="GHz")
    energy_figure = plt.gcf()
    lines = energy_figure.axes[0].lines
    energy = {"fields": lines[0].get_xdata(),
              "energies": np.column_stack([line.get_ydata() for line in lines])}
    plt.close(energy_figure)
    fields = fields * 1e3
    fig = plt.figure(figsize=(160 / 25.4, 55 / 25.4))
    ax_e = fig.add_axes([0.09, 0.26, 0.18, 0.65])
    ax_t = fig.add_axes([0.32, 0.26, 0.34, 0.65])
    ax_s = fig.add_axes([0.71, 0.26, 0.27, 0.65])
    ax_e.plot(energy["fields"] * 1e3, energy["energies"], color="0.30", lw=0.65)
    ax_e.set(xlabel="B (mT)", ylabel="E(GHz)", xlim=(fields[0], fields[-1]))
    ax_e.xaxis.set_major_locator(MaxNLocator(2))
    ax_e.yaxis.set_major_locator(MaxNLocator(4))
    ax_e.spines[["top", "right"]].set_visible(False)
    text = [[f"{100 * value:.2g}" for value in row] for row in values]
    colors = np.where(values < 1e-2, "#c8f0c8", np.where(values < 1e-1, "#fff2b3", "#f7c6c6"))
    table = ax_t.table(cellText=text, cellColours=colors.tolist(),
                       rowLabels=[str(int(n)) for n in meshes],
                       colLabels=[f"{tol:.0e}".replace("e-0", "e-") for tol in tolerances],
                       cellLoc="center", rowLoc="center", bbox=[0.10, 0.10, 0.85, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.4)
        cell.PAD = 0.025
    ax_t.set_axis_off()
    ax_s.plot(fields, reference / amplitude, color="black", lw=1.5, label="EasySpin")
    ax_s.plot(fields, scale * best / amplitude, color="#d62728", lw=0.9, label="MaRs")
    ax_s.set(xlabel="B (mT)", xlim=(fields[0], fields[-1]))
    ax_s.spines[["top", "right", "left"]].set_visible(False)
    ax_s.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=4))
    ax_s.set_ylim(-1.08, 1.65)
    ax_s.set_yticks([])
    ax_s.tick_params(axis="x", top=False, bottom=True)
    ax_s.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.0, borderpad=0.1, labelspacing=0.1)
    fig.savefig(ROOT / f"example_{k}.eps", format="eps")
    fig.savefig(ROOT / f"example_{k}.png", dpi=200)
    plt.show()
    return fig


if __name__ == "__main__":
    import degenerate_examples as examples

    for k in range(1, 5):
        spectra, fields = load_mars(k)
        sample, _, _ = examples.EXAMPLES[k](80, 1e-7, 1e-4, save_path=None)
        plot_example(k, sample, spectra, fields)
