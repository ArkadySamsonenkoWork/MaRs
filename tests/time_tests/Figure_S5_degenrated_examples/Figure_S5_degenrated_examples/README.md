The example docstrings describe the spin topology, tensor symmetry, coupling classes, molecular-frame orientations, line broadening, and returned objects. Model parameter values remain in executable code.

Files

- `degenerate_examples.py`: the four model constructors, with formal docstrings.
- `prepare_examples.py`: recomputes the MaRs convergence grid, exports each current model to MATLAB, and obtains energy-level curves through `mars.visualization.plot_energy_system`.
- `easyspin_degenerate_examples.m`: performs one EasySpin calculation per example, using `GridSize = [80 0]`, `ModellingAccuracy = 1e-7`, and the shared intensity threshold. The CSV contains only the field in mT and that spectrum. A settings file records the actual inputs.
- `plot_examples.py`: produces one three-panel EPS and PNG per example. There are no comments, docstrings, panel letters, titles, captions, or table annotations in this plotting script or its figures.
- `degenerate_examples.ipynb`: a clean entry point for preparation and plotting.

Regenerate the comparison

Use the MaRs installation corresponding to your project. From this directory:

```sh
python prepare_examples.py
```

Run `easyspin_degenerate_examples.m` in MATLAB with EasySpin on its path, then:

```sh
python plot_examples.py
```

The resulting files are `figures/example_1.eps` through `figures/example_4.eps`. Preparation removes previous EasySpin outputs, so each freshly simulated model requires a fresh EasySpin calculation. Plotting checks the recorded EasySpin grid, tolerance, model parameters, experimental parameters, and field axis. It compares only the MaRs result at mesh 80 and tolerance 1e-7 with the single EasySpin reference. It does not compare lower-resolution EasySpin cases.

Figure conventions

Each separate figure is 160 mm wide and 126 mm high. All text is 10 points; fonts are embedded in the EPS. The fixed page boundary is retained during export. The energy panel covers exactly the supplied field sweep and uses orientation index zero from the MaRs sample. All energy levels are included; degenerate curves may overlap. This is a single-orientation energy diagram, while the spectrum is a powder average.

The table contains relative L2 differences from the MaRs result at the largest mesh and smallest tolerance. Numeric row headers are mesh frequencies; numeric column headers are solver tolerances. Every value occupies its own cell. Green denotes a difference below one percent, yellow below ten percent, and red the remaining values. The table has no title, prose labels, colour bar, or explanatory annotations.

The spectral panel shows the finest MaRs result and the single EasySpin reference. One least-squares multiplicative factor aligns MaRs amplitudes to EasySpin; both curves are then divided by the maximum absolute EasySpin amplitude. No field shift or offset is fitted. Only the bottom x-axis is visible. The scale and relative L2 comparison are recorded in `comparison.json`.

Included figures and limitations

`supplied_figures/` contains four EPS files and matching PNG previews generated here. Their energy levels were calculated from the current Python models using MaRs. Their spectra and convergence tables reuse the supplied NPZ data, and only the last reference column from each supplied EasySpin CSV is used. These reference files have not been rerun or independently verified in MATLAB. The uploaded CSVs contain a field column even though the uploaded MATLAB script does not export one, so their exact generation history cannot be confirmed.

The supplied Example 2 MAT file uses a different final zero-field-splitting frame angle from the Python constructor. The included `example_2.mat` was regenerated from the Python model, as were the other model exports. The included Example 2 spectrum is therefore retained as supplied and must be regenerated before interpreting it together with the current energy diagram. No physical parameters in the Python constructors were changed.

The included data are marked as supplied and unverified in each provenance JSON. To reproduce these supplied-data previews explicitly, use:

```sh
python plot_examples.py --allow-supplied
```

After recomputation, use the ordinary plotting command. MATLAB and EasySpin were unavailable during this revision, so no fresh EasySpin simulation or runtime validation of the MATLAB script was performed.

Validation performed here: all four model constructors and current MATLAB exports, all four energy datasets, one finite MaRs powder-spectrum smoke calculation, Python syntax, numeric-free example docstrings, visual inspection of the four layouts, and EPS page-width verification.
