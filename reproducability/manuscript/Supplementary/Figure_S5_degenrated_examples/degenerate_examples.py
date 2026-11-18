import typing as tp

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..")))


import mars
from mars import spin_model, spectra_manager, mesher, constants
from mars.spectra_manager.utils import ComputationalDetails

dtype = torch.float64
freq = 9.5e9
temperature = 293.0


def _cm_to_hz(value: float) -> float:
    return constants.unit_converter(value, "cm-1_to_Hz")


def _build(spin_system: spin_model.SpinSystem,
           mesh_size: int, r_tol: float, intensity_threshold: float,
           field_range: tuple[float, float],
           n_points: int, gauss: float, lorentz: float,
           save_path: tp.Optional[str],
           chunk_size: tp.Optional[tp.Union[int, str]] = None):

    mesh = mesher.DelaunayMesh(initial_grid_frequency=mesh_size, interpolate=False, dtype=dtype)
    sample = spin_model.SolidSample(spin_system, gauss=gauss, lorentz=lorentz, mesh=mesh, dtype=dtype)
    details = dict(res_field_r_tol=r_tol, intensity_threshold=intensity_threshold)
    if chunk_size is not None:
        details['res_field_chunk_size'] = chunk_size
    spectra_creator = spectra_manager.StationarySpectra(
        freq=freq, sample=sample, temperature=temperature, harmonic=1, dtype=dtype,

        computational_details=ComputationalDetails(**details))

    fields = torch.linspace(field_range[0], field_range[1], n_points, dtype=dtype)

    if save_path is not None:
        mars.save(save_path, sample=sample, spectra_creator=spectra_creator, field=fields, format_type="easyspin")
    return sample, spectra_creator, fields


def example_1(mesh_size: int = 40, r_tol: float = 1e-5, intensity_threshold: float = 1e-4,
              save_path: tp.Optional[str] = "example_1.mat"):
    """Construct a tetrahedral cluster of equivalent electron spins.

    Identical anisotropic Zeeman tensors are assigned to all sites. Each
    electron pair is coupled by the same isotropic exchange interaction and
    the same rhombic anisotropic interaction, whose principal axes are rotated
    relative to the molecular frame. No nuclei are included.

    :param mesh_size: Initial frequency of the noninterpolated Delaunay
        orientation grid.
    :param r_tol: Relative tolerance of the resonance-field solver.
    :param intensity_threshold: Threshold used to retain transition intensities.
    :param save_path: Destination for the EasySpin export. If None, no file
        is exported.
    :return: Powder sample containing the spin Hamiltonian and line broadening,
        equilibrium derivative-spectrum calculator, and magnetic-field sweep
        in tesla.
    :rtype: tuple[SolidSample, StationarySpectra, torch.Tensor]
    """
    g = spin_model.Interaction((2.00, 2.03, 2.06), dtype=dtype)
    coupling = (spin_model.Interaction(_cm_to_hz(0.2), dtype=dtype) +
                spin_model.DEInteraction([3e9, 0.3e9], frame=[0.4, 0.6, 0.3], dtype=dtype))

    pairs = [(i, j, coupling) for i in range(4) for j in range(i + 1, 4)]

    system = spin_model.SpinSystem(electrons=[0.5] * 4, g_tensors=[g] * 4, electron_electron=pairs, dtype=dtype)

    return _build(system, mesh_size, r_tol, intensity_threshold, field_range=(0.150, 0.520), n_points=3000,
                  gauss=1.2e-3, lorentz=0.5e-3, save_path=save_path)


def example_2(mesh_size: int = 40, r_tol: float = 1e-5, intensity_threshold: float = 1e-4,
              save_path: tp.Optional[str] = "example_2.mat"):
    """Construct a triangular cluster with equivalent local anisotropy.

    Identical axial Zeeman tensors and rhombic single-ion zero-field-splitting
    tensors are assigned to the sites. The zero-field-splitting principal axes
    are rotated relative to the molecular frame. Every distinct electron pair
    is coupled by isotropic exchange. No nuclei are included. 

    :param mesh_size: Initial frequency of the noninterpolated Delaunay
        orientation grid.
    :param r_tol: Relative tolerance of the resonance-field solver.
    :param intensity_threshold: Threshold used to retain transition intensities.
    :param save_path: Destination for the EasySpin export. If None, no file
        is exported.
    :return: Powder sample containing the spin Hamiltonian and line broadening,
        equilibrium derivative-spectrum calculator, and magnetic-field sweep
        in tesla.
    :rtype: tuple[SolidSample, StationarySpectra, torch.Tensor]
    """
    g = spin_model.Interaction((2.15, 2.15, 2.20), dtype=dtype)

    zfs = spin_model.DEInteraction([5e9, 0.5e9], frame=[0.3, 0.5, 0.2], dtype=dtype)

    exchange = spin_model.Interaction(_cm_to_hz(0.5), dtype=dtype)

    couplings = [(i, i, zfs) for i in range(3)] + [(i, j, exchange) for i in range(3) for j in range(i + 1, 3)]

    system = spin_model.SpinSystem(electrons=[1.0] * 3, g_tensors=[g] * 3, electron_electron=couplings, dtype=dtype)

    return _build(system, mesh_size, r_tol, intensity_threshold, field_range=(0.100, 0.500), n_points=3000,
                  gauss=1.2e-3, lorentz=0.5e-3, save_path=save_path, chunk_size=None)


def example_3(mesh_size: int = 40, r_tol: float = 1e-5, intensity_threshold: float = 1e-4,
              save_path: tp.Optional[str] = "example_3.mat"):
    """Construct a triangular cluster of equivalent electron spins.

    Identical rhombic Zeeman tensors are assigned to all sites. Every distinct
    electron pair is coupled by the same isotropic exchange interaction and
    the same rhombic anisotropic interaction. The interaction principal axes
    are rotated relative to the molecular frame. No nuclei are included. 

    :param mesh_size: Initial frequency of the noninterpolated Delaunay
        orientation grid.
    :param r_tol: Relative tolerance of the resonance-field solver.
    :param intensity_threshold: Threshold used to retain transition intensities.
    :param save_path: Destination for the EasySpin export. If None, no file
        is exported.
    :return: Powder sample containing the spin Hamiltonian and line broadening,
        equilibrium derivative-spectrum calculator, and magnetic-field sweep
        in tesla.
    :rtype: tuple[SolidSample, StationarySpectra, torch.Tensor]
    """
    g = spin_model.Interaction((2.05, 2.10, 2.25), dtype=dtype)

    coupling = (spin_model.Interaction(_cm_to_hz(0.25), dtype=dtype) +
                spin_model.DEInteraction([4e9, 0.4e9], frame=[0.5, 0.8, 0.2], dtype=dtype))

    pairs = [(i, j, coupling) for i in range(3) for j in range(i + 1, 3)]

    system = spin_model.SpinSystem(electrons=[0.5] * 3, g_tensors=[g] * 3, electron_electron=pairs, dtype=dtype)

    return _build(system, mesh_size, r_tol, intensity_threshold, field_range=(0.150, 0.500), n_points=3000,
                  gauss=1.0e-3, lorentz=0.5e-3, save_path=save_path)


def example_4(mesh_size: int = 40, r_tol: float = 1e-5, intensity_threshold: float = 1e-4,
              save_path: tp.Optional[str] = "example_4.mat"):
    """Construct a central-site cluster with equivalent sites.

    An isotropic Zeeman tensor is assigned to the central site; identical
    anisotropic Zeeman tensors are assigned to the peripheral sites. Both
    central-to-peripheral and peripheral-to-peripheral couplings are included.
    Each coupling class combines isotropic exchange and a rotated rhombic
    anisotropic interaction, with a distinct parameter set for each class.
    No nuclei are included. A

    :param mesh_size: Initial frequency of the noninterpolated Delaunay
        orientation grid.
    :param r_tol: Relative tolerance of the resonance-field solver.
    :param intensity_threshold: Threshold used to retain transition intensities.
    :param save_path: Destination for the EasySpin export. If None, no file
        is exported.
    :return: Powder sample containing the spin Hamiltonian and line broadening,
        equilibrium derivative-spectrum calculator, and magnetic-field sweep
        in tesla.
    :rtype: tuple[SolidSample, StationarySpectra, torch.Tensor]
    """
    g_center = spin_model.Interaction(2.0023, dtype=dtype)
    g_outer = spin_model.Interaction((2.02, 2.05, 2.08), dtype=dtype)

    center_outer = (spin_model.Interaction(_cm_to_hz(0.12), dtype=dtype) +
                    spin_model.DEInteraction([1.5e9, 0.15e9], frame=[0.2, 0.7, 0.4], dtype=dtype))

    outer_outer = (spin_model.Interaction(_cm_to_hz(0.03), dtype=dtype) +
                   spin_model.DEInteraction([0.8e9, 0.08e9], frame=[0.6, 0.3, 0.1], dtype=dtype))

    pairs = [(0, i, center_outer) for i in range(1, 4)] + [(i, j, outer_outer) for i in range(1, 4)
                                                           for j in range(i + 1, 4)]
    system = spin_model.SpinSystem(electrons=[0.5] * 4, g_tensors=[g_center] + [g_outer] * 3,
                                   electron_electron=pairs, dtype=dtype)
    return _build(system, mesh_size, r_tol, intensity_threshold, field_range=(0.260, 0.420), n_points=3000,
                  gauss=1.0e-3, lorentz=0.5e-3, save_path=save_path)


EXAMPLES = {1: example_1, 2: example_2, 3: example_3, 4: example_4}
