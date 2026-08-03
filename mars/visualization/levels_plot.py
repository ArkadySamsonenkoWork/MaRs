import typing as tp
from enum import Enum

from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..spin_model import MultiOrientedSample
from .. import constants


class EnergyUnits(Enum):
    Hz = "Hz"
    CM_1 = "cm-1"
    GHz = "GHz"
    MHz = "MHz"
    K = "K"
    T_e = "Tesla"


def unit_converter(energy: torch.Tensor, energy_units: EnergyUnits):
    """
    Convert energy from Hz to the specified unit.

    :param energy: Energy values in hertz (Hz), as a torch.Tensor.
    :param energy_units: Target unit for conversion.
    :return: Energy tensor converted to the requested unit.
    """
    if energy_units == EnergyUnits.Hz:
        return energy
    elif energy_units == EnergyUnits.CM_1:
        return constants.unit_converter(energy, "Hz_to_cm-1")
    elif energy_units == EnergyUnits.T_e:
        return constants.unit_converter(energy, "Hz_to_T_e")
    elif energy_units == EnergyUnits.K:
        return constants.unit_converter(energy, "Hz_to_K")
    elif energy_units == EnergyUnits.GHz:
        return energy.mul_(1e-9)
    elif energy_units == EnergyUnits.MHz:
        return energy.mul_(1e-6)


def plot_energy_system(sample: MultiOrientedSample,
                       B_range: tuple[float, float],
                       levels: tp.Optional[list[int]] = None,
                       saved_order: bool = False,
                       orientation_idx: int = 0,
                       energy_units: tp.Union[str, EnergyUnits] = EnergyUnits.Hz,
                       ) -> None:
    """
    Plot energy levels of a spin system as a function of magnetic field.

    This function computes and visualizes the eigenenergies of the spin Hamiltonian
    over a specified magnetic field range along the z-axis. Optionally, it can preserve
    the physical identity of energy levels across field values using eigenvector overlap
    tracking (avoiding level crossings in the plot).

    :param sample: A multi-oriented spin system sample. It is assumed that magnetic field is directed along z-axis.
    :param B_range: Magnetic field range [B_min, B_max] in tesla (T).
    :param levels: List of level indices to plot. If None, all levels are plotted.
    :param saved_order: If True, eigenvector overlap tracking is used to preserve the physical identity
                        of energy levels across field values (prevents color swapping at crossings).
                        If False, levels are plotted in ascending energy order at each field point.
    :param orientation_idx: The index of orientation which energy system should be plotted
    :param energy_units: Unit for energy axis. Must be a member of EnergyUnits.
    :return: None
    """
    if isinstance(energy_units, str):
        energy_units = EnergyUnits(energy_units)

    num_field_points = 200
    F, _, _, Gz = sample.get_hamiltonian_terms()
    B = torch.linspace(B_range[0], B_range[1], num_field_points, device=sample.device, dtype=sample.dtype)
    H = F[orientation_idx] + B.unsqueeze(-1).unsqueeze(-1) * Gz[orientation_idx]
    energies, vecs = torch.linalg.eigh(H)

    energies = unit_converter(energies, energy_units)

    if levels is None:
        levels = list(range(sample.spin_system_dim))

    vecs = vecs.cpu().numpy()
    energies = energies.cpu().numpy()
    B = B.cpu().numpy()

    if saved_order:
        energies = get_saved_order(energies, vecs)

    plt.figure(figsize=(6, 4))
    for i in levels:
        plt.plot(B.squeeze(), energies[:, i], linewidth=1.2)
    plt.xlabel("Field (T)")
    plt.ylabel(f"Energy ({energy_units.value})")


def get_saved_order(energies: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """
    :param energies: The energies of the levels.

    The shape is [num_points, spin_dimension]
    :param vecs: The eigen vectors of the levels. The shape is [num_points, spin_dimension, spin_dimension]
    :return: tracked_eps: energies in saved order. The shape is [num_points, spin_dimension]
    """
    nB, dim = energies.shape
    tracked_eps = np.zeros_like(energies)
    tracked_eps[0] = energies[0]

    tracked_vecs = np.zeros_like(vecs)
    tracked_vecs[0] = vecs[0]

    for b in range(1, nB):
        prev_vecs = tracked_vecs[b - 1]
        curr_vecs = vecs[b]

        overlap = np.abs(prev_vecs.conj().T @ curr_vecs)

        row_idx, col_idx = linear_sum_assignment(-overlap)

        tracked_eps[b, row_idx] = energies[b, col_idx]
        tracked_vecs[b] = curr_vecs[:, col_idx]

    return tracked_eps
