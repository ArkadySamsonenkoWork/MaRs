import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, root_dir)

import time
import torch
import typing as tp
from mars import spin_model, mesher, constants


def create_5_electrons_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with five coupled S=1/2 electrons.

    Models a tetraradical system. Hilbert space dimension = 16 states.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Use coarse mesh (e.g., (30,30)) for initial tests.
    device : torch.device, optional
        Computation device. GPU strongly recommended.
    dtype : torch.dtype, optional
        Floating point precision. float32 acceptable for timing tests.

    Returns
    -------
    SolidSample
    """
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.02), device=device, dtype=dtype)
    exchange_val = constants.unit_converter(0.3, "cm-1_to_Hz")
    exchange = spin_model.Interaction(exchange_val, device=device, dtype=dtype)

    interactions = []

    pair_idx = 0
    dipolar_vals = (500e6, 50e6)
    for i in range(5):
        for j in range(i + 1, 5):
            dipolar = spin_model.DEInteraction(dipolar_vals, device=device, dtype=dtype)
            interactions.append((i, j, exchange + dipolar))
            pair_idx += 1

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5, 0.5, 0.5, 0.5],
        g_tensors=[g_tensor] * 5,
        electron_electron=interactions,
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.001,
        lorentz=0.001,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_2_electrons_2_nuclei_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with two electrons coupled with two nitrogen nuclei.

    Models a system where both electrons share hyperfine coupling to one ¹⁴N
    nucleus (I=1). Hilbert space dimension = 12 states

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh for powder averaging.
    device : torch.device, optional
        Computation device.
    dtype : torch.dtype, optional
        Floating point precision.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.00, 2.001, 2.002), device=device, dtype=dtype)

    # ¹⁴N hyperfine tensor (anisotropic)
    A_tensor = spin_model.Interaction((40e6, 40e6, 40e6), device=device, dtype=dtype)

    exchange = spin_model.Interaction(
        constants.unit_converter(1.0, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )
    dipolar = spin_model.DEInteraction([20e6, 10e6], device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5],
        nuclei=["14N", "14N"],
        g_tensors=[g_tensor, g_tensor],
        electron_nuclei=[
            (0, 0, A_tensor),  # e0 coupled to N
            (1, 0, A_tensor),  # e1 coupled to same N
            (0, 1, A_tensor),  # e0 coupled to N
            (1, 1, A_tensor)  # e1 coupled to same N
        ],
        electron_electron=[(0, 1, exchange + dipolar)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=None,
        gauss=0.0001,
        lorentz=0.0001,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_1_high_spin_electron_1_nuclei_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with one high-spin electron (S=5/2) coupled to ⁵⁵Mn nucleus.

    Models Mn(II) ion system with zero-field splitting and hyperfine coupling.
    Hilbert space dimension = 36 states. Computationally intensive due to
    large matrix diagonalization requirements.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    # Anisotropic g-tensor for Mn(II)
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.02), device=device, dtype=dtype)

    # Isotropic ⁵⁵Mn hyperfine (I=5/2)
    A_hf = spin_model.Interaction(300e6, device=device, dtype=dtype)  # 300 MHz

    zfs = spin_model.DEInteraction(
        [200e6, 10e6],
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[5 / 2],  # S = 5/2
        nuclei=["55Mn"],
        g_tensors=[g_tensor],
        electron_nuclei=[(0, 0, A_hf)],
        electron_electron=[(0, 0, zfs)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_2_middle_spin_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with two Cobalt spins

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.02), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [constants.unit_converter(10.0, "cm-1_to_Hz"), constants.unit_converter(0.0, "cm-1_to_Hz")],
        device=device,
        dtype=dtype
    )

    dipolar = spin_model.DEInteraction(
        [0e6, 0e6],
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        constants.unit_converter(0.001, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[3/2, 3/2],
        g_tensors=[g_tensor, g_tensor],
        electron_electron=[(0, 0, zfs), (0, 0, zfs), (0, 1, dipolar + exchange)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_3_middle_spin_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with 3 Cobalt spins

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.02), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [constants.unit_converter(-100.0, "cm-1_to_Hz"), constants.unit_converter(0.0, "cm-1_to_Hz")],
        device=device,
        dtype=dtype
    )

    dipolar = spin_model.DEInteraction(
        [100e6, 10e6],
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        constants.unit_converter(0.001, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[3/2, 1/2, 3/2],
        g_tensors=[g_tensor, g_tensor, g_tensor],
        electron_electron=[(0, 0, zfs), (2, 2, zfs), (0, 1, dipolar + exchange), (2, 1, dipolar + exchange)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_heterospin_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with 2 Cobalt spins exchange connected to radical spin

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.10), device=device, dtype=dtype)
    g_tensor_rad = spin_model.Interaction((2.000, 2.002, 2.004), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [constants.unit_converter(-100.0, "cm-1_to_Hz"), constants.unit_converter(-10.0, "cm-1_to_Hz")],
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        constants.unit_converter(0.01, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[3/2, 1/2, 3/2],
        g_tensors=[g_tensor, g_tensor_rad, g_tensor],
        electron_electron=[(0, 0, zfs), (2, 2, zfs), (0, 1, exchange), (1, 2, exchange)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_heterospin_sample_freq_domain(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with 2 Cobalt spins exchange connected to radical spin
    The spectra are created in frequency domain

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.00, 2.01, 2.10), device=device, dtype=dtype)
    g_tensor_rad = spin_model.Interaction((2.000, 2.002, 2.004), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [constants.unit_converter(-100.0, "cm-1_to_Hz"), constants.unit_converter(-10.0, "cm-1_to_Hz")],
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        constants.unit_converter(0.01, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[3/2, 1/2, 3/2],
        g_tensors=[g_tensor, g_tensor_rad, g_tensor],
        electron_electron=[(0, 0, zfs), (2, 2, zfs), (0, 1, exchange), (1, 2, exchange)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=2e10,
        lorentz=2e10,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_tripletfission_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample of 2 triplets with fission

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.000, 2.002, 2.004), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [1000.0e6, 100.0e6],
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        1.0e11,
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[1.0, 1.0],
        g_tensors=[g_tensor, g_tensor],
        electron_electron=[(0, 0, zfs), (1, 1, zfs), (0, 1, exchange)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample