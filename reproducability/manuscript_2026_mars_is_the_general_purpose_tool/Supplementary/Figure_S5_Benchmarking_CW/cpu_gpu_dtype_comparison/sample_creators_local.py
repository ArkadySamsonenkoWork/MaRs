import sys
import os
import math
import random

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, root_dir)

import time
import torch
import typing as tp
from mars import spin_model, mesher, constants


def _get_random_principle_values(
        g_mean: tp.Tuple[float, float, float],
        g_std: tp.Tuple[float, float, float],
        disable_randomness: bool = False
) -> tp.Tuple[float, float, float]:
    """
    Generate principle values for g-tensor or similar interactions.

    When disable_randomness=True, uses fixed offsets: [-0.73, -0.47, 0.0] * std
    This gives (2.04, 2.08, 2.15) for g_mean=(2.15, 2.15, 2.15), g_std=(0.15, 0.15, 0.15)
    """
    if disable_randomness:
        offsets = [-0.73, -0.47, 0.0]
        gx = g_mean[0] + g_std[0] * offsets[0]
        gy = g_mean[1] + g_std[1] * offsets[1]
        gz = g_mean[2] + g_std[2] * offsets[2]
    else:
        gx = g_mean[0] + g_std[0] * 2 * (random.random() - 0.5)
        gy = g_mean[1] + g_std[1] * 2 * (random.random() - 0.5)
        gz = g_mean[2] + g_std[2] * 2 * (random.random() - 0.5)
    return gx, gy, gz


def _get_random_single_value(
        g_mean: float,
        g_std: float,
        disable_randomness: bool = False
) -> float:
    """Generate a single random value with fixed offset when disabled."""
    if disable_randomness:
        return g_mean + g_std * (-0.73)
    else:
        return g_mean + g_std * 2 * (random.random() - 0.5)


def _get_random_two_values(
        g_mean: tp.Tuple[float, float],
        g_std: tp.Tuple[float, float],
        disable_randomness: bool = False
) -> tp.Tuple[float, float]:
    """Generate two random values with fixed offsets when disabled."""
    if disable_randomness:
        offsets = [-0.73, -0.47]
        gx = g_mean[0] + g_std[0] * offsets[0]
        gy = g_mean[1] + g_std[1] * offsets[1]
    else:
        gx = g_mean[0] + g_std[0] * 2 * (random.random() - 0.5)
        gy = g_mean[1] + g_std[1] * 2 * (random.random() - 0.5)
    return gx, gy


def create_electron_nucleus(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_nuclei: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with S=1/2 electron coupled with N=1 nuclei. The S=1/2 is close to isotropic, N=1 is anisotropic.
    """
    g_mean = (2.01, 2.01, 2.01)
    g_std = (0.01, 0.01, 0.01)
    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype)

    interactions = []

    A_mean = (30*1e6, 30*1e6, 70*1e6)
    A_std = (5*1e6, 5*1e6, 5*1e6)
    if num_nuclei == 0:
        spin_sys = spin_model.SpinSystem(
            electrons=[0.5],
            g_tensors=[g_tensor],
            device=device,
            dtype=dtype
        )
    else:
        for i in range(num_nuclei):
            hyperfine = spin_model.Interaction(
                _get_random_principle_values(A_mean, A_std, disable_randomness=disable_randomness),
                device=device, dtype=dtype
            )
            interactions.append((0, i, hyperfine))

        spin_sys = spin_model.SpinSystem(
            electrons=[0.5],
            nuclei=["14N"] * num_nuclei,
            electron_nuclei=interactions,
            g_tensors=[g_tensor],
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


def create_electron_anisotropic_chain(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_electrons: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with a lot of S=1/2 electrons coupled in chain with isotropic J-exchange.
    """
    g_mean = (2.15, 2.15, 2.20)
    g_std = (0.05, 0.05, 0.05)
    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype
    )

    if num_electrons == 1:
        spin_sys = spin_model.SpinSystem(
            electrons=[0.5] * num_electrons,
            g_tensors=[g_tensor],
            device=device,
            dtype=dtype
        )
    else:
        interactions = []
        g_tensors_additional = []
        J_mean = constants.unit_converter(50.0, "cm-1_to_Hz")
        J_std = constants.unit_converter(5.0, "cm-1_to_Hz")
        for i in range(num_electrons-1):
            exchange = spin_model.Interaction(
                _get_random_single_value(J_mean, J_std,
                                         disable_randomness=disable_randomness), device=device, dtype=dtype
            )
            interactions.append((i, i+1, exchange))
            g_tensors_additional.append(
                spin_model.Interaction(
                    _get_random_principle_values(g_mean, g_std,
                                                 disable_randomness=disable_randomness), device=device, dtype=dtype
                )
            )

        spin_sys = spin_model.SpinSystem(
            electrons=[0.5] * num_electrons,
            electron_electron=interactions,
            g_tensors=[g_tensor] + g_tensors_additional,
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


def create_heterospin_connected_to_Co(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_electrons: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with Co spin connected to S=1/2 spins via exchange interaction
    """
    g_mean = (2.15, 2.15, 2.20)
    g_std = (0.05, 0.05, 0.05)
    D_mean = constants.unit_converter(0, "cm-1_to_Hz")
    E_mean = constants.unit_converter(0, "cm-1_to_Hz")

    D_std = constants.unit_converter(20, "cm-1_to_Hz")
    E_std = constants.unit_converter(1, "cm-1_to_Hz")

    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype
    )
    zfs = spin_model.DEInteraction(
        _get_random_two_values((D_mean, E_mean), (D_std, E_std), disable_randomness=disable_randomness),
                               device=device, dtype=dtype
    )

    if num_electrons == 1:
        spin_sys = spin_model.SpinSystem(
            electrons=[3/2] * num_electrons,
            g_tensors=[g_tensor],
            electron_electron=[(0, 0, zfs)],
            device=device,
            dtype=dtype)
    else:
        interactions = []
        g_tensors_additional = []
        J_mean = constants.unit_converter(5.0, "cm-1_to_Hz")
        J_std = constants.unit_converter(2.0, "cm-1_to_Hz")
        for i in range(num_electrons-1):
            exchange = spin_model.Interaction(
                _get_random_single_value(J_mean, J_std,
                                         disable_randomness=disable_randomness), device=device, dtype=dtype
            )
            interactions.append((0, i+1, exchange))
            g_tensors_additional.append(
                spin_model.Interaction(
                    _get_random_principle_values(g_mean, g_std,
                                                 disable_randomness=disable_randomness), device=device, dtype=dtype
                )
            )

        spin_sys = spin_model.SpinSystem(
            electrons=[3/2] + [1/2] * (num_electrons - 1),
            electron_electron=[(0, 0, zfs)] + interactions,
            g_tensors=[g_tensor] + g_tensors_additional,
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


def create_heterospin_connected_to_Co_high_values(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_electrons: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with Co spin connected to S=1/2 spins via exchange interaction
    """
    g_mean = (2.15, 2.15, 2.20)
    g_std = (0.05, 0.05, 0.05)
    D_mean = constants.unit_converter(0, "cm-1_to_Hz")
    E_mean = constants.unit_converter(0, "cm-1_to_Hz")

    D_std = constants.unit_converter(100, "cm-1_to_Hz")
    E_std = constants.unit_converter(1, "cm-1_to_Hz")

    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype
    )
    zfs = spin_model.DEInteraction(
        _get_random_two_values((D_mean, E_mean), (D_std, E_std), disable_randomness=disable_randomness),
                               device=device, dtype=dtype
    )

    if num_electrons == 1:
        spin_sys = spin_model.SpinSystem(
            electrons=[3/2] * num_electrons,
            g_tensors=[g_tensor],
            electron_electron=[(0, 0, zfs)],
            device=device,
            dtype=dtype)
    else:
        interactions = []
        g_tensors_additional = []
        J_mean = constants.unit_converter(50.0, "cm-1_to_Hz")
        J_std = constants.unit_converter(5.0, "cm-1_to_Hz")
        for i in range(num_electrons-1):
            exchange = spin_model.Interaction(
                _get_random_single_value(J_mean, J_std,
                                         disable_randomness=disable_randomness), device=device, dtype=dtype
            )
            interactions.append((0, i+1, exchange))
            g_tensors_additional.append(
                spin_model.Interaction(
                    _get_random_principle_values(g_mean, g_std,
                                                 disable_randomness=disable_randomness), device=device, dtype=dtype
                )
            )

        spin_sys = spin_model.SpinSystem(
            electrons=[3/2] + [1/2] * (num_electrons - 1),
            electron_electron=[(0, 0, zfs)] + interactions,
            g_tensors=[g_tensor] + g_tensors_additional,
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



def create_electron_DE_chain(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_electrons: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with a lot of S=1/2 electrons coupled in chain with isotropic D-E values.
    """
    g_mean = (2.01, 2.01, 2.01)
    g_std = (0.01, 0.01, 0.01)
    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype
    )

    if num_electrons == 1:
        spin_sys = spin_model.SpinSystem(
            electrons=[0.5] * num_electrons,
            g_tensors=[g_tensor],
            device=device,
            dtype=dtype
        )
    else:
        interactions = []
        g_tensors_additional = []
        D_E_mean = (500e6, 50e6)
        D_E_std = (300e6, 30e6)

        for i in range(num_electrons-1):
            exchange = spin_model.DEInteraction(
                _get_random_two_values(D_E_mean, D_E_std,
                                       disable_randomness=disable_randomness), device=device, dtype=dtype)
            interactions.append((i, i+1, exchange))
            g_tensors_additional.append(
                spin_model.Interaction(_get_random_principle_values(
                    g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype)
            )

        spin_sys = spin_model.SpinSystem(
            electrons=[0.5] * num_electrons,
            electron_electron=interactions,
            g_tensors=[g_tensor] + g_tensors_additional,
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


def create_electron_batch_size(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        batch_size: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with a lot of S=1/2 electrons coupled with N=1 nuclei
    """
    g_mean = torch.tensor([2.01, 2.01, 2.01], dtype=dtype, device=device)
    g_std = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, device=device)
    g_values = g_mean + g_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)

    A_mean = torch.tensor([30e6, 30e6, 70e6], dtype=dtype, device=device)
    A_std = torch.tensor([5e6, 5e6, 5e6], dtype=dtype, device=device)
    A_values = A_mean + A_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)

    g_tensor = spin_model.Interaction(g_values, device=device, dtype=dtype)
    hyperfine = spin_model.Interaction(A_values, device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5],
        nuclei=["14N"],
        electron_nuclei=[(0, 0, hyperfine)],
        g_tensors=[g_tensor],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        lorentz=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample
    
  
def create_electron_batch_size_2N(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        batch_size: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with a lot of S=1/2 electrons coupled with N=1 nuclei
    """
    g_mean = torch.tensor([2.01, 2.01, 2.01], dtype=dtype, device=device)
    g_std = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, device=device)
    g_values = g_mean + g_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)

    A_mean = torch.tensor([30e6, 30e6, 70e6], dtype=dtype, device=device)
    A_std = torch.tensor([5e6, 5e6, 5e6], dtype=dtype, device=device)
    A_values = A_mean + A_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)

    g_tensor = spin_model.Interaction(g_values, device=device, dtype=dtype)
    hyperfine_1 = spin_model.Interaction(A_values, device=device, dtype=dtype)
    
    A_mean = torch.tensor([30e6, 30e6, 70e6], dtype=dtype, device=device)
    A_std = torch.tensor([5e6, 5e6, 5e6], dtype=dtype, device=device)
    A_values = A_mean + A_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)
    hyperfine_2 = spin_model.Interaction(A_values, device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5],
        nuclei=["14N", "14N"],
        electron_nuclei=[(0, 0, hyperfine_1), (0, 1, hyperfine_1)],
        g_tensors=[g_tensor],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        lorentz=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_relaxation_batch_size(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        batch_size: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with S=1 electron
    """
    g_mean = torch.tensor([2.01, 2.01, 2.01], dtype=dtype, device=device)
    g_std = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, device=device)
    g_values = g_mean + g_std * 2 * (torch.rand(batch_size, 3, dtype=dtype, device=device) - 0.5)
    g_tensor = spin_model.Interaction(g_values, device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[1.0],
        g_tensors=[g_tensor],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        gauss=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        lorentz=torch.tensor([0.001] * batch_size, dtype=dtype, device=device),
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_relaxation_coupled(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        num_electrons: int = 1,
        disable_randomness: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with S=1 electron
    """
    g_mean = (2.15, 2.15, 2.20)
    g_std = (0.05, 0.05, 0.05)
    g_tensor = spin_model.Interaction(
        _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness), device=device, dtype=dtype
    )

    if num_electrons == 1:
        spin_sys = spin_model.SpinSystem(
            electrons=[1],
            g_tensors=[g_tensor],
            device=device,
            dtype=dtype
        )
    else:
        interactions = []
        g_tensors_additional = []
        for i in range(num_electrons-1):
            interactions = []
            J_mean = constants.unit_converter(0.5, "cm-1_to_Hz")
            J_std = constants.unit_converter(0.05, "cm-1_to_Hz")

            exchange = spin_model.Interaction(
                _get_random_single_value(J_mean, J_std, disable_randomness=disable_randomness),
                device=device, dtype=dtype)

            g_tensors_additional.append(
                spin_model.Interaction(
                    _get_random_principle_values(g_mean, g_std, disable_randomness=disable_randomness),
                    device=device, dtype=dtype)
            )
            interactions.append((i, i+1, exchange))

        spin_sys = spin_model.SpinSystem(
            electrons=[1] + [0.5] * (num_electrons-1),
            electron_electron=interactions,
            g_tensors=[g_tensor] + g_tensors_additional,
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
